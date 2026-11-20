# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.162", "uvicorn>=0.54", "python-multipart>=0.0.33", "openai>=3.51,<4",
#                 "psycopg[binary]>=3.3", "boto3>=1.44", "python-dotenv>=1.2.4"]
# ///
"""The workshop demo, plus AIRLOCK: a privacy + cost gateway in front of Akamai Inference.

    uv run demo/app.py      then open http://localhost:8000

Inference is required. The app uses the database and the bucket once you fill in their lines in .env.
AIRLOCK adds POST /v1/chat: it redacts secrets before they reach the model, caches identical
redacted prompts, and restores the original values in the response. Nothing raw is ever persisted.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import boto3
import psycopg
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(override=True)  # reads .env from the repo folder, or next to this file
for name in ("AKAMAI_BASE_URL", "AKAMAI_API_KEY", "CHAT_MODEL"):
    if not os.environ.get(name):
        sys.exit(f"{name} is not set. On a laptop, add it to .env and run uv run check.py. On LKE, remove the # and "
                 "the space after it from the start of the last four lines of 05-lke/deployment.yaml, then apply it again.")

client = OpenAI(base_url=os.environ["AKAMAI_BASE_URL"], api_key=os.environ["AKAMAI_API_KEY"])
CHAT_MODEL = os.environ["CHAT_MODEL"]
DATABASE = bool(os.environ.get("PGHOST"))
BUCKET = os.environ.get("OBJ_BUCKET")
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("OBJ_ENDPOINT"),
    aws_access_key_id=os.environ.get("OBJ_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("OBJ_SECRET_KEY"),
    config=Config(signature_version="s3v4", request_checksum_calculation="when_required"),  # see 04-object-storage
) if BUCKET else None
MAX_UPLOAD = 1_000_000  # bytes per uploaded file
MAX_CHARS = 20_000  # characters of a file sent to the model
SCHEMA = """create table if not exists questions (
  id bigserial primary key,
  question text not null,
  answer text not null,
  file text,
  created_at timestamptz not null default now()
)"""
app = FastAPI()


def db():
    """A fresh connection with the table in place. psycopg reads the PG settings from the environment."""
    conn = psycopg.connect(connect_timeout=5)
    conn.execute(SCHEMA)
    return conn


def reason(error):
    return f"error: {str(error).strip().splitlines()[0][:150]}"


class Question(BaseModel):
    question: str
    file: str | None = None


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    """connected means one round trip worked in this request. Inference says configured, because a check would spend tokens."""
    status = {"inference": "configured", "database": "not set", "object_storage": "not set"}
    if DATABASE:
        try:
            with db() as conn:
                conn.execute("select 1")
            status["database"] = "connected"
        except Exception as error:
            status["database"] = reason(error)
    if s3:
        try:
            s3.head_bucket(Bucket=BUCKET)
            status["object_storage"] = "connected"
        except Exception as error:
            status["object_storage"] = reason(error)
    return status


@app.post("/ask")
def ask(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    prompt = question
    if body.file:
        if not s3:
            raise HTTPException(400, "No bucket is connected.")
        text = s3.get_object(Bucket=BUCKET, Key=body.file)["Body"].read().decode("utf-8", "replace")
        prompt = f"Use this file, {body.file}, to answer. If it lacks the answer, say so.\n\n{text[:MAX_CHARS]}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer in a few sentences of plain text, with no Markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
    )
    answer = response.choices[0].message.content or ""
    saved = False
    if DATABASE:
        try:
            with db() as conn:
                conn.execute("insert into questions (question, answer, file) values (%s, %s, %s)",
                             (question, answer, body.file))
            saved = True
        except Exception:
            pass  # the answer still goes back. /health shows the database error
    return {"answer": answer, "saved": saved}


@app.get("/history")
def history():
    if not DATABASE:
        return []
    with db() as conn:
        rows = conn.execute("select question, answer, file, created_at from questions order by id desc limit 20")
        return [{"question": q, "answer": a, "file": f, "created_at": t.isoformat()} for q, a, f, t in rows]


@app.get("/files")
def files():
    if not s3:
        return []
    return [item["Key"] for item in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


@app.post("/files")
async def upload(file: UploadFile):
    if not s3:
        raise HTTPException(400, "No bucket is connected.")
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Files are limited to 1 MB.")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(415, "Upload a text file, such as .txt or .md.")
    key = Path(file.filename or "upload.txt").name
    s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="text/plain; charset=utf-8")
    return {"file": key}


# ============================================================================
# AIRLOCK: privacy + cost gateway in front of the same Akamai Inference client
# ============================================================================

PRICE_PER_1K_TOKENS = float(os.environ.get("PRICE_PER_1K_TOKENS", "0.002"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
ARCHIVE_BATCH_SIZE = int(os.environ.get("ARCHIVE_BATCH_SIZE", "10"))

SYSTEM_NOTE = (
    "Some text below may contain bracketed placeholders such as [EMAIL_1] or [API_KEY_1]. "
    "These stand in for sensitive values that were redacted before reaching you. Treat each "
    "placeholder as an opaque token: never guess, decode, or describe what it might be. Copy "
    "every placeholder back into your reply exactly as written, in the same place, wherever "
    "your answer would otherwise include that value."
)

# --- redaction detectors -----------------------------------------------------

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
IPV4_PATTERN = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
)
CREDITCARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
SECRET_PREFIX_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
]
BEARER_PATTERN = re.compile(r"\b(Bearer\s+)([A-Za-z0-9\-_.]{10,})\b", re.IGNORECASE)
GENERIC_TOKEN_PATTERNS = [
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
]
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")


def luhn_valid(raw):
    digits = re.sub(r"[ \-]", "", raw)
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def build_redactor():
    """Returns (apply, mapping, counts). mapping and counts live only for one request."""
    mapping = {}  # raw value -> placeholder, this request only
    next_index = Counter()  # type -> next placeholder number
    counts = Counter()  # type -> total occurrences redacted

    def placeholder_for(raw, type_key):
        existing = mapping.get(raw)
        if existing:
            return existing
        next_index[type_key] += 1
        placeholder = f"[{type_key}_{next_index[type_key]}]"
        mapping[raw] = placeholder
        return placeholder

    def make_repl(type_key, validator=None):
        def repl(m):
            raw = m.group(0)
            if validator and not validator(raw):
                return raw
            counts[type_key] += 1
            return placeholder_for(raw, type_key)
        return repl

    def bearer_repl(m):
        token = m.group(2)
        counts["API_KEY"] += 1
        return f"{m.group(1)}{placeholder_for(token, 'API_KEY')}"

    def apply(text):
        for pattern in SECRET_PREFIX_PATTERNS:
            text = pattern.sub(make_repl("API_KEY"), text)
        text = BEARER_PATTERN.sub(bearer_repl, text)
        text = CREDITCARD_PATTERN.sub(make_repl("CREDIT_CARD", luhn_valid), text)
        text = SSN_PATTERN.sub(make_repl("SSN"), text)
        text = EMAIL_PATTERN.sub(make_repl("EMAIL"), text)
        text = PHONE_PATTERN.sub(make_repl("PHONE"), text)
        text = IPV4_PATTERN.sub(make_repl("IPV4"), text)
        for pattern in GENERIC_TOKEN_PATTERNS:
            text = pattern.sub(make_repl("API_KEY"), text)
        return text

    return apply, mapping, counts


def restore_placeholders(text, mapping):
    reverse = {placeholder: raw for raw, placeholder in mapping.items()}
    return PLACEHOLDER_PATTERN.sub(lambda m: reverse.get(m.group(0), m.group(0)), text)


# --- rate limiting -------------------------------------------------------

class RateLimiter:
    def __init__(self, limit, window_seconds=60):
        self.limit = limit
        self.window = window_seconds
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key):
        now = time.monotonic()
        with self.lock:
            dq = self.hits[key]
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False, int(self.window - (now - dq[0])) + 1
            dq.append(now)
            return True, 0


rate_limiter = RateLimiter(RATE_LIMIT_PER_MIN)


# --- stats, always in-memory, regardless of storage backend --------------

class StatsAggregator:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.cache_hits = 0
        self.redactions = Counter()
        self.cost_actual = 0.0
        self.cost_saved = 0.0
        self.latency_cached = deque(maxlen=200)
        self.latency_uncached = deque(maxlen=200)

    def record(self, rec):
        with self.lock:
            self.total += 1
            if rec["cache_hit"]:
                self.cache_hits += 1
                self.cost_saved += rec["est_cost_if_uncached"]
                self.latency_cached.append(rec["latency_ms"])
            else:
                self.cost_actual += rec["cost_usd"]
                self.latency_uncached.append(rec["latency_ms"])
            for type_key, n in rec["redactions"].items():
                self.redactions[type_key] += n

    def snapshot(self):
        def avg(dq):
            return round(sum(dq) / len(dq), 1) if dq else None

        with self.lock:
            return {
                "total_requests": self.total,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hits / self.total, 4) if self.total else 0.0,
                "redactions_by_type": dict(self.redactions),
                "secrets_blocked": sum(self.redactions.values()),
                "cost_actual_usd": round(self.cost_actual, 6),
                "cost_saved_usd": round(self.cost_saved, 6),
                "avg_latency_ms_cached": avg(self.latency_cached),
                "avg_latency_ms_uncached": avg(self.latency_uncached),
            }


stats = StatsAggregator()

# --- storage: Postgres when reachable, in-memory fallback otherwise -------

REQUESTS_SCHEMA = """create table if not exists requests (
  id uuid primary key,
  created_at timestamptz not null default now(),
  latency_ms int,
  cache_hit bool,
  redactions jsonb,
  tokens_in int,
  tokens_out int,
  cost_usd numeric,
  redacted_prompt text
)"""
CACHE_SCHEMA = """create table if not exists cache (
  key text primary key,
  response text,
  created_at timestamptz not null default now(),
  hits int not null default 0
)"""


def pg_dsn():
    """A DATABASE_URL conninfo string, "" to read PG* env vars via libpq, or None if neither is set."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if os.environ.get("PGHOST"):
        return ""
    return None


class MemoryBackend:
    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}
        self.requests = deque(maxlen=500)

    def cache_get(self, key):
        with self.lock:
            entry = self.cache.get(key)
            if not entry:
                return None
            entry["hits"] += 1
            return {"response": entry["response"]}

    def cache_set(self, key, response_text):
        with self.lock:
            self.cache[key] = {"response": response_text, "hits": 0}

    def log_request(self, rec):
        with self.lock:
            self.requests.appendleft(rec)

    def recent(self, limit):
        with self.lock:
            return list(self.requests)[:limit]


class PostgresBackend:
    def __init__(self, dsn):
        self.dsn = dsn

    def _conn(self):
        return psycopg.connect(self.dsn, connect_timeout=5) if self.dsn else psycopg.connect(connect_timeout=5)

    def init(self):
        with self._conn() as conn:
            conn.execute(REQUESTS_SCHEMA)
            conn.execute(CACHE_SCHEMA)

    def cache_get(self, key):
        with self._conn() as conn:
            row = conn.execute("select response from cache where key = %s", (key,)).fetchone()
            if not row:
                return None
            conn.execute("update cache set hits = hits + 1 where key = %s", (key,))
            return {"response": row[0]}

    def cache_set(self, key, response_text):
        with self._conn() as conn:
            conn.execute(
                "insert into cache (key, response) values (%s, %s) "
                "on conflict (key) do update set response = excluded.response",
                (key, response_text),
            )

    def log_request(self, rec):
        with self._conn() as conn:
            conn.execute(
                "insert into requests (id, latency_ms, cache_hit, redactions, tokens_in, tokens_out, cost_usd, redacted_prompt) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s)",
                (rec["id"], rec["latency_ms"], rec["cache_hit"], json.dumps(rec["redactions"]),
                 rec["tokens_in"], rec["tokens_out"], rec["cost_usd"], rec["redacted_prompt"]),
            )

    def recent(self, limit):
        with self._conn() as conn:
            rows = conn.execute(
                "select id, created_at, latency_ms, cache_hit, redactions, tokens_in, tokens_out, cost_usd, redacted_prompt "
                "from requests order by created_at desc limit %s", (limit,),
            )
            return [
                {"id": str(r[0]), "created_at": r[1].isoformat(), "latency_ms": r[2], "cache_hit": r[3],
                 "redactions": r[4], "tokens_in": r[5], "tokens_out": r[6],
                 "cost_usd": float(r[7]) if r[7] is not None else None, "redacted_prompt": r[8]}
                for r in rows
            ]


class Storage:
    """Tries Postgres once at startup. Any later failure demotes to the in-memory backend for good,
    so a database outage never crashes a request; it only loses durability."""

    def __init__(self):
        self.memory = MemoryBackend()
        self.pg = None
        dsn = pg_dsn()
        if dsn is not None:
            backend = PostgresBackend(dsn)
            try:
                backend.init()
                self.pg = backend
            except Exception:
                self.pg = None

    @property
    def backend_name(self):
        return "postgres" if self.pg else "memory"

    def cache_get(self, key):
        if self.pg:
            try:
                return self.pg.cache_get(key)
            except Exception:
                self.pg = None
        return self.memory.cache_get(key)

    def cache_set(self, key, response_text):
        self.memory.cache_set(key, response_text)
        if self.pg:
            try:
                self.pg.cache_set(key, response_text)
            except Exception:
                self.pg = None

    def log_request(self, rec):
        self.memory.log_request(rec)
        if self.pg:
            try:
                self.pg.log_request(rec)
            except Exception:
                self.pg = None

    def recent(self, limit):
        if self.pg:
            try:
                return self.pg.recent(limit)
            except Exception:
                self.pg = None
        return self.memory.recent(limit)


storage = Storage()


# --- object storage archive: batched, redacted-only, append-free JSONL objects ----

class ArchiveBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = []

    def add(self, record):
        if not s3:
            return
        with self.lock:
            self.buffer.append(record)
            full = len(self.buffer) >= ARCHIVE_BATCH_SIZE
        if full:
            self.flush()

    def flush(self):
        if not s3:
            return None
        with self.lock:
            if not self.buffer:
                return None
            batch, self.buffer = self.buffer, []
        body = "\n".join(json.dumps(r, default=str) for r in batch).encode("utf-8")
        key = f"airlock-logs/{datetime.now(timezone.utc):%Y%m%d}/{uuid.uuid4().hex}.jsonl"
        s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/x-ndjson")
        return key


archive_buffer = ArchiveBuffer()


# --- the gateway endpoint --------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    restore: bool = True


@app.post("/v1/chat")
def gateway_chat(body: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.check(client_ip)
    if not allowed:
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded. Slow down.",
                                                        "retry_after_s": retry_after})
    if not body.messages:
        raise HTTPException(400, "messages is required")

    start = time.monotonic()
    request_id = str(uuid.uuid4())
    apply_redaction, mapping, occurrence_counts = build_redactor()
    redacted_messages = [{"role": m.role, "content": apply_redaction(m.content)} for m in body.messages]
    redacted_prompt = next(
        (m["content"] for m in reversed(redacted_messages) if m["role"] == "user"),
        redacted_messages[-1]["content"],
    )

    normalized = [{"role": m["role"], "content": " ".join(m["content"].split())} for m in redacted_messages]
    cache_key = hashlib.sha256((CHAT_MODEL + json.dumps(normalized, sort_keys=True)).encode()).hexdigest()

    cached = storage.cache_get(cache_key)
    completion_usage = None
    if cached:
        response_text = cached["response"]
        cache_hit = True
    else:
        try:
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "system", "content": SYSTEM_NOTE}, *redacted_messages],
                max_tokens=500,
            )
        except Exception as error:
            return JSONResponse(status_code=502, content={
                "error": reason(error), "id": request_id, "latency_ms": int((time.monotonic() - start) * 1000),
            })
        response_text = completion.choices[0].message.content or ""
        completion_usage = getattr(completion, "usage", None)
        cache_hit = False
        storage.cache_set(cache_key, response_text)

    if completion_usage is not None and getattr(completion_usage, "prompt_tokens", None) is not None:
        tokens_in = completion_usage.prompt_tokens
        tokens_out = completion_usage.completion_tokens
    else:
        input_text = " ".join(m["content"] for m in redacted_messages)
        tokens_in = max(1, len(input_text) // 4)
        tokens_out = max(1, len(response_text) // 4)

    est_cost = round((tokens_in + tokens_out) / 1000 * PRICE_PER_1K_TOKENS, 6)
    cost_usd = 0.0 if cache_hit else est_cost
    latency_ms = int((time.monotonic() - start) * 1000)
    final_text = restore_placeholders(response_text, mapping) if body.restore else response_text
    redactions = dict(occurrence_counts)

    record = {
        "id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "redactions": redactions,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "est_cost_if_uncached": est_cost,
        "redacted_prompt": redacted_prompt[:300],
    }
    storage.log_request(record)
    stats.record(record)
    archive_buffer.add(record)

    return {
        "id": request_id,
        "response": final_text,
        "redacted_prompt": redacted_prompt,
        "redactions": redactions,
        "cache_hit": cache_hit,
        "latency_ms": latency_ms,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "cost_usd": cost_usd,
    }


@app.get("/api/stats")
def api_stats():
    snapshot = stats.snapshot()
    snapshot["backends"] = {
        "inference": "configured",
        "postgresql": "connected" if storage.pg else "fallback: in-memory",
        "object_storage": "connected" if s3 else "not set",
    }
    return snapshot


@app.get("/api/requests")
def api_requests(limit: int = 20):
    return storage.recent(max(1, min(limit, 100)))


@app.post("/api/archive")
def api_archive():
    if not s3:
        raise HTTPException(400, "No bucket is connected.")
    key = archive_buffer.flush()
    if not key:
        raise HTTPException(400, "Nothing to archive yet. Send a few requests first.")
    return {"key": key}


if __name__ == "__main__":  # a server sets PORT. Without it, only this computer can open the app.
    uvicorn.run(app, host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=int(os.environ.get("PORT", 8000)))

"""
AskMark Backend — Complete FastAPI Server
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import os
import io
import redis as redis_lib
from datetime import datetime, timezone
import httpx
import time
import collections
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

app = FastAPI(title="AskMark API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] and allow_credentials=True cannot coexist (CORS spec).
# Set ALLOWED_ORIGINS env var (comma-separated) to enable credentials for specific origins.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials="*" not in _allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REDIS ─────────────────────────────────────────────────────────────────────
try:
    r = redis_lib.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True
    )
    r.ping()
    print("✅ Redis connected")
except Exception as e:
    print(f"⚠️  Redis not available: {e} — using in-memory fallback")
    r = None

_mem = {}

def db_set(key, value):
    if r:
        r.set(key, json.dumps(value))
    else:
        _mem[key] = value

def db_get(key):
    if r:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    return _mem.get(key)

def db_delete(key):
    if r: r.delete(key)
    elif key in _mem: del _mem[key]

def db_sadd(key, val):
    if r: r.sadd(key, val)
    else: _mem.setdefault(key, set()).add(val)

def db_srem(key, val):
    if r: r.srem(key, val)
    elif key in _mem: _mem[key].discard(val)

def db_smembers(key):
    if r: return r.smembers(key)
    return _mem.get(key, set())

def now():
    return datetime.now(timezone.utc).isoformat()

# ── AUTH ──────────────────────────────────────────────────────────────────────
# Set API_KEY env var to enforce a shared secret on all endpoints.
# If unset, auth is disabled (open access — suitable for local dev only).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: Optional[str] = Depends(_api_key_header)):
    server_key = os.getenv("API_KEY")
    if server_key and key != server_key:
        raise HTTPException(401, "Invalid or missing API key")

# ── RATE LIMITING ─────────────────────────────────────────────────────────────
# Limits /claude calls to CLAUDE_RATE_LIMIT requests per minute per IP (default 10).
_claude_calls: dict = collections.defaultdict(list)
_RATE_LIMIT = int(os.getenv("CLAUDE_RATE_LIMIT", "10"))
_RATE_WINDOW = 60  # seconds

def check_rate_limit(ip: str):
    now_ts = time.monotonic()
    window_start = now_ts - _RATE_WINDOW
    _claude_calls[ip] = [t for t in _claude_calls[ip] if t > window_start]
    if len(_claude_calls[ip]) >= _RATE_LIMIT:
        raise HTTPException(429, f"Rate limit: max {_RATE_LIMIT} Claude requests per minute")
    _claude_calls[ip].append(now_ts)

# ── MODELS ────────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    project_type: Optional[str] = "tvc"
    client_name: Optional[str] = ""
    director_name: Optional[str] = ""
    producer_name: Optional[str] = ""
    total_budget: Optional[float] = 0
    currency: Optional[str] = "INR"
    shoot_start_date: Optional[str] = ""
    shoot_end_date: Optional[str] = ""
    delivery_date: Optional[str] = ""
    status: Optional[str] = "pre-production"
    brief: Optional[str] = ""

class ProjectUpdate(BaseModel):
    project_id: str
    name: Optional[str] = None
    project_type: Optional[str] = None
    client_name: Optional[str] = None
    director_name: Optional[str] = None
    producer_name: Optional[str] = None
    total_budget: Optional[float] = None
    currency: Optional[str] = None
    shoot_start_date: Optional[str] = None
    shoot_end_date: Optional[str] = None
    delivery_date: Optional[str] = None
    status: Optional[str] = None
    brief: Optional[str] = None

class ProjectIdRequest(BaseModel):
    project_id: str

class CrewCreate(BaseModel):
    project_id: str
    name: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    role_title: Optional[str] = ""
    department: Optional[str] = "Other"
    day_rate: Optional[float] = 0
    rate_currency: Optional[str] = "INR"
    dietary_requirements: Optional[str] = ""
    emergency_contact: Optional[str] = ""
    notes: Optional[str] = ""

class CrewUpdate(BaseModel):
    crew_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    role_title: Optional[str] = None
    department: Optional[str] = None
    day_rate: Optional[float] = None
    rate_currency: Optional[str] = None
    dietary_requirements: Optional[str] = None
    emergency_contact: Optional[str] = None
    notes: Optional[str] = None

class CrewIdRequest(BaseModel):
    crew_id: str

class BudgetSave(BaseModel):
    project_id: str
    budget_data: dict
    version: Optional[str] = "1.0"

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    options: Optional[list] = None

class BudgetGenerate(BaseModel):
    # Either project_id (project-based flow) OR script + region + currency (standalone flow).
    project_id: Optional[str] = None
    script: Optional[str] = ""
    region: Optional[str] = "india"
    currency: Optional[dict] = None
    qa: Optional[list] = None
    answers: Optional[dict] = None  # legacy flat map; auto-converted if `qa` is missing
    breakdown: Optional[dict] = None  # output of /script/parse summary, when available
    version: Optional[str] = "1.0"

class CrewEnrich(BaseModel):
    crew_id: str

class FeedbackCreate(BaseModel):
    context_type: str  # "budget" | "question" | "crew" | "general"
    rating: str        # "up" | "down" | "1".."5" — caller decides; we store as a string
    comment: Optional[str] = ""
    context_id: Optional[str] = None       # budget_id / project_id / crew_id, when applicable
    snapshot: Optional[dict] = None        # the artifact the user is rating (e.g. budget JSON)
    user_email: Optional[str] = None
    user_agent: Optional[str] = None
    page: Optional[str] = None

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _apply_partial_update(existing: dict, data: BaseModel, exclude_keys: set) -> dict:
    """Only update fields that were explicitly provided in the request body."""
    # model_fields_set (Pydantic v2) or __fields_set__ (Pydantic v1)
    fields_set = getattr(data, "model_fields_set", None) or getattr(data, "__fields_set__", set())
    for field in fields_set:
        if field not in exclude_keys:
            existing[field] = getattr(data, field)
    return existing

# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.post("/")
def health():
    return {"status": "Mark is live", "version": "1.0.0"}

@app.get("/health")
def health_get():
    return {"status": "Mark is live", "version": "1.0.0"}

# ── PROJECTS ──────────────────────────────────────────────────────────────────

@app.post("/projects/create")
def create_project(data: ProjectCreate, _=Depends(require_api_key)):
    pid = str(uuid.uuid4())
    project = {
        "id": pid, "name": data.name, "project_type": data.project_type,
        "client_name": data.client_name, "director_name": data.director_name,
        "producer_name": data.producer_name, "total_budget": data.total_budget,
        "currency": data.currency, "shoot_start_date": data.shoot_start_date,
        "shoot_end_date": data.shoot_end_date, "delivery_date": data.delivery_date,
        "status": data.status, "brief": data.brief,
        "created_at": now(), "updated_at": now()
    }
    db_set(f"project:{pid}", project)
    db_sadd("projects:all", pid)
    return {"success": True, "project": project}

@app.post("/projects/list")
def list_projects(_=Depends(require_api_key)):
    ids = db_smembers("projects:all")
    projects = []
    for pid in ids:
        p = db_get(f"project:{pid}")
        if p:
            p["crew_total"] = len(db_smembers(f"project:{pid}:crew"))
            projects.append(p)
    return {"success": True, "projects": sorted(projects, key=lambda x: x.get("created_at", ""), reverse=True)}

@app.post("/projects/get")
def get_project(data: ProjectIdRequest, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    return {"success": True, "project": p}

@app.post("/projects/update")
def update_project(data: ProjectUpdate, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    _apply_partial_update(p, data, exclude_keys={"project_id"})
    p["updated_at"] = now()
    db_set(f"project:{data.project_id}", p)
    return {"success": True, "project": p}

@app.post("/projects/delete")
def delete_project(data: ProjectIdRequest, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    for cid in db_smembers(f"project:{data.project_id}:crew"):
        db_delete(f"crew:{cid}")
    db_delete(f"project:{data.project_id}:crew")
    for bid in db_smembers(f"project:{data.project_id}:budgets"):
        db_delete(f"budget:{bid}")
    db_delete(f"project:{data.project_id}:budgets")
    db_delete(f"budget:{data.project_id}:latest")
    db_delete(f"project:{data.project_id}")
    db_srem("projects:all", data.project_id)
    return {"success": True, "message": "Deleted"}

@app.post("/projects/dashboard")
def dashboard(data: ProjectIdRequest, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    crew = [db_get(f"crew:{cid}") for cid in db_smembers(f"project:{data.project_id}:crew")]
    crew = [c for c in crew if c]
    dept_breakdown = {}
    total_day_rates = 0
    for c in crew:
        dept = c.get("department", "Other")
        dept_breakdown[dept] = dept_breakdown.get(dept, 0) + 1
        total_day_rates += c.get("day_rate", 0)
    # Use tracked spend from saved budget if available; day_rate alone isn't total spend
    budget = db_get(f"budget:{data.project_id}:latest")
    budget_spend = 0
    if budget and isinstance(budget.get("budget_data"), dict):
        budget_spend = budget["budget_data"].get("total_spend", 0) or 0
    return {
        "success": True,
        "dashboard": {
            "project": p,
            "stats": {
                "crew_total": len(crew),
                "total_day_rates": total_day_rates,
                "budget_spend": budget_spend,
                "budget_remaining": (p.get("total_budget") or 0) - budget_spend,
            },
            "department_breakdown": dept_breakdown,
        }
    }

# ── CREW ──────────────────────────────────────────────────────────────────────

@app.post("/crew/create")
def create_crew(data: CrewCreate, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    cid = str(uuid.uuid4())
    member = {
        "id": cid, "project_id": data.project_id, "name": data.name,
        "email": data.email, "phone": data.phone, "role_title": data.role_title,
        "department": data.department, "day_rate": data.day_rate,
        "rate_currency": data.rate_currency,
        "dietary_requirements": data.dietary_requirements,
        "emergency_contact": data.emergency_contact, "notes": data.notes,
        "created_at": now(), "updated_at": now()
    }
    db_set(f"crew:{cid}", member)
    db_sadd(f"project:{data.project_id}:crew", cid)
    return {"success": True, "crew_member": member}

@app.post("/crew/list")
def list_crew(data: ProjectIdRequest, _=Depends(require_api_key)):
    ids = db_smembers(f"project:{data.project_id}:crew")
    crew = [db_get(f"crew:{cid}") for cid in ids]
    crew = [c for c in crew if c]
    return {"success": True, "crew": sorted(crew, key=lambda x: x.get("created_at", "")), "total": len(crew)}

@app.post("/crew/get")
def get_crew(data: CrewIdRequest, _=Depends(require_api_key)):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    return {"success": True, "crew_member": c}

@app.post("/crew/update")
def update_crew(data: CrewUpdate, _=Depends(require_api_key)):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    _apply_partial_update(c, data, exclude_keys={"crew_id"})
    c["updated_at"] = now()
    db_set(f"crew:{data.crew_id}", c)
    return {"success": True, "crew_member": c}

@app.post("/crew/delete")
def delete_crew(data: CrewIdRequest, _=Depends(require_api_key)):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    db_srem(f"project:{c['project_id']}:crew", data.crew_id)
    db_delete(f"crew:{data.crew_id}")
    return {"success": True, "message": "Deleted"}

# ── BUDGET STORAGE ────────────────────────────────────────────────────────────

@app.post("/budget/save")
def save_budget(data: BudgetSave, _=Depends(require_api_key)):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    bid = str(uuid.uuid4())
    budget = {
        "id": bid, "project_id": data.project_id,
        "version": data.version, "budget_data": data.budget_data,
        "created_at": now(), "locked": False
    }
    db_set(f"budget:{data.project_id}:latest", budget)
    db_set(f"budget:{bid}", budget)
    db_sadd(f"project:{data.project_id}:budgets", bid)
    return {"success": True, "budget_id": bid}

@app.post("/budget/get")
def get_budget(data: ProjectIdRequest, _=Depends(require_api_key)):
    b = db_get(f"budget:{data.project_id}:latest")
    if not b: raise HTTPException(404, "No budget found for this project")
    return {"success": True, "budget": b}

@app.post("/budget/history")
def budget_history(data: ProjectIdRequest, _=Depends(require_api_key)):
    ids = db_smembers(f"project:{data.project_id}:budgets")
    budgets = [db_get(f"budget:{bid}") for bid in ids]
    budgets = [b for b in budgets if b]
    return {"success": True, "budgets": sorted(budgets, key=lambda x: x.get("created_at", ""), reverse=True)}

# ── FEEDBACK ──────────────────────────────────────────────────────────────────
# Captures thumbs/comments on budgets, questions, etc. Persists to Redis (or
# in-memory fallback). Listing requires the API key when one is configured.

@app.post("/feedback/create")
def create_feedback(data: FeedbackCreate, request: Request, _=Depends(require_api_key)):
    fid = str(uuid.uuid4())
    entry = {
        "id": fid,
        "context_type": data.context_type,
        "context_id": data.context_id,
        "rating": data.rating,
        "comment": (data.comment or "").strip(),
        "snapshot": data.snapshot,
        "user_email": (data.user_email or "").strip() or None,
        "user_agent": data.user_agent or request.headers.get("user-agent"),
        "page": data.page,
        "ip": request.client.host if request.client else None,
        "created_at": now(),
    }
    db_set(f"feedback:{fid}", entry)
    db_sadd("feedback:all", fid)
    if data.context_id:
        db_sadd(f"feedback:by-context:{data.context_id}", fid)
    return {"success": True, "feedback_id": fid}

@app.post("/feedback/list")
def list_feedback(_=Depends(require_api_key)):
    ids = db_smembers("feedback:all")
    items = [db_get(f"feedback:{fid}") for fid in ids]
    items = [i for i in items if i]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"success": True, "feedback": items, "total": len(items)}

# ── SCRIPT PARSING ────────────────────────────────────────────────────────────
# Uses SMASH-CUT/screenplay-pdf-to-json to turn a PDF screenplay into structured
# scene/location/INT-EXT/character data. The agent uses this as ground truth so
# it doesn't have to re-extract structure from prose every call.

def _iter_scenes(pages: list):
    """
    Yield (scene_info, snippets) for every scene block in a parsed screenplay.
    The actual parser shape is:
        [{ "page": int, "content": [{ "scene_info": {...}|None, "scene": [...] }, ...], "type"?: "FIRST_PAGES" }]
    The README documents a flatter shape — it is wrong. Verified against parser output.
    """
    for page in pages or []:
        if page.get("type") == "FIRST_PAGES":
            continue
        for block in page.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            yield block.get("scene_info") or None, block.get("scene") or []

# Caps on summary lists — top-N by scene_count keeps Claude input-token cost bounded.
# Long-tail one-line characters and one-off locations don't drive budget decisions.
_MAX_LOCATIONS = 25
_MAX_CHARACTERS = 30

def _process_pages(pages: list) -> tuple[dict, str]:
    """
    Single pass over the parser output: builds both the compact summary
    AND reconstructs plain script text. Avoids two traversals of a large structure.
    """
    int_count = ext_count = day_count = night_count = total_scenes = 0
    locations: dict = {}
    characters: dict = {}
    text_chunks: list = []

    for scene_info, snippets in _iter_scenes(pages):
        if scene_info:
            total_scenes += 1
            region = (scene_info.get("region") or "").upper()
            if "INT" in region:
                int_count += 1
            if "EXT" in region:
                ext_count += 1
            for t in scene_info.get("time") or []:
                tu = (t or "").upper()
                if any(k in tu for k in ("DAY", "MORNING", "AFTERNOON")):
                    day_count += 1
                if any(k in tu for k in ("NIGHT", "EVENING", "DUSK")):
                    night_count += 1
            loc = scene_info.get("location")
            if loc:
                locations[loc] = locations.get(loc, 0) + 1
            heading = (
                f"{scene_info.get('region','')} {scene_info.get('location','') or ''}"
                f" - {' / '.join(scene_info.get('time') or [])}"
            ).strip(" -")
            if heading:
                text_chunks.append(heading)
        for snippet in snippets:
            content = snippet.get("content")
            if snippet.get("type") == "CHARACTER" and isinstance(content, dict):
                name = content.get("character")
                if name:
                    characters[name] = characters.get(name, 0) + 1
            if isinstance(content, str):
                text_chunks.append(content)
            elif isinstance(content, dict):
                for v in content.values():
                    if isinstance(v, str):
                        text_chunks.append(v)
                    elif isinstance(v, list):
                        text_chunks.extend(x for x in v if isinstance(x, str))
            elif isinstance(content, list):
                text_chunks.extend(x for x in content if isinstance(x, str))

    summary = {
        "total_scenes": total_scenes,
        "int_count": int_count,
        "ext_count": ext_count,
        "day_count": day_count,
        "night_count": night_count,
        "unique_locations": [
            {"name": k, "scene_count": v}
            for k, v in sorted(locations.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_LOCATIONS]
        ],
        "characters": [
            {"name": k, "scene_count": v}
            for k, v in sorted(characters.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_CHARACTERS]
        ],
    }
    return summary, "\n".join(c for c in text_chunks if c)

_MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(25 * 1024 * 1024)))  # 25MB default

def _parse_pdf_sync(raw: bytes) -> tuple[dict, str]:
    """CPU-bound work — must run via run_in_threadpool, not on the event loop."""
    from screenplay_pdf_to_json import convert
    pages = convert(io.BytesIO(raw), 0)  # second arg is start-page; 0 = scan from beginning
    return _process_pages(pages)

@app.post("/script/parse")
async def parse_script(file: UploadFile = File(...), _=Depends(require_api_key)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF screenplays are supported")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > _MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF too large (max {_MAX_PDF_BYTES // 1024 // 1024} MB)")

    try:
        summary, extracted_text = await run_in_threadpool(_parse_pdf_sync, raw)
    except ImportError:
        raise HTTPException(500, "screenplay-pdf-to-json not installed on server")
    except Exception as e:
        raise HTTPException(422, f"Failed to parse PDF: {e}")

    return {
        "success": True,
        "summary": summary,
        "extracted_text": extracted_text,
    }

# ── FLUE AGENT PROXY ──────────────────────────────────────────────────────────
# Forwards questionnaire answers + project context to the Flue sidecar so the
# agent can produce a typed budget. Set FLUE_BASE_URL to the running Flue server
# (default http://localhost:3583 in dev, or the deployed Node URL in prod).

_FLUE_BASE_URL = os.getenv("FLUE_BASE_URL", "http://localhost:3583")

async def _flue_call(agent_name: str, run_id: str, payload: dict) -> dict:
    url = f"{_FLUE_BASE_URL}/agents/{agent_name}/{run_id}"
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(url, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(504, f"flue/{agent_name}: agent timed out")
    except httpx.RequestError as e:
        raise HTTPException(503, f"flue/{agent_name}: agent unavailable ({type(e).__name__})")
    if not resp.is_success:
        try:
            detail = resp.json().get("error") or resp.text
        except Exception:
            detail = f"Flue agent error (HTTP {resp.status_code})"
        raise HTTPException(resp.status_code, f"flue/{agent_name}: {detail}")
    return resp.json()

@app.post("/budget/generate")
async def generate_budget(data: BudgetGenerate, _=Depends(require_api_key)):
    # Build the qa array. Prefer the structured `qa` payload; fall back to
    # the legacy flat `answers` map (where the question text is missing).
    qa = data.qa or []
    if not qa and data.answers:
        qa = [{"id": k, "question": k, "answer": str(v)} for k, v in data.answers.items()]

    project = None
    crew: list = []
    if data.project_id:
        project = db_get(f"project:{data.project_id}")
        if not project:
            raise HTTPException(404, "Project not found")
        crew = [db_get(f"crew:{cid}") for cid in db_smembers(f"project:{data.project_id}:crew")]
        crew = [c for c in crew if c]

    # `project_type` is "tvc"/"music_video", NOT a region — never use it as a fallback.
    region = data.region or "india"
    _CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "GBP": "£"}
    currency = data.currency
    if not currency and project:
        code = project.get("currency", "INR")
        currency = {"code": code, "symbol": _CURRENCY_SYMBOLS.get(code, code)}
    currency = currency or {"code": "INR", "symbol": "₹"}

    payload = {
        "script": data.script or (project.get("brief", "") if project else ""),
        "region": region,
        "currency": currency,
        "qa": qa,
    }
    if data.breakdown:
        payload["breakdown"] = data.breakdown
    if project:
        payload["project"] = project
        payload["crew"] = crew

    run_id = data.project_id or str(uuid.uuid4())
    agent_result = await _flue_call("generate-budget", run_id, payload)

    if not data.project_id:
        # Standalone (scriptless) flow — return without persisting.
        return {"success": True, "budget": {"budget_data": agent_result, "qa": qa, "source": "flue:generate-budget"}}

    bid = str(uuid.uuid4())
    budget = {
        "id": bid,
        "project_id": data.project_id,
        "version": data.version,
        "budget_data": agent_result,
        "qa": qa,
        "created_at": now(),
        "locked": False,
        "source": "flue:generate-budget",
    }
    db_set(f"budget:{data.project_id}:latest", budget)
    db_set(f"budget:{bid}", budget)
    db_sadd(f"project:{data.project_id}:budgets", bid)
    return {"success": True, "budget_id": bid, "budget": budget}

@app.post("/crew/enrich")
async def enrich_crew_member(data: CrewEnrich, _=Depends(require_api_key)):
    member = db_get(f"crew:{data.crew_id}")
    if not member:
        raise HTTPException(404, "Crew member not found")
    project = db_get(f"project:{member['project_id']}")
    if not project:
        raise HTTPException(404, "Project not found for this crew member")

    enrichment = await _flue_call(
        "enrich-crew",
        data.crew_id,
        {"crew_member": member, "project": project},
    )
    return {"success": True, "crew_id": data.crew_id, "enrichment": enrichment}

# ── CLAUDE PROXY ──────────────────────────────────────────────────────────────

class ClaudeRequest(BaseModel):
    system: str
    user: str
    max_tokens: Optional[int] = 3000

@app.post("/claude")
async def claude_proxy(data: ClaudeRequest, request: Request, _=Depends(require_api_key)):
    check_rate_limit(request.client.host)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise HTTPException(500, "API key not configured on server")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": data.max_tokens,
                "system": data.system,
                "messages": [{"role": "user", "content": data.user}],
            },
        )
    if not resp.is_success:
        try:
            detail = resp.json().get("error", {}).get("message", "Claude API error")
        except Exception:
            detail = f"Claude API error (HTTP {resp.status_code})"
        raise HTTPException(resp.status_code, detail)
    return resp.json()

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
_frontend_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'public')
)

if os.path.exists(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/")
    def serve_landing():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))

    @app.get("/app")
    @app.get("/app.html")
    def serve_app():
        return FileResponse(os.path.join(_frontend_dir, "app.html"))

    @app.get("/budget.html")
    def serve_budget():
        return FileResponse(os.path.join(_frontend_dir, "budget.html"))

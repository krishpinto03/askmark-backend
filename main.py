"""
AskMark Backend — Complete FastAPI Server
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import os
import redis as redis_lib
from datetime import datetime
import httpx
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

app = FastAPI(title="AskMark API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

# In-memory fallback if Redis not available
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
    return datetime.utcnow().isoformat()

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

# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.post("/")
def health():
    return {"status": "Mark is live", "version": "1.0.0"}

@app.get("/health")
def health_get():
    return {"status": "Mark is live", "version": "1.0.0"}

# ── PROJECTS ──────────────────────────────────────────────────────────────────

@app.post("/projects/create")
def create_project(data: ProjectCreate):
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
def list_projects():
    ids = db_smembers("projects:all")
    projects = []
    for pid in ids:
        p = db_get(f"project:{pid}")
        if p:
            crew = [db_get(f"crew:{cid}") for cid in db_smembers(f"project:{pid}:crew")]
            crew = [c for c in crew if c]
            p["crew_total"] = len(crew)
            projects.append(p)
    return {"success": True, "projects": sorted(projects, key=lambda x: x.get("created_at",""), reverse=True)}

@app.post("/projects/get")
def get_project(data: ProjectIdRequest):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    return {"success": True, "project": p}

@app.post("/projects/update")
def update_project(data: ProjectUpdate):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    updates = {k: v for k, v in data.dict().items() if v is not None and k != "project_id"}
    p.update(updates)
    p["updated_at"] = now()
    db_set(f"project:{data.project_id}", p)
    return {"success": True, "project": p}

@app.post("/projects/delete")
def delete_project(data: ProjectIdRequest):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    for cid in db_smembers(f"project:{data.project_id}:crew"):
        db_delete(f"crew:{cid}")
    db_delete(f"project:{data.project_id}")
    db_delete(f"project:{data.project_id}:crew")
    db_srem("projects:all", data.project_id)
    return {"success": True, "message": "Deleted"}

@app.post("/projects/dashboard")
def dashboard(data: ProjectIdRequest):
    p = db_get(f"project:{data.project_id}")
    if not p: raise HTTPException(404, "Project not found")
    crew = [db_get(f"crew:{cid}") for cid in db_smembers(f"project:{data.project_id}:crew")]
    crew = [c for c in crew if c]
    dept_breakdown = {}
    total_rates = 0
    for c in crew:
        dept = c.get("department", "Other")
        dept_breakdown[dept] = dept_breakdown.get(dept, 0) + 1
        total_rates += c.get("day_rate", 0)
    return {
        "success": True,
        "dashboard": {
            "project": p,
            "stats": {
                "crew_total": len(crew),
                "total_day_rates": total_rates,
                "budget_remaining": (p.get("total_budget") or 0) - total_rates,
            },
            "department_breakdown": dept_breakdown,
        }
    }

# ── CREW ──────────────────────────────────────────────────────────────────────

@app.post("/crew/create")
def create_crew(data: CrewCreate):
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
def list_crew(data: ProjectIdRequest):
    ids = db_smembers(f"project:{data.project_id}:crew")
    crew = [db_get(f"crew:{cid}") for cid in ids]
    crew = [c for c in crew if c]
    return {"success": True, "crew": sorted(crew, key=lambda x: x.get("created_at","")), "total": len(crew)}

@app.post("/crew/get")
def get_crew(data: CrewIdRequest):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    return {"success": True, "crew_member": c}

@app.post("/crew/update")
def update_crew(data: CrewUpdate):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    updates = {k: v for k, v in data.dict().items() if v is not None and k != "crew_id"}
    c.update(updates)
    c["updated_at"] = now()
    db_set(f"crew:{data.crew_id}", c)
    return {"success": True, "crew_member": c}

@app.post("/crew/delete")
def delete_crew(data: CrewIdRequest):
    c = db_get(f"crew:{data.crew_id}")
    if not c: raise HTTPException(404, "Crew member not found")
    db_srem(f"project:{c['project_id']}:crew", data.crew_id)
    db_delete(f"crew:{data.crew_id}")
    return {"success": True, "message": "Deleted"}

# ── BUDGET STORAGE ────────────────────────────────────────────────────────────

@app.post("/budget/save")
def save_budget(data: BudgetSave):
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
    return {"success": True, "budget_id": bid}

@app.post("/budget/get")
def get_budget(data: ProjectIdRequest):
    b = db_get(f"budget:{data.project_id}:latest")
    if not b: raise HTTPException(404, "No budget found for this project")
    return {"success": True, "budget": b}

# ── CLAUDE PROXY ─────────────────────────────────────────────────────────────

class ClaudeRequest(BaseModel):
    system: str
    user: str
    max_tokens: Optional[int] = 3000

@app.post("/claude")
async def claude_proxy(data: ClaudeRequest):
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
                "model": "claude-sonnet-4-20250514",
                "max_tokens": data.max_tokens,
                "system": data.system,
                "messages": [{"role": "user", "content": data.user}],
            },
        )
    if not resp.is_success:
        raise HTTPException(resp.status_code, resp.json().get("error", {}).get("message", "Claude API error"))
    return resp.json()

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
if os.path.exists("../frontend/public"):
    app.mount("/static", StaticFiles(directory="../frontend/public"), name="static")

    @app.get("/")
    def serve_landing():
        return FileResponse("../frontend/public/index.html")

    @app.get("/app")
    @app.get("/app.html")
    def serve_app():
        return FileResponse("../frontend/public/app.html")

    @app.get("/budget.html")
    def serve_budget():
        return FileResponse("../frontend/public/budget.html")

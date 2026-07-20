from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from inference import analyze_bug
from database import init_db, get_bugs, get_stats, delete_bug
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend", "index.html")

app = FastAPI(title="Adaptive Bug Report Structurer")

@app.on_event("startup")
def startup():
    init_db()

class BugReport(BaseModel):
    description: str

@app.post("/analyze")
def analyze(bug: BugReport):
    return analyze_bug(bug.description)

@app.get("/bugs")
def list_bugs(
    severity: Optional[str] = None,
    needs_review: Optional[bool] = None,
    schema_type: Optional[str] = None,
    limit: int = 100,
):
    return get_bugs(severity=severity, needs_review=needs_review, schema_type=schema_type, limit=limit)


@app.get("/stats")
def stats():
    return get_stats()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/dashboard")
def dashboard():
    return FileResponse(FRONTEND_PATH)

@app.delete("/bugs/{bug_id}")
def remove_bug(bug_id: int):
    deleted = delete_bug(bug_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bug not found")
    return {"deleted": True, "id": bug_id}
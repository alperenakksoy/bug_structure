from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from inference import analyze_bug
from database import init_db, get_bugs, get_stats
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def list_bugs(severity: Optional[str] = None, needs_review: Optional[bool] = None, limit: int = 100):
    return get_bugs(severity=severity, needs_review=needs_review, limit=limit)

@app.get("/stats")
def stats():
    return get_stats()

@app.get("/health")
def health():
    return {"status": "ok"}
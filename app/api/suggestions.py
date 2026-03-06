from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import ImprovementSuggestion
from app.services.suggestion_service import SuggestionService

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


@router.get("/")
def list_suggestions(
    status: Optional[str] = None,
    suggestion_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    svc = SuggestionService(db)
    suggestions = svc.list_suggestions(status=status, limit=limit)
    if suggestion_type:
        suggestions = [s for s in suggestions if s.suggestion_type == suggestion_type]
    return [_serialize(s) for s in suggestions]


@router.post("/generate", status_code=202)
def trigger_generation(db: Session = Depends(get_db)):
    """Manually trigger improvement suggestion generation."""
    from app.workers.tasks import auto_generate_suggestions
    auto_generate_suggestions.delay()
    return {"message": "Suggestion generation job queued"}


@router.post("/generate/sync")
def generate_sync(db: Session = Depends(get_db)):
    """Synchronously generate suggestions (for testing / demo)."""
    svc = SuggestionService(db)
    count = svc.generate_suggestions()
    return {"suggestions_created": count}


@router.patch("/{suggestion_id}/status")
def update_suggestion_status(
    suggestion_id: str,
    status: str,
    db: Session = Depends(get_db),
):
    if status not in ("pending", "applied", "rejected"):
        raise HTTPException(status_code=400, detail="Invalid status")
    svc = SuggestionService(db)
    s = svc.update_status(suggestion_id, status)
    if not s:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return _serialize(s)


def _serialize(s: ImprovementSuggestion) -> dict:
    return {
        "suggestion_id": s.suggestion_id,
        "suggestion_type": s.suggestion_type,
        "target": s.target,
        "suggestion": s.suggestion,
        "rationale": s.rationale,
        "expected_impact": s.expected_impact,
        "confidence": s.confidence,
        "failure_patterns": s.failure_patterns,
        "sample_conversation_ids": s.sample_conversation_ids,
        "status": s.status,
        "created_at": s.created_at,
    }

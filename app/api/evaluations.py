from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.db_models import Conversation, Evaluation

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/")
def list_evaluations(
    conversation_id: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Evaluation)
    if conversation_id:
        q = q.filter_by(conversation_id=conversation_id)
    if min_score is not None:
        q = q.filter(Evaluation.overall_score >= min_score)
    if max_score is not None:
        q = q.filter(Evaluation.overall_score <= max_score)
    total = q.count()
    items = q.order_by(Evaluation.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [_serialize(e) for e in items]}


@router.get("/{evaluation_id}")
def get_evaluation(evaluation_id: str, db: Session = Depends(get_db)):
    ev = db.query(Evaluation).filter_by(evaluation_id=evaluation_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _serialize(ev)


@router.get("/by-conversation/{conversation_id}")
def get_by_conversation(conversation_id: str, db: Session = Depends(get_db)):
    evs = (
        db.query(Evaluation)
        .filter_by(conversation_id=conversation_id)
        .order_by(Evaluation.created_at.desc())
        .all()
    )
    return [_serialize(e) for e in evs]


@router.post("/retry/{conversation_id}", status_code=202)
def retry_evaluation(conversation_id: str, db: Session = Depends(get_db)):
    """Re-queue a failed or pending conversation for evaluation."""
    conv = db.query(Conversation).filter_by(conversation_id=conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.status = "pending"
    db.commit()
    if settings.sync_evaluation:
        from app.evaluation_runner import run_evaluation_sync
        run_evaluation_sync(conversation_id, db)
        return {"conversation_id": conversation_id, "status": "re-evaluated"}
    else:
        from app.workers.tasks import run_evaluation
        run_evaluation.delay(conversation_id)
        return {"conversation_id": conversation_id, "status": "re-queued"}


@router.get("/stats/summary")
def evaluation_stats(db: Session = Depends(get_db)):
    from app.models.db_models import ImprovementSuggestion
    total_convs = db.query(Conversation).count()
    total_evals = db.query(Evaluation).count()
    pending = db.query(Conversation).filter_by(status="pending").count()
    open_suggestions = db.query(ImprovementSuggestion).filter_by(status="pending").count()

    avg = db.query(
        func.avg(Evaluation.overall_score).label("overall"),
    ).first()

    # compute per-dimension averages from JSON — pull all scores
    evals = db.query(Evaluation.scores).filter(Evaluation.scores.isnot(None)).limit(1000).all()
    rq = [e[0].get("response_quality", 0) for e in evals if e[0]]
    ta = [e[0].get("tool_accuracy", 0) for e in evals if e[0]]
    co = [e[0].get("coherence", 0) for e in evals if e[0]]

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "total_conversations": total_convs,
        "total_evaluations": total_evals,
        "pending_conversations": pending,
        "open_suggestions": open_suggestions,
        "avg_overall_score": round(float(avg.overall or 0), 4),
        "avg_response_quality": safe_avg(rq),
        "avg_tool_accuracy": safe_avg(ta),
        "avg_coherence": safe_avg(co),
    }


def _serialize(ev: Evaluation) -> dict:
    return {
        "evaluation_id": ev.evaluation_id,
        "conversation_id": ev.conversation_id,
        "overall_score": ev.overall_score,
        "scores": ev.scores,
        "tool_evaluation": ev.tool_evaluation,
        "issues": ev.issues,
        "improvement_suggestions": ev.improvement_suggestions,
        "evaluator_version": ev.evaluator_version,
        "created_at": ev.created_at,
    }

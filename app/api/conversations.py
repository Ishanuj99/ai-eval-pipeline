import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.db_models import Conversation
from app.models.schemas import ConversationIngest

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/", status_code=202)
def ingest_conversation(payload: ConversationIngest, db: Session = Depends(get_db)):
    """Ingest a conversation and queue it for async evaluation."""
    existing = db.query(Conversation).filter_by(conversation_id=payload.conversation_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Conversation {payload.conversation_id} already exists")

    conv = Conversation(
        conversation_id=payload.conversation_id,
        agent_version=payload.agent_version,
        turns=[t.model_dump() for t in payload.turns],
        feedback_data=payload.feedback.model_dump() if payload.feedback else None,
        metadata_=payload.metadata,
        status="pending",
    )
    db.add(conv)
    db.commit()

    if settings.sync_evaluation:
        from app.evaluation_runner import run_evaluation_sync
        run_evaluation_sync(payload.conversation_id, db, fast_only=False)
        return {"conversation_id": payload.conversation_id, "status": "completed"}
    else:
        from app.workers.tasks import run_evaluation
        run_evaluation.delay(payload.conversation_id)
        return {"conversation_id": payload.conversation_id, "status": "queued"}


@router.post("/batch", status_code=202)
def ingest_batch(payloads: list[ConversationIngest], db: Session = Depends(get_db)):
    """Ingest multiple conversations at once."""
    queued = []
    skipped = []
    for payload in payloads:
        existing = db.query(Conversation).filter_by(conversation_id=payload.conversation_id).first()
        if existing:
            skipped.append(payload.conversation_id)
            continue
        conv = Conversation(
            conversation_id=payload.conversation_id,
            agent_version=payload.agent_version,
            turns=[t.model_dump() for t in payload.turns],
            feedback_data=payload.feedback.model_dump() if payload.feedback else None,
            metadata_=payload.metadata,
            status="pending",
        )
        db.add(conv)
        queued.append(payload.conversation_id)

    db.commit()

    if settings.sync_evaluation:
        from app.evaluation_runner import run_evaluation_sync
        for cid in queued:
            run_evaluation_sync(cid, db, fast_only=False)
    else:
        from app.workers.tasks import run_evaluation
        for cid in queued:
            run_evaluation.delay(cid)

    return {"queued": queued, "skipped": skipped}


@router.get("/")
def list_conversations(
    status: Optional[str] = None,
    agent_version: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Conversation)
    if status:
        q = q.filter_by(status=status)
    if agent_version:
        q = q.filter_by(agent_version=agent_version)
    total = q.count()
    items = q.order_by(Conversation.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "conversation_id": c.conversation_id,
                "agent_version": c.agent_version,
                "status": c.status,
                "turn_count": len(c.turns),
                "created_at": c.created_at,
            }
            for c in items
        ],
    }


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter_by(conversation_id=conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conv.conversation_id,
        "agent_version": conv.agent_version,
        "status": conv.status,
        "turns": conv.turns,
        "feedback": conv.feedback_data,
        "metadata": conv.metadata_,
        "created_at": conv.created_at,
    }

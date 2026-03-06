from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import AnnotationCreate
from app.services.feedback_service import FeedbackService
from app.services.meta_eval_service import MetaEvalService

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/annotations", status_code=201)
def add_annotation(payload: AnnotationCreate, db: Session = Depends(get_db)):
    try:
        svc = FeedbackService(db)
        ann = svc.add_annotation(payload)
        return {
            "id": str(ann.id),
            "conversation_id": ann.conversation_id,
            "annotator_id": ann.annotator_id,
            "annotation_type": ann.annotation_type,
            "label": ann.label,
            "confidence": ann.confidence,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/annotations/{conversation_id}")
def list_annotations(conversation_id: str, db: Session = Depends(get_db)):
    svc = FeedbackService(db)
    anns = svc.list_annotations(conversation_id)
    return [
        {
            "id": str(a.id),
            "annotator_id": a.annotator_id,
            "annotation_type": a.annotation_type,
            "label": a.label,
            "confidence": a.confidence,
            "notes": a.notes,
            "created_at": a.created_at,
        }
        for a in anns
    ]


@router.get("/agreement/{conversation_id}/{annotation_type}")
def get_agreement(conversation_id: str, annotation_type: str, db: Session = Depends(get_db)):
    svc = FeedbackService(db)
    return svc.get_agreement(conversation_id, annotation_type)


@router.post("/meta-eval/compute")
def compute_meta_eval(db: Session = Depends(get_db)):
    """Manually trigger meta-evaluation computation."""
    svc = MetaEvalService(db)
    metrics = svc.compute_and_store()
    return {"computed": metrics}


@router.get("/meta-eval/metrics")
def get_meta_eval_metrics(db: Session = Depends(get_db)):
    svc = MetaEvalService(db)
    metrics = svc.get_latest_metrics()
    return [
        {
            "evaluator_name": m.evaluator_name,
            "metric_name": m.metric_name,
            "value": m.value,
            "sample_size": m.sample_size,
            "computed_at": m.computed_at,
        }
        for m in metrics
    ]


@router.get("/meta-eval/drift")
def check_drift(db: Session = Depends(get_db)):
    """Check for evaluators that are drifting from human judgement."""
    svc = MetaEvalService(db)
    return svc.check_evaluator_drift()

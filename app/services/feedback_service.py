"""
Feedback & annotation service.
- Stores annotator labels
- Computes inter-annotator agreement (Cohen's / Fleiss' Kappa)
- Routes conversations to human review when agreement is below threshold
"""
import uuid
from collections import defaultdict

from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import Annotation, Conversation
from app.models.schemas import AnnotationCreate, AnnotationAgreement
from app.utils.agreement import cohen_kappa, fleiss_kappa, majority_label


class FeedbackService:
    def __init__(self, db: Session):
        self.db = db

    def add_annotation(self, payload: AnnotationCreate) -> Annotation:
        conv = self.db.query(Conversation).filter_by(
            conversation_id=payload.conversation_id
        ).first()
        if not conv:
            raise ValueError(f"Conversation {payload.conversation_id} not found")

        ann = Annotation(
            conversation_fk=conv.id,
            conversation_id=payload.conversation_id,
            annotator_id=payload.annotator_id,
            annotation_type=payload.annotation_type,
            label=payload.label,
            confidence=payload.confidence,
            notes=payload.notes,
        )
        self.db.add(ann)
        self.db.commit()
        self.db.refresh(ann)
        return ann

    def get_agreement(self, conversation_id: str, annotation_type: str) -> AnnotationAgreement:
        annotations = (
            self.db.query(Annotation)
            .filter_by(conversation_id=conversation_id, annotation_type=annotation_type)
            .all()
        )

        if not annotations:
            return AnnotationAgreement(
                conversation_id=conversation_id,
                annotation_type=annotation_type,
                agreement_score=1.0,
                majority_label=None,
                needs_review=False,
                annotator_count=0,
            )

        # Group by annotator
        by_annotator: dict[str, list[str]] = defaultdict(list)
        for ann in annotations:
            by_annotator[ann.annotator_id].append(ann.label)

        n_annotators = len(by_annotator)
        all_labels = [ann.label for ann in annotations]
        maj_label, _ = majority_label(all_labels)

        if n_annotators == 1:
            return AnnotationAgreement(
                conversation_id=conversation_id,
                annotation_type=annotation_type,
                agreement_score=1.0,
                majority_label=maj_label,
                needs_review=False,
                annotator_count=1,
            )

        # Use per-item ratings for Fleiss' kappa (one label per annotator per item)
        annotator_ids = list(by_annotator.keys())
        if n_annotators == 2:
            labels_a = by_annotator[annotator_ids[0]]
            labels_b = by_annotator[annotator_ids[1]]
            # Pad to same length
            max_len = max(len(labels_a), len(labels_b))
            labels_a = (labels_a + [labels_a[-1]] * max_len)[:max_len]
            labels_b = (labels_b + [labels_b[-1]] * max_len)[:max_len]
            kappa = cohen_kappa(labels_a, labels_b)
        else:
            # Build per-item rows: one label per annotator
            max_items = max(len(v) for v in by_annotator.values())
            rows = []
            for i in range(max_items):
                row = []
                for aid in annotator_ids:
                    labels = by_annotator[aid]
                    row.append(labels[i] if i < len(labels) else labels[-1])
                rows.append(row)
            kappa = fleiss_kappa(rows)

        needs_review = kappa < settings.min_annotation_agreement

        return AnnotationAgreement(
            conversation_id=conversation_id,
            annotation_type=annotation_type,
            agreement_score=round(float(kappa), 4),
            majority_label=maj_label,
            needs_review=needs_review,
            annotator_count=n_annotators,
        )

    def list_annotations(self, conversation_id: str) -> list[Annotation]:
        return (
            self.db.query(Annotation)
            .filter_by(conversation_id=conversation_id)
            .order_by(Annotation.created_at)
            .all()
        )

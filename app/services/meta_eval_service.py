"""
Meta-evaluation service.
Tracks how well automated evaluators align with human annotations.
Computes precision, recall, and correlation for each evaluator,
then stores the metrics in evaluator_metrics table.
"""
import logging
from collections import defaultdict
from datetime import datetime

try:
    import numpy as np
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from sqlalchemy.orm import Session

from app.models.db_models import Annotation, Evaluation, EvaluatorMetric

logger = logging.getLogger(__name__)

# Map annotation labels to numeric scores for correlation analysis
_LABEL_TO_SCORE = {
    "correct": 1.0, "good": 1.0, "positive": 1.0, "pass": 1.0,
    "incorrect": 0.0, "bad": 0.0, "negative": 0.0, "fail": 0.0,
    "partial": 0.5, "acceptable": 0.7,
}

# Map annotation types to evaluator score keys
_ANN_TYPE_TO_SCORE_KEY = {
    "tool_accuracy": "tool_accuracy",
    "response_quality": "response_quality",
    "coherence": "coherence",
    "overall": "overall",
}


class MetaEvalService:
    def __init__(self, db: Session):
        self.db = db

    def compute_and_store(self) -> dict:
        """
        For each annotation type that maps to an evaluator score:
        1. Collect (human_label, auto_score) pairs
        2. Compute Pearson correlation and simple precision/recall
        3. Persist results to evaluator_metrics
        """
        annotations = self.db.query(Annotation).all()
        if not annotations:
            return {}

        # Group by annotation_type
        by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for ann in annotations:
            score_key = _ANN_TYPE_TO_SCORE_KEY.get(ann.annotation_type)
            if not score_key:
                continue
            human_score = _LABEL_TO_SCORE.get(ann.label.lower())
            if human_score is None:
                continue

            # Find the evaluation for this conversation
            eval_row = (
                self.db.query(Evaluation)
                .filter_by(conversation_id=ann.conversation_id)
                .order_by(Evaluation.created_at.desc())
                .first()
            )
            if not eval_row or not eval_row.scores:
                continue

            auto_score = eval_row.scores.get(score_key, eval_row.overall_score)
            if auto_score is not None:
                by_type[ann.annotation_type].append((human_score, float(auto_score)))

        results = {}
        for ann_type, pairs in by_type.items():
            if len(pairs) < 3 or not _HAS_SCIPY:
                continue  # not enough data or scipy not available

            human_scores = np.array([p[0] for p in pairs])
            auto_scores = np.array([p[1] for p in pairs])

            # Pearson correlation
            if np.std(human_scores) > 0 and np.std(auto_scores) > 0:
                corr, _ = pearsonr(human_scores, auto_scores)
            else:
                corr = 1.0 if np.allclose(human_scores, auto_scores) else 0.0

            # Binary precision/recall (threshold: >= 0.7 = positive)
            threshold = 0.7
            human_pos = human_scores >= threshold
            auto_pos = auto_scores >= threshold
            tp = np.sum(human_pos & auto_pos)
            fp = np.sum(~human_pos & auto_pos)
            fn = np.sum(human_pos & ~auto_pos)
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            evaluator_name = _ANN_TYPE_TO_SCORE_KEY.get(ann_type, ann_type)
            sample_size = len(pairs)

            for metric_name, value in [
                ("correlation", float(corr)),
                ("precision", precision),
                ("recall", recall),
                ("f1", f1),
            ]:
                metric = EvaluatorMetric(
                    evaluator_name=evaluator_name,
                    metric_name=metric_name,
                    value=round(value, 4),
                    sample_size=sample_size,
                    computed_at=datetime.utcnow(),
                )
                self.db.add(metric)
                results[f"{evaluator_name}.{metric_name}"] = round(value, 4)

        self.db.commit()
        return results

    def get_latest_metrics(self) -> list[EvaluatorMetric]:
        """Return the most recent metric per evaluator+metric_name pair."""
        all_metrics = (
            self.db.query(EvaluatorMetric)
            .order_by(EvaluatorMetric.computed_at.desc())
            .all()
        )
        seen: set[str] = set()
        result = []
        for m in all_metrics:
            key = f"{m.evaluator_name}.{m.metric_name}"
            if key not in seen:
                seen.add(key)
                result.append(m)
        return result

    def check_evaluator_drift(self, min_correlation: float = 0.6) -> list[dict]:
        """Flag evaluators whose correlation with human labels has dropped below threshold."""
        metrics = self.get_latest_metrics()
        flagged = []
        for m in metrics:
            if m.metric_name == "correlation" and m.value < min_correlation:
                flagged.append({
                    "evaluator": m.evaluator_name,
                    "correlation": m.value,
                    "sample_size": m.sample_size,
                    "message": f"Evaluator '{m.evaluator_name}' correlation {m.value:.2f} below threshold {min_correlation}",
                })
        return flagged

"""
Celery tasks: async evaluation, suggestion generation, meta-eval computation.
"""
import logging
import uuid
from datetime import datetime

from celery import shared_task

from app.config import settings
from app.database import SessionLocal
from app.evaluators.coherence import CoherenceEvaluator
from app.evaluators.heuristic import HeuristicEvaluator
from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.evaluators.tool_evaluator import ToolCallEvaluator
from app.models.db_models import Conversation, Evaluation
from app.models.schemas import ConversationIngest

logger = logging.getLogger(__name__)

_heuristic = HeuristicEvaluator()
_tool_eval = ToolCallEvaluator()


@shared_task(bind=True, max_retries=3, default_retry_delay=30, name="app.workers.tasks.run_evaluation")
def run_evaluation(self, conversation_id: str):
    """Full evaluation pipeline for a single conversation."""
    db = SessionLocal()
    try:
        conv_row = db.query(Conversation).filter_by(conversation_id=conversation_id).first()
        if not conv_row:
            logger.error("Conversation %s not found", conversation_id)
            return

        conv_row.status = "evaluating"
        db.commit()

        conversation = ConversationIngest(
            conversation_id=conv_row.conversation_id,
            agent_version=conv_row.agent_version,
            turns=conv_row.turns,
            feedback=conv_row.feedback_data,
            metadata=conv_row.metadata_,
        )

        # --- run all evaluators ---
        heuristic_result = _heuristic.safe_evaluate(conversation)
        tool_result = _tool_eval.safe_evaluate(conversation)

        # LLM evaluators — these may fail if API key is missing
        try:
            llm_judge = LLMJudgeEvaluator()
            llm_result = llm_judge.safe_evaluate(conversation)
        except Exception as exc:
            logger.warning("Gemini LLM judge failed: %s", exc)
            llm_result = {"score": 0.5, "scores": {}, "issues": [], "prompt_suggestions": []}

        try:
            coherence_eval = CoherenceEvaluator()
            coherence_result = coherence_eval.safe_evaluate(conversation)
        except Exception as exc:
            logger.warning("Gemini coherence evaluator failed: %s", exc)
            coherence_result = {"score": 0.5, "scores": {}, "issues": [], "failures": []}

        # --- aggregate scores ---
        scores = {
            "response_quality": llm_result.get("score", 0.5),
            "tool_accuracy": tool_result.get("score", 1.0),
            "coherence": coherence_result.get("score", 0.5),
            "heuristic": heuristic_result.get("score", 1.0),
        }
        overall = (
            scores["response_quality"] * 0.35
            + scores["tool_accuracy"] * 0.30
            + scores["coherence"] * 0.20
            + scores["heuristic"] * 0.15
        )

        # --- merge issues ---
        all_issues = (
            heuristic_result.get("issues", [])
            + tool_result.get("issues", [])
            + llm_result.get("issues", [])
            + coherence_result.get("issues", [])
        )

        # --- collect inline suggestions ---
        inline_suggestions = [
            {"type": "prompt", **s}
            for s in llm_result.get("prompt_suggestions", [])
        ]

        evaluation = Evaluation(
            evaluation_id=f"eval_{uuid.uuid4().hex[:12]}",
            conversation_fk=conv_row.id,
            conversation_id=conversation_id,
            overall_score=round(overall, 4),
            scores=scores,
            tool_evaluation=tool_result.get("tool_evaluation"),
            issues=all_issues,
            improvement_suggestions=inline_suggestions,
        )
        db.add(evaluation)

        conv_row.status = "completed"
        db.commit()

        # Trigger auto-suggest if threshold reached
        total_evals = db.query(Evaluation).count()
        if total_evals % settings.auto_suggest_after_n_evals == 0:
            auto_generate_suggestions.delay()

        logger.info("Evaluation %s completed — overall: %.3f", evaluation.evaluation_id, overall)
        return evaluation.evaluation_id

    except Exception as exc:
        logger.exception("Evaluation failed for %s: %s", conversation_id, exc)
        try:
            conv_row.status = "failed"
            db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        db.close()


@shared_task(name="app.workers.tasks.auto_generate_suggestions")
def auto_generate_suggestions():
    """Batch job: analyze low-scored conversations and generate improvement suggestions."""
    from app.services.suggestion_service import SuggestionService
    db = SessionLocal()
    try:
        service = SuggestionService(db)
        count = service.generate_suggestions()
        logger.info("Auto-generated %d improvement suggestions", count)
        return count
    finally:
        db.close()


@shared_task(name="app.workers.tasks.compute_meta_eval_metrics")
def compute_meta_eval_metrics():
    """Periodic job: compare LLM judge scores vs human annotations."""
    from app.services.meta_eval_service import MetaEvalService
    db = SessionLocal()
    try:
        service = MetaEvalService(db)
        metrics = service.compute_and_store()
        logger.info("Meta-eval metrics computed: %s", metrics)
        return metrics
    finally:
        db.close()

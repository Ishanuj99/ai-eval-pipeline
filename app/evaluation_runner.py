"""
Synchronous evaluation runner — used on Vercel (no Celery workers).
Mirrors app/workers/tasks.py run_evaluation but runs in-process.
"""
import logging
import uuid

from sqlalchemy.orm import Session

from app.evaluators.coherence import CoherenceEvaluator
from app.evaluators.heuristic import HeuristicEvaluator
from app.evaluators.tool_evaluator import ToolCallEvaluator
from app.models.db_models import Conversation, Evaluation
from app.models.schemas import ConversationIngest

logger = logging.getLogger(__name__)

_heuristic = HeuristicEvaluator()
_tool_eval = ToolCallEvaluator()


def run_evaluation_sync(conversation_id: str, db: Session, fast_only: bool = False) -> str | None:
    conv_row = db.query(Conversation).filter_by(conversation_id=conversation_id).first()
    if not conv_row:
        return None

    conv_row.status = "evaluating"
    db.commit()

    conversation = ConversationIngest(
        conversation_id=conv_row.conversation_id,
        agent_version=conv_row.agent_version,
        turns=conv_row.turns,
        feedback=conv_row.feedback_data,
        metadata=conv_row.metadata_,
    )

    heuristic_result = _heuristic.safe_evaluate(conversation)
    tool_result = _tool_eval.safe_evaluate(conversation)

    if fast_only:
        # Vercel mode: skip LLM calls to stay within 10s timeout
        llm_result = {"score": 0.5, "scores": {}, "issues": [], "prompt_suggestions": []}
        coherence_result = {"score": 0.5, "scores": {}, "issues": [], "failures": []}
    else:
        try:
            from app.evaluators.llm_judge import LLMJudgeEvaluator
            llm_result = LLMJudgeEvaluator().safe_evaluate(conversation)
        except Exception as exc:
            logger.warning("LLM judge failed (sync): %s", exc)
            llm_result = {"score": 0.5, "scores": {}, "issues": [], "prompt_suggestions": []}

        try:
            from app.evaluators.coherence import CoherenceEvaluator
            coherence_result = CoherenceEvaluator().safe_evaluate(conversation)
        except Exception as exc:
            logger.warning("Coherence evaluator failed (sync): %s", exc)
            coherence_result = {"score": 0.5, "scores": {}, "issues": [], "failures": []}

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

    all_issues = (
        heuristic_result.get("issues", [])
        + tool_result.get("issues", [])
        + llm_result.get("issues", [])
        + coherence_result.get("issues", [])
    )
    inline_suggestions = [
        {"type": "prompt", **s} for s in llm_result.get("prompt_suggestions", [])
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
    return evaluation.evaluation_id

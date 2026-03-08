"""
Synchronous evaluation runner — used on Vercel (no Celery workers).
Mirrors app/workers/tasks.py run_evaluation but runs in-process.
"""
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from sqlalchemy.orm import Session

from app.evaluators.heuristic import HeuristicEvaluator
from app.evaluators.tool_evaluator import ToolCallEvaluator
from app.models.db_models import Conversation, Evaluation
from app.models.schemas import ConversationIngest

logger = logging.getLogger(__name__)

_heuristic = HeuristicEvaluator()
_tool_eval = ToolCallEvaluator()

# Single Gemini call budget — flash model typically responds in 2-4s
_LLM_TIMEOUT = 8


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

    # Instant evaluators — no LLM, always run
    heuristic_result = _heuristic.safe_evaluate(conversation)
    tool_result = _tool_eval.safe_evaluate(conversation)

    _llm_fallback = {"score": 0.5, "scores": {}, "coherence_score": 0.5, "issues": [], "prompt_suggestions": []}

    if fast_only:
        llm_result = _llm_fallback
    else:
        # Single Gemini call scores both quality AND coherence — avoids double-call latency
        try:
            from app.evaluators.llm_judge import LLMJudgeEvaluator
            _judge = LLMJudgeEvaluator()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_judge.safe_evaluate, conversation)
                try:
                    llm_result = future.result(timeout=_LLM_TIMEOUT) or _llm_fallback
                except (FuturesTimeoutError, Exception) as exc:
                    logger.warning("LLM judge timed out or failed: %s", exc)
                    llm_result = _llm_fallback
        except Exception as exc:
            logger.warning("LLM judge init failed: %s", exc)
            llm_result = _llm_fallback

    scores = {
        "response_quality": llm_result.get("score", 0.5),
        "tool_accuracy": tool_result.get("score", 1.0),
        "coherence": llm_result.get("coherence_score", 0.5),
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

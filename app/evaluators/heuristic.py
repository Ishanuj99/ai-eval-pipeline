from typing import Any

from app.config import settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import ConversationIngest, Issue


class HeuristicEvaluator(BaseEvaluator):
    """
    Fast rule-based checks: latency, required fields, empty responses,
    tool execution success rate.  No LLM calls — always runs synchronously.
    """

    name = "heuristic"

    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        issues: list[Issue] = []
        checks_passed = 0
        total_checks = 0

        # --- latency check ---
        total_checks += 1
        metadata = conversation.metadata or {}
        total_latency = metadata.get("total_latency_ms")
        if total_latency is not None:
            if total_latency > settings.latency_threshold_ms:
                issues.append(Issue(
                    type="latency",
                    severity="warning",
                    description=f"Total latency {total_latency}ms exceeds {settings.latency_threshold_ms}ms target",
                ))
            else:
                checks_passed += 1
        else:
            checks_passed += 1  # no data = not a failure

        # --- no empty assistant responses ---
        assistant_turns = [t for t in conversation.turns if t.role == "assistant"]
        total_checks += 1
        empty = [t for t in assistant_turns if not t.content.strip()]
        if empty:
            issues.append(Issue(
                type="empty_response",
                severity="error",
                description=f"{len(empty)} assistant turn(s) have empty content",
            ))
        else:
            checks_passed += 1

        # --- tool execution success ---
        all_tool_calls = [tc for t in conversation.turns for tc in t.tool_calls]
        if all_tool_calls:
            total_checks += 1
            failed = [tc for tc in all_tool_calls if tc.result.get("status") == "error"]
            if failed:
                issues.append(Issue(
                    type="tool_execution_failure",
                    severity="error",
                    description=f"{len(failed)}/{len(all_tool_calls)} tool call(s) failed",
                ))
            else:
                checks_passed += 1

        # --- per-tool latency ---
        for tc in all_tool_calls:
            if tc.latency_ms and tc.latency_ms > settings.latency_threshold_ms:
                issues.append(Issue(
                    type="tool_latency",
                    severity="warning",
                    description=f"Tool '{tc.tool_name}' latency {tc.latency_ms}ms exceeds threshold",
                ))

        # --- mission completed ---
        total_checks += 1
        mission_done = metadata.get("mission_completed")
        if mission_done is False:
            issues.append(Issue(
                type="mission_incomplete",
                severity="warning",
                description="Conversation ended without completing the user's goal",
            ))
        else:
            checks_passed += 1

        score = checks_passed / total_checks if total_checks else 1.0
        return {"score": round(score, 4), "issues": [i.model_dump() for i in issues]}

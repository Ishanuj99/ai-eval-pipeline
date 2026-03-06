from typing import Any

from app.evaluators.base import BaseEvaluator
from app.models.schemas import ConversationIngest, ToolEvaluation


class ToolCallEvaluator(BaseEvaluator):
    """
    Evaluates tool call quality without requiring ground-truth expectations:
    - Selection accuracy:  was any tool called when the user clearly needed one?
    - Parameter accuracy:  are parameter values traceable back to the conversation?
    - Execution success:   did the tool result indicate success?
    - Hallucination:       did parameters contain fabricated values?
    """

    name = "tool_call"

    # Words that strongly signal the user wants an action performed.
    _ACTION_KEYWORDS = {
        "book", "search", "find", "get", "show", "check", "buy", "order",
        "cancel", "schedule", "send", "create", "update", "delete", "fetch",
        "look", "retrieve", "calculate", "track", "compare",
    }

    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        all_tool_calls = [tc for t in conversation.turns for tc in t.tool_calls]

        if not all_tool_calls:
            # No tools were used — check if any were needed
            needed = self._user_needed_tool(conversation)
            if needed:
                return {
                    "score": 0.2,
                    "tool_evaluation": ToolEvaluation(
                        selection_accuracy=0.0,
                        parameter_accuracy=1.0,
                        execution_success=True,
                        hallucination_detected=False,
                    ).model_dump(),
                    "note": "Tool usage expected but no calls made",
                }
            return {
                "score": 1.0,
                "tool_evaluation": ToolEvaluation(
                    selection_accuracy=1.0,
                    parameter_accuracy=1.0,
                    execution_success=True,
                ).model_dump(),
            }

        selection_score = self._score_selection(conversation, all_tool_calls)
        param_score, hallucination = self._score_parameters(conversation, all_tool_calls)
        exec_success = all(
            tc.result.get("status") != "error" for tc in all_tool_calls
        )

        overall = (selection_score * 0.35 + param_score * 0.45 + (1.0 if exec_success else 0.0) * 0.2)

        return {
            "score": round(overall, 4),
            "tool_evaluation": ToolEvaluation(
                selection_accuracy=round(selection_score, 4),
                parameter_accuracy=round(param_score, 4),
                execution_success=exec_success,
                hallucination_detected=hallucination,
            ).model_dump(),
        }

    def _user_needed_tool(self, conversation: ConversationIngest) -> bool:
        user_text = " ".join(
            t.content.lower() for t in conversation.turns if t.role == "user"
        )
        return any(kw in user_text for kw in self._ACTION_KEYWORDS)

    def _score_selection(
        self, conversation: ConversationIngest, tool_calls: list
    ) -> float:
        needed = self._user_needed_tool(conversation)
        if needed and tool_calls:
            return 1.0
        if not needed and not tool_calls:
            return 1.0
        if needed and not tool_calls:
            return 0.0
        return 0.7  # tool called but might not have been needed (partial credit)

    def _score_parameters(self, conversation: ConversationIngest, tool_calls: list) -> tuple[float, bool]:
        """
        Heuristic: parameters are 'grounded' if their string values appear
        (or are paraphrases of) text already present in the conversation.
        """
        full_text = " ".join(t.content.lower() for t in conversation.turns).split()
        word_set = set(full_text)

        total_params = 0
        grounded = 0
        hallucination_detected = False

        for tc in tool_calls:
            for key, value in tc.parameters.items():
                if value is None:
                    continue
                total_params += 1
                val_str = str(value).lower()
                # Consider grounded if any token from the value appears in conversation
                val_tokens = val_str.split()
                if any(token in word_set for token in val_tokens if len(token) > 2):
                    grounded += 1
                else:
                    hallucination_detected = True

        if total_params == 0:
            return 1.0, False

        return grounded / total_params, hallucination_detected

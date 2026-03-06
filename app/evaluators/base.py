from abc import ABC, abstractmethod
from typing import Any

from app.models.schemas import ConversationIngest


class BaseEvaluator(ABC):
    """All evaluators implement this interface."""

    name: str = "base"
    version: str = "1.0.0"

    @abstractmethod
    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        """Return a dict with scores and optional metadata."""
        ...

    def safe_evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        """Wraps evaluate() with error handling — returns neutral score on failure."""
        try:
            return self.evaluate(conversation)
        except Exception as exc:
            return {
                "score": 0.5,
                "error": str(exc),
                "evaluator": self.name,
            }

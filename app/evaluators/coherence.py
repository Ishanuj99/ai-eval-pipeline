import json
from typing import Any

import google.generativeai as genai

from app.config import settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import ConversationIngest

_COHERENCE_PROMPT = """\
You are evaluating multi-turn conversation coherence for an AI assistant.

Conversation:
{conversation_text}

Evaluate these dimensions (score 0.0 to 1.0):
1. context_maintenance  – Does the agent remember and correctly use information from earlier turns?
2. consistency          – Does the agent contradict itself across turns?
3. reference_resolution – Does the agent correctly resolve pronouns and references (e.g., "it", "that", "the flight")?
4. topic_coherence      – Does the conversation flow logically without random topic jumps?

Flag specific coherence failures you find.

Respond ONLY with valid JSON:
{{
  "context_maintenance": <float 0-1>,
  "consistency": <float 0-1>,
  "reference_resolution": <float 0-1>,
  "topic_coherence": <float 0-1>,
  "failures": [
    {{"turn_id": <int>, "type": "<context_loss|contradiction|bad_reference|topic_jump>", "description": "<string>"}}
  ]
}}"""


class CoherenceEvaluator(BaseEvaluator):
    """
    Uses Gemini to measure multi-turn coherence:
    context maintenance, self-consistency, reference resolution, topic flow.
    Falls back to a heuristic score for single-turn conversations.
    """

    name = "coherence"

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.llm_judge_model)

    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        assistant_turns = [t for t in conversation.turns if t.role == "assistant"]

        if len(assistant_turns) <= 1:
            return {
                "score": 1.0,
                "scores": {
                    "context_maintenance": 1.0,
                    "consistency": 1.0,
                    "reference_resolution": 1.0,
                    "topic_coherence": 1.0,
                },
                "failures": [],
                "note": "single-turn — heuristic score applied",
            }

        conversation_text = self._format_conversation(conversation)
        prompt = _COHERENCE_PROMPT.format(conversation_text=conversation_text)

        response = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        scores = {
            "context_maintenance": float(data.get("context_maintenance", 0.5)),
            "consistency": float(data.get("consistency", 0.5)),
            "reference_resolution": float(data.get("reference_resolution", 0.5)),
            "topic_coherence": float(data.get("topic_coherence", 0.5)),
        }
        overall = sum(scores.values()) / len(scores)

        issues = []
        for failure in data.get("failures", []):
            issues.append({
                "type": "coherence_failure",
                "severity": "warning",
                "description": f"Turn {failure.get('turn_id')}: [{failure.get('type')}] {failure.get('description')}",
            })

        return {
            "score": round(overall, 4),
            "scores": scores,
            "issues": issues,
            "failures": data.get("failures", []),
        }

    @staticmethod
    def _format_conversation(conversation: ConversationIngest) -> str:
        lines = []
        for turn in conversation.turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"Turn {turn.turn_id} [{prefix}]: {turn.content}")
        return "\n".join(lines)

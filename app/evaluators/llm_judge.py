import json
from typing import Any

import google.generativeai as genai

from app.config import settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import ConversationIngest

_JUDGE_PROMPT = """\
You are an AI conversation evaluator. Score the conversation below across all dimensions.

Conversation:
{conversation_text}

Score every dimension from 0.0 to 1.0. Be concise.

Respond ONLY with this JSON, no extra text:
{{
  "response_quality": <float>,
  "helpfulness": <float>,
  "factuality": <float>,
  "tone_appropriateness": <float>,
  "context_maintenance": <float>,
  "consistency": <float>,
  "issues": [{{"type": "<str>", "severity": "info|warning|error", "description": "<str>"}}],
  "prompt_suggestions": [{{"suggestion": "<str>", "rationale": "<str>", "confidence": <float>}}]
}}"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Single Gemini call scoring both response quality and coherence dimensions."""

    name = "llm_judge"

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.llm_judge_model)

    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        conversation_text = self._format_conversation(conversation)
        prompt = _JUDGE_PROMPT.format(conversation_text=conversation_text)

        response = self._model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        quality_scores = {
            "response_quality": float(data.get("response_quality", 0.5)),
            "helpfulness": float(data.get("helpfulness", 0.5)),
            "factuality": float(data.get("factuality", 0.5)),
            "tone_appropriateness": float(data.get("tone_appropriateness", 0.5)),
        }
        overall = sum(quality_scores.values()) / len(quality_scores)

        # Also expose coherence scores so evaluation_runner can use them
        coherence_scores = {
            "context_maintenance": float(data.get("context_maintenance", 0.5)),
            "consistency": float(data.get("consistency", 0.5)),
        }
        coherence_overall = sum(coherence_scores.values()) / len(coherence_scores)

        return {
            "score": round(overall, 4),
            "scores": quality_scores,
            "coherence_score": round(coherence_overall, 4),
            "coherence_scores": coherence_scores,
            "issues": data.get("issues", []),
            "prompt_suggestions": data.get("prompt_suggestions", []),
        }

    @staticmethod
    def _format_conversation(conversation: ConversationIngest) -> str:
        lines = [f"[Agent version: {conversation.agent_version}]"]
        for turn in conversation.turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"\n{prefix} (turn {turn.turn_id}): {turn.content}")
            for tc in turn.tool_calls:
                status = tc.result.get("status", "unknown")
                lines.append(f"  [Tool: {tc.tool_name} | params: {tc.parameters} | status: {status}]")
        return "\n".join(lines)

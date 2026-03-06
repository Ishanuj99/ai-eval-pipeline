import json
from typing import Any

import anthropic

from app.config import settings
from app.evaluators.base import BaseEvaluator
from app.models.schemas import ConversationIngest

_JUDGE_PROMPT = """\
You are an expert AI conversation evaluator. Analyze the conversation below and score each dimension from 0.0 to 1.0.

Conversation:
{conversation_text}

Score these dimensions honestly:
1. response_quality     – Are responses helpful, accurate, and well-formed?
2. helpfulness          – Did the agent actually solve the user's problem?
3. factuality           – Are claims made by the agent factually plausible?
4. tone_appropriateness – Is the tone professional and appropriate?

Also identify up to 3 specific issues and suggest 1-2 prompt improvements if warranted.

Respond ONLY with valid JSON matching this schema exactly:
{{
  "response_quality": <float 0-1>,
  "helpfulness": <float 0-1>,
  "factuality": <float 0-1>,
  "tone_appropriateness": <float 0-1>,
  "issues": [
    {{"type": "<string>", "severity": "info|warning|error", "description": "<string>"}}
  ],
  "prompt_suggestions": [
    {{"suggestion": "<string>", "rationale": "<string>", "confidence": <float 0-1>}}
  ]
}}"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses Claude as an LLM-as-Judge to score response quality, helpfulness, and factuality."""

    name = "llm_judge"

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def evaluate(self, conversation: ConversationIngest) -> dict[str, Any]:
        conversation_text = self._format_conversation(conversation)
        prompt = _JUDGE_PROMPT.format(conversation_text=conversation_text)

        message = self._client.messages.create(
            model=settings.llm_judge_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        scores = {
            "response_quality": float(data.get("response_quality", 0.5)),
            "helpfulness": float(data.get("helpfulness", 0.5)),
            "factuality": float(data.get("factuality", 0.5)),
            "tone_appropriateness": float(data.get("tone_appropriateness", 0.5)),
        }
        overall = sum(scores.values()) / len(scores)

        return {
            "score": round(overall, 4),
            "scores": scores,
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

"""
Self-updating suggestion service.
Clusters low-scored evaluations, uses Claude to identify failure patterns,
and generates actionable prompt/tool improvement suggestions.
"""
import json
import logging
import uuid
from collections import defaultdict

import google.generativeai as genai
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import Evaluation, ImprovementSuggestion

logger = logging.getLogger(__name__)

_SUGGESTION_PROMPT = """\
You are an AI agent improvement specialist. Below are recent evaluation failures grouped by type.

Failure summary:
{failure_summary}

Sample failed conversation snippets:
{samples}

Generate specific, actionable improvement suggestions for the agent's prompts and/or tool schemas.

Respond ONLY with valid JSON:
{{
  "suggestions": [
    {{
      "type": "prompt|tool",
      "target": "<prompt name or tool name>",
      "suggestion": "<specific change to make>",
      "rationale": "<why this will help>",
      "expected_impact": "<what metric should improve>",
      "confidence": <float 0-1>
    }}
  ],
  "root_causes": ["<pattern 1>", "<pattern 2>"]
}}"""


class SuggestionService:
    def __init__(self, db: Session):
        self.db = db
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.llm_judge_model)

    def generate_suggestions(self, min_score: float = 0.7, limit: int = 50) -> int:
        """Analyze low-scored evaluations and generate suggestions. Returns count created."""
        if not settings.gemini_api_key:
            logger.warning("ANTHROPIC_API_KEY not set — skipping LLM suggestion generation")
            return 0

        low_evals = (
            self.db.query(Evaluation)
            .filter(Evaluation.overall_score < min_score)
            .order_by(Evaluation.created_at.desc())
            .limit(limit)
            .all()
        )

        if not low_evals:
            return 0

        failure_summary, samples = self._build_failure_context(low_evals)

        try:
            prompt = _SUGGESTION_PROMPT.format(
                failure_summary=failure_summary, samples=samples
            )
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, max_output_tokens=2048
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
        except Exception as exc:
            logger.error("Suggestion generation LLM call failed: %s", exc)
            return 0

        created = 0
        conv_ids = [e.conversation_id for e in low_evals[:5]]
        for s in data.get("suggestions", []):
            suggestion = ImprovementSuggestion(
                suggestion_id=f"sug_{uuid.uuid4().hex[:12]}",
                suggestion_type=s.get("type", "prompt"),
                target=s.get("target"),
                suggestion=s.get("suggestion", ""),
                rationale=s.get("rationale"),
                expected_impact=s.get("expected_impact"),
                confidence=s.get("confidence"),
                failure_patterns=data.get("root_causes", []),
                sample_conversation_ids=conv_ids,
            )
            self.db.add(suggestion)
            created += 1

        self.db.commit()
        return created

    def list_suggestions(self, status: str | None = None, limit: int = 50) -> list[ImprovementSuggestion]:
        q = self.db.query(ImprovementSuggestion)
        if status:
            q = q.filter_by(status=status)
        return q.order_by(ImprovementSuggestion.created_at.desc()).limit(limit).all()

    def update_status(self, suggestion_id: str, status: str) -> ImprovementSuggestion | None:
        s = self.db.query(ImprovementSuggestion).filter_by(suggestion_id=suggestion_id).first()
        if s:
            s.status = status
            self.db.commit()
        return s

    @staticmethod
    def _build_failure_context(evaluations: list[Evaluation]) -> tuple[str, str]:
        issue_counts: dict[str, int] = defaultdict(int)
        for ev in evaluations:
            for issue in (ev.issues or []):
                issue_counts[issue.get("type", "unknown")] += 1

        failure_summary = "\n".join(
            f"- {itype}: {count} occurrences"
            for itype, count in sorted(issue_counts.items(), key=lambda x: -x[1])
        ) or "No specific issues tagged"

        samples = []
        for ev in evaluations[:5]:
            samples.append(
                f"conv_id={ev.conversation_id} | score={ev.overall_score:.2f} | "
                f"issues={[i.get('type') for i in (ev.issues or [])]}"
            )
        return failure_summary, "\n".join(samples)

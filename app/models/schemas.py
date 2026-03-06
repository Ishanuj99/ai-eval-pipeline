from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------- Ingestion ----------

class ToolCall(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = {}
    result: dict[str, Any] = {}
    latency_ms: Optional[int] = None


class Turn(BaseModel):
    turn_id: int
    role: str  # user | assistant
    content: str
    tool_calls: list[ToolCall] = []
    timestamp: Optional[datetime] = None


class OpsReview(BaseModel):
    quality: Optional[str] = None
    notes: Optional[str] = None


class AnnotationIn(BaseModel):
    type: str
    label: str
    annotator_id: str
    confidence: float = 1.0
    notes: Optional[str] = None


class FeedbackIn(BaseModel):
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    ops_review: Optional[OpsReview] = None
    annotations: list[AnnotationIn] = []


class ConversationMetadata(BaseModel):
    total_latency_ms: Optional[int] = None
    mission_completed: Optional[bool] = None
    extra: dict[str, Any] = {}

    model_config = {"extra": "allow"}


class ConversationIngest(BaseModel):
    conversation_id: str
    agent_version: str
    turns: list[Turn]
    feedback: Optional[FeedbackIn] = None
    metadata: Optional[dict[str, Any]] = None


# ---------- Evaluation Output ----------

class ToolEvaluation(BaseModel):
    selection_accuracy: float
    parameter_accuracy: float
    execution_success: bool
    hallucination_detected: bool = False


class Issue(BaseModel):
    type: str
    severity: str  # info | warning | error
    description: str


class SuggestionOut(BaseModel):
    type: str           # prompt | tool
    suggestion: str
    rationale: str
    confidence: float


class EvaluationScores(BaseModel):
    response_quality: float
    tool_accuracy: float
    coherence: float
    heuristic: float


class EvaluationResult(BaseModel):
    evaluation_id: str
    conversation_id: str
    scores: EvaluationScores
    overall: float
    tool_evaluation: Optional[ToolEvaluation] = None
    issues: list[Issue] = []
    improvement_suggestions: list[SuggestionOut] = []
    evaluator_version: str = "1.0.0"
    created_at: Optional[datetime] = None


# ---------- Feedback / Annotation ----------

class AnnotationCreate(BaseModel):
    conversation_id: str
    annotator_id: str
    annotation_type: str
    label: str
    confidence: float = 1.0
    notes: Optional[str] = None


class AnnotationAgreement(BaseModel):
    conversation_id: str
    annotation_type: str
    agreement_score: float        # Cohen's / Fleiss' Kappa
    majority_label: Optional[str]
    needs_review: bool
    annotator_count: int


# ---------- Improvement Suggestions ----------

class ImprovementSuggestionOut(BaseModel):
    suggestion_id: str
    suggestion_type: str
    target: Optional[str]
    suggestion: str
    rationale: Optional[str]
    expected_impact: Optional[str]
    confidence: Optional[float]
    failure_patterns: Optional[list[str]]
    sample_conversation_ids: Optional[list[str]]
    status: str
    created_at: Optional[datetime]


# ---------- Meta-Evaluation ----------

class EvaluatorMetricOut(BaseModel):
    evaluator_name: str
    metric_name: str
    value: float
    sample_size: Optional[int]
    computed_at: Optional[datetime]


# ---------- Dashboard ----------

class DashboardStats(BaseModel):
    total_conversations: int
    total_evaluations: int
    avg_overall_score: float
    avg_response_quality: float
    avg_tool_accuracy: float
    avg_coherence: float
    pending_conversations: int
    open_suggestions: int
    recent_issues: list[Issue] = []

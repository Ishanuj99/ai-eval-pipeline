import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String, unique=True, nullable=False, index=True)
    agent_version = Column(String, nullable=False)
    turns = Column(JSON, nullable=False)
    feedback_data = Column(JSON)
    metadata_ = Column("metadata", JSON)
    status = Column(String, default="pending")  # pending | evaluating | completed | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    evaluations = relationship("Evaluation", back_populates="conversation", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="conversation", cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(String, unique=True, nullable=False, index=True)
    conversation_fk = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    conversation_id = Column(String, index=True)  # denormalized for easy querying
    overall_score = Column(Float)
    scores = Column(JSON)           # {response_quality, tool_accuracy, coherence, heuristic}
    tool_evaluation = Column(JSON)  # {selection_accuracy, parameter_accuracy, execution_success}
    issues = Column(JSON)           # [{type, severity, description}]
    improvement_suggestions = Column(JSON)  # [{type, suggestion, rationale, confidence}]
    evaluator_version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="evaluations")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_fk = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    conversation_id = Column(String, index=True)
    annotator_id = Column(String, nullable=False, index=True)
    annotation_type = Column(String, nullable=False)  # tool_accuracy | response_quality | coherence
    label = Column(String, nullable=False)            # correct | incorrect | good | bad | etc.
    confidence = Column(Float, default=1.0)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="annotations")


class ImprovementSuggestion(Base):
    __tablename__ = "improvement_suggestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    suggestion_id = Column(String, unique=True, nullable=False, index=True)
    suggestion_type = Column(String, nullable=False)  # prompt | tool
    target = Column(String)                           # which prompt/tool
    suggestion = Column(String, nullable=False)
    rationale = Column(String)
    expected_impact = Column(String)
    confidence = Column(Float)
    failure_patterns = Column(JSON)
    sample_conversation_ids = Column(JSON)
    status = Column(String, default="pending")  # pending | applied | rejected
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluatorMetric(Base):
    __tablename__ = "evaluator_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluator_name = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False)  # precision | recall | f1 | correlation | kappa
    value = Column(Float, nullable=False)
    sample_size = Column(Integer)
    computed_at = Column(DateTime, default=datetime.utcnow)

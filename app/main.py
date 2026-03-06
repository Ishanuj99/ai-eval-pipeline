import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import conversations, evaluations, feedback, suggestions
from app.config import settings
from app.database import init_db

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent Evaluation Pipeline",
    description="Automated pipeline for evaluating, monitoring, and improving AI agents in production.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)
app.include_router(evaluations.router)
app.include_router(feedback.router)
app.include_router(suggestions.router)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database ready.")


@app.get("/", tags=["health"])
def root():
    return {"service": "AI Agent Evaluation Pipeline", "version": "1.0.0", "status": "ok"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

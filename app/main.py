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
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as e:
        logger.error("Database init failed (will retry on first request): %s", e)


@app.get("/", tags=["health"])
def root():
    return {"service": "AI Agent Evaluation Pipeline", "version": "1.0.0", "status": "ok"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/debug/gemini", tags=["health"])
def debug_gemini():
    """Test Gemini connectivity and return the actual error if it fails."""
    import time
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.llm_judge_model)
        t0 = time.time()
        response = model.generate_content(
            "Reply with this JSON only: {\"ok\": true}",
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=20),
        )
        elapsed = round(time.time() - t0, 2)
        return {"status": "ok", "model": settings.llm_judge_model, "elapsed_s": elapsed, "response": response.text.strip()}
    except Exception as e:
        return {"status": "error", "model": settings.llm_judge_model, "error": str(e), "api_key_set": bool(settings.gemini_api_key)}

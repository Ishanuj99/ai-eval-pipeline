from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "eval_pipeline",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Periodic tasks
    beat_schedule={
        "auto-generate-suggestions": {
            "task": "app.workers.tasks.auto_generate_suggestions",
            "schedule": crontab(minute="*/30"),  # every 30 minutes
        },
        "compute-meta-eval-metrics": {
            "task": "app.workers.tasks.compute_meta_eval_metrics",
            "schedule": crontab(minute="0", hour="*/2"),  # every 2 hours
        },
    },
)

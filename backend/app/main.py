"""FastAPI application entry point.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyze, broadcast, evaluation, meta, news, statistics
from app.config import settings
from app.database.session import init_db

# Thai text in log output breaks on Windows' default cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create tables on startup so a fresh clone works with no extra step."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Build the application (factory keeps tests independent of import order)."""
    app = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "NLP-powered Thai news topic & sentiment analysis. "
            "Ingests Thai YouTube live chat and pasted articles, then classifies "
            "topic, analyses sentiment, extracts keywords and generates summaries."
        ),
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta.router, prefix=settings.api_prefix)
    app.include_router(news.router, prefix=settings.api_prefix)
    app.include_router(statistics.router, prefix=settings.api_prefix)
    app.include_router(analyze.router, prefix=settings.api_prefix)
    app.include_router(evaluation.router, prefix=settings.api_prefix)
    app.include_router(broadcast.router, prefix=settings.api_prefix)

    # Captured video frames are served straight from the data directory.
    # They are files rather than database blobs because they are static,
    # cacheable and regenerable, and because SQLite is a poor image store.
    frames_dir = settings.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media/frames",
        StaticFiles(directory=frames_dir),
        name="frames",
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "api": settings.api_prefix,
        }

    return app


app = create_app()

"""FastAPI application entry point.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyze, meta, news, statistics
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

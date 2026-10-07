from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings

app = FastAPI(
    title="HumanIntent API",
    description="Intent-Preserving Adaptive Prompt Optimization System - Phase 1 Foundation",
    version="0.1.0",
)

# Configure CORS for local Next.js frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes with /api prefix
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "HumanIntent API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }

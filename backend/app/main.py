from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import admin, chat, health
from app.services.app_factory import initialize_app_state, shutdown_app_state

settings = get_settings()

app = FastAPI(title="Vin Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup() -> None:
    await initialize_app_state(app, settings)


@app.on_event("shutdown")
async def shutdown() -> None:
    await shutdown_app_state(app)

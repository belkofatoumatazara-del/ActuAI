from fastapi import FastAPI

from actuai_backend.src.api.routers import hitl, triggers, webhooks
from actuai_backend.src.database.connection import init_db

app = FastAPI(
    title="ActuAI Backend",
    description="API de l'orchestrateur LangGraph et Datalake",
)


@app.on_event("startup")
def on_startup():
    print("🚀 Démarrage du backend ActuAI...")
    init_db()


# Inclusion des routeurs
app.include_router(triggers.router, prefix="/api/triggers", tags=["Triggers"])
app.include_router(hitl.router, prefix="/api/hitl", tags=["Human in the Loop"])
# Alias de compatibilité : chemin de webhook documenté par actuai_mock_data (.env WEBHOOK_TARGET_URL)
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])


@app.get("/health")
def health_check():
    return {"status": "ok", "system": "ActuAI Backend Online"}

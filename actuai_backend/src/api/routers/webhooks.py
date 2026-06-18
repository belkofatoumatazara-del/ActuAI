"""
Routeur Webhooks — compatibilité avec le simulateur MS Exchange.

Le module `actuai_mock_data` (generators/emails.py) envoie les emails fournisseurs
vers l'URL documentée dans son `.env` :
    WEBHOOK_TARGET_URL=http://localhost:8000/api/v1/webhooks/exchange

Ce routeur expose ce chemin et réutilise exactement la même logique d'ingestion
que `POST /api/triggers/email`, afin que le flux généré arrive sur le graphe sans
modifier la configuration du module mock.
"""

from fastapi import APIRouter, Depends

from actuai_backend.src.api.dependencies import get_graph
from actuai_backend.src.api.routers.triggers import _start_run
from actuai_backend.src.api.schemas import EmailTrigger, RunResponse

router = APIRouter()


@router.post("/exchange", response_model=RunResponse)
def receive_exchange_email(email: EmailTrigger, graph=Depends(get_graph)):
    """Réceptionne un email fournisseur simulé (webhook MS Exchange)."""
    return _start_run(graph, "supplier_email", email.model_dump(), email.message_id)

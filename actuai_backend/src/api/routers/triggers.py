"""
Routeur d'ingestion : transforme un événement opérationnel en cycle d'orchestration.

Chaque endpoint dépose la donnée brute dans l'État Global et lance le graphe. Le
graphe s'exécute jusqu'à la pause HITL (`interrupt`) puis renvoie le brouillon en
attente de validation. La reprise se fait via le routeur `hitl`.
"""

import uuid

from fastapi import APIRouter, Depends

from actuai_backend.src.api.dependencies import get_graph
from actuai_backend.src.api.schemas import (
    AuditTrigger,
    DiscrepancyTrigger,
    EmailTrigger,
    RunResponse,
)

router = APIRouter()


def _start_run(graph, trigger_type: str, raw_input: dict, thread_id: str) -> RunResponse:
    """Lance le graphe jusqu'à la pause HITL et formate la réponse."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"trigger_type": trigger_type, "raw_input": raw_input}, config=config
    )

    # Si le graphe s'est interrompu, un brouillon attend la validation humaine.
    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        return RunResponse(
            thread_id=thread_id,
            status="pending_validation",
            route=result.get("route"),
            draft=payload.get("draft"),
        )

    # Sinon, le cycle est terminé (cas rare : pas de brouillon).
    return RunResponse(
        thread_id=thread_id,
        status="completed",
        route=result.get("route"),
        draft=result.get("draft_action"),
        execution_result=result.get("execution_result"),
    )


@router.post("/email", response_model=RunResponse)
def trigger_email(email: EmailTrigger, graph=Depends(get_graph)):
    """Ingestion d'un email fournisseur (Missions 1-3)."""
    return _start_run(graph, "supplier_email", email.model_dump(), email.message_id)


@router.post("/erp-discrepancy", response_model=RunResponse)
def trigger_discrepancy(payload: DiscrepancyTrigger, graph=Depends(get_graph)):
    """Ingestion d'une discordance de planning ERP (Mission 2)."""
    thread_id = f"disc-{payload.po_number}-{uuid.uuid4().hex[:6]}"
    return _start_run(graph, "erp_discrepancy", payload.model_dump(), thread_id)


@router.post("/audit", response_model=RunResponse)
def trigger_audit(payload: AuditTrigger, graph=Depends(get_graph)):
    """Ingestion d'une requête d'audit de traçabilité (Missions 4-5)."""
    thread_id = f"audit-{uuid.uuid4().hex[:8]}"
    return _start_run(graph, "compliance_audit", payload.model_dump(), thread_id)

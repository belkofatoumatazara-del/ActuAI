"""
Routeur Human-in-the-Loop : consultation et validation des brouillons en attente.

Le frontend React consulte le brouillon mis en pause (`GET /pending/{thread_id}`)
puis transmet la décision de l'expert (`POST /{thread_id}/validate`). Une
validation positive fait reprendre le graphe jusqu'à l'écriture SAP ; un refus
clôt le cycle sans aucune écriture.
"""

from fastapi import APIRouter, Depends, HTTPException

from actuai_backend.src.agents.graph import Command
from actuai_backend.src.api.dependencies import get_graph
from actuai_backend.src.api.schemas import RunResponse, ValidationRequest

router = APIRouter()


@router.get("/pending/{thread_id}", response_model=RunResponse)
def get_pending(thread_id: str, graph=Depends(get_graph)):
    """Renvoie le brouillon actuellement en attente de validation pour ce thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if not snapshot.next or "human_review" not in snapshot.next:
        raise HTTPException(
            status_code=404,
            detail="Aucun brouillon en attente de validation pour ce thread.",
        )

    values = snapshot.values
    return RunResponse(
        thread_id=thread_id,
        status="pending_validation",
        route=values.get("route"),
        draft=values.get("draft_action"),
    )


@router.post("/{thread_id}/validate", response_model=RunResponse)
def validate(thread_id: str, decision: ValidationRequest, graph=Depends(get_graph)):
    """Reprend le graphe avec la décision humaine (validation ou refus)."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.next or "human_review" not in snapshot.next:
        raise HTTPException(status_code=409, detail="Ce thread n'attend pas de validation.")

    result = graph.invoke(
        Command(resume={"approved": decision.approved, "comment": decision.comment}),
        config=config,
    )

    status = "completed" if decision.approved else "rejected"
    return RunResponse(
        thread_id=thread_id,
        status=status,
        route=result.get("route"),
        draft=result.get("draft_action"),
        execution_result=result.get("execution_result"),
    )

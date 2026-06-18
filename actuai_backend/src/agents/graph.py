"""
Définition du graphe LangGraph d'ActuAI (orchestration multi-agents).

Topologie :

    START
      │
      ▼
   superviseur ──(route)──► transactionnel ─┐
      │                                      ├─► human_review ──(approuvé)──► execute ─► END
      └────────► investigateur ──────────────┘        │
                                                       └──(rejeté)──────────────────────► END

Le nœud `human_review` met le graphe EN PAUSE via `interrupt()` : l'exécution
s'arrête, le brouillon est exposé à l'expert humain (frontend / API HITL), et ne
reprend qu'avec une décision explicite. Seule une validation déclenche l'écriture
réelle dans SAP par le nœud `execute`.
"""

import os
from typing import Optional

import requests
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from actuai_backend.src.agents.investigative import investigative_node
from actuai_backend.src.agents.state import GraphState
from actuai_backend.src.agents.supervisor import supervisor_node
from actuai_backend.src.agents.transactional import transactional_node

SAP_MOCK_URL = os.getenv("ACTUAI_SAP_MOCK_URL", "http://localhost:8080")


# --- Nœud de validation humaine (Human-in-the-Loop) -----------------------

def human_review_node(state: GraphState) -> dict:
    """Met le graphe en pause et attend la décision de l'expert humain.

    `interrupt()` renvoie la valeur fournie à la reprise via
    `Command(resume=...)`. On accepte soit un booléen, soit un dict
    {"approved": bool, "comment": str}.
    """
    decision = interrupt(
        {
            "type": "hitl_validation",
            "route": state.get("route"),
            "draft": state.get("draft_action"),
        }
    )

    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
        comment = decision.get("comment")
    else:
        approved = bool(decision)
        comment = None

    return {
        "hitl_status": "approved" if approved else "rejected",
        "human_comment": comment,
    }


# --- Nœud d'exécution réelle (écriture SAP après validation) --------------

def execute_node(state: GraphState) -> dict:
    """Exécute l'action validée contre le mock SAP (PUT/POST)."""
    draft = state.get("draft_action") or {}
    execution = draft.get("execution", {})
    method = str(execution.get("method", "NONE")).upper()

    # Pas d'écriture pour les dossiers documentaires ou les brouillons hors-ligne.
    if method in ("NONE", "") or draft.get("engine") == "deterministic-offline":
        return {
            "execution_result": {
                "status": "skipped",
                "reason": "Aucune écriture SAP requise (dossier/archive ou mode hors-ligne).",
                "action_type": draft.get("action_type"),
            }
        }

    url = f"{SAP_MOCK_URL}{execution.get('path', '')}"
    try:
        if method == "PUT":
            resp = requests.put(url, params=execution.get("params"), timeout=10)
        elif method == "POST":
            resp = requests.post(url, json=execution.get("json"), timeout=10)
        else:
            return {"execution_result": {"status": "error", "reason": f"Méthode non gérée : {method}"}}
        resp.raise_for_status()
        return {
            "execution_result": {
                "status": "success",
                "http_status": resp.status_code,
                "sap_response": resp.json(),
            }
        }
    except requests.exceptions.RequestException as e:
        return {"execution_result": {"status": "error", "reason": str(e)}}


# --- Sélecteurs d'arêtes conditionnelles ----------------------------------

def route_selector(state: GraphState) -> str:
    """Aiguille vers l'agent spécialiste choisi par le Superviseur."""
    return state.get("route") or "transactional"


def hitl_selector(state: GraphState) -> str:
    """Décide d'exécuter ou de terminer selon la validation humaine."""
    return "execute" if state.get("hitl_status") == "approved" else "end"


# --- Construction du graphe ------------------------------------------------

def build_graph(checkpointer: Optional[object] = None):
    """Assemble et compile le graphe.

    Un checkpointer est INDISPENSABLE pour la pause HITL : il persiste l'état du
    thread entre l'interruption et la reprise. Par défaut, MemorySaver (mémoire
    du process) ; en production, utiliser un checkpointer PostgreSQL.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("transactional", transactional_node)
    workflow.add_node("investigative", investigative_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("execute", execute_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_selector,
        {"transactional": "transactional", "investigative": "investigative"},
    )
    workflow.add_edge("transactional", "human_review")
    workflow.add_edge("investigative", "human_review")
    workflow.add_conditional_edges(
        "human_review", hitl_selector, {"execute": "execute", "end": END}
    )
    workflow.add_edge("execute", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


# Graphe compilé partagé par l'application (singleton de module).
graph = build_graph()

__all__ = ["build_graph", "graph", "Command"]

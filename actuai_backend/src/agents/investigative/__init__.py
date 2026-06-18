"""
Agent Investigateur — Données non structurées & outils RAG (Missions 4 et 5).

Équipé d'outils de Retrieval-Augmented Generation sur la base vectorielle Qdrant,
cet agent traite les requêtes documentaires :
  - Mission 4 : récupération de la bonne version d'un document technique.
  - Mission 5 : reconstitution de l'historique end-to-end d'un composant (traçabilité).

Il agrège les sources structurées (datalake SAP) et non structurées (documents
indexés) pour produire un BROUILLON de dossier soumis à validation humaine.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from actuai_backend.src.agents._runtime import (
    build_task_prompt,
    extract_draft,
    parse_po_number,
)
from actuai_backend.src.agents.investigative.tools import INVESTIGATIVE_TOOLS
from actuai_backend.src.agents.llm import get_investigative_llm
from actuai_backend.src.agents.state import GraphState

_SYSTEM_PROMPT = (
    "You are the Investigative Agent of ActuAI, specialised in unstructured "
    "aerospace documentation. Use search_technical_documents to locate the correct "
    "document version, and reconstruct_traceability to rebuild a component's "
    "end-to-end history for a compliance audit. You MUST finish by calling "
    "reconstruct_traceability to produce the dossier draft for human validation. "
    "Never invent document content — rely only on retrieved evidence."
)


def _offline_draft(state: GraphState) -> dict[str, Any]:
    """Brouillon déterministe (sans LLM ni Qdrant) pour le mode hors-ligne."""
    raw = state.get("raw_input", {})
    po = parse_po_number(raw) or raw.get("po_number") or "PO-INCONNU"
    return {
        "action_type": "TRACEABILITY_DOSSIER",
        "agent": "investigative",
        "mission": "5 - Component Traceability",
        "engine": "deterministic-offline",
        "summary": f"Squelette de dossier de traçabilité pour {po} (RAG ignoré, mode hors-ligne).",
        "sap_record": {"po_number": po, "note": "À compléter via le datalake."},
        "document_evidence": [],
        "execution": {"method": "NONE", "note": "Archivage du dossier après validation."},
    }


def investigative_node(state: GraphState) -> dict:
    """Nœud LangGraph : produit un brouillon de dossier documentaire / traçabilité."""
    llm = get_investigative_llm()
    if llm is None:
        draft = _offline_draft(state)
        return {"draft_action": draft}

    agent = create_react_agent(llm, INVESTIGATIVE_TOOLS, prompt=_SYSTEM_PROMPT)
    result = agent.invoke({"messages": [HumanMessage(build_task_prompt(state))]})
    draft = extract_draft(result["messages"], fallback_agent="investigative")
    return {"messages": result["messages"], "draft_action": draft}

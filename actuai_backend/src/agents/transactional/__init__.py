"""
Agent Transactionnel — Données structurées & outils SQL (Missions 1 à 3).

Équipé d'outils SQL sur le datalake PostgreSQL, cet agent traite les tâches à
forte exigence de précision logique :
  - Mission 1 : synchronisation des dates de livraison depuis les emails (payload SAP).
  - Mission 2 : croisement livraison / planning d'assemblage (risque AOG).
  - Mission 3 : pré-remplissage des Fiches de Non-Conformité (FNC).

Il ne produit que des BROUILLONS ; l'écriture SAP est réalisée plus tard, après
validation humaine, par le nœud d'exécution.
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from actuai_backend.src.agents._runtime import (
    build_task_prompt,
    extract_draft,
    parse_po_number,
)
from actuai_backend.src.agents.llm import get_transactional_llm
from actuai_backend.src.agents.state import GraphState
from actuai_backend.src.agents.transactional.tools import TRANSACTIONAL_TOOLS

_SYSTEM_PROMPT = (
    "You are the Transactional Agent of ActuAI, specialised in structured SAP data "
    "for an aerospace Actuation department. Use your SQL tools to gather context "
    "(get_purchase_order, check_delivery_risk). To synchronise a delivery date use "
    "draft_sap_date_update; to document a quality defect use draft_fnc. You MUST "
    "finish by calling exactly one draft_* tool. Never claim to have written to SAP "
    "— you only prepare drafts for human validation."
)

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DEFECT_HINTS = ("défaut", "defect", "non-conform", "qualité", "quality", "8d", "fnc")


def _offline_draft(state: GraphState) -> dict[str, Any]:
    """Brouillon déterministe (sans LLM ni base de données) pour le mode hors-ligne."""
    raw = state.get("raw_input", {})
    po = parse_po_number(raw) or "PO-INCONNU"
    blob = json.dumps(raw, ensure_ascii=False).lower()

    date_match = _DATE_RE.search(json.dumps(raw, ensure_ascii=False))
    if date_match:
        return {
            "action_type": "SAP_DATE_UPDATE",
            "agent": "transactional",
            "mission": "1 - Supply Chain Monitoring",
            "engine": "deterministic-offline",
            "summary": f"Repousser la date de livraison de {po} au {date_match.group(1)}.",
            "execution": {
                "method": "PUT",
                "path": f"/api/bapi/purchase-orders/{po}/update-date",
                "params": {"new_date": date_match.group(1)},
            },
        }
    if any(h in blob for h in _DEFECT_HINTS):
        return {
            "action_type": "FNC_CREATION",
            "agent": "transactional",
            "mission": "3 - Quality & Non-Conformity Management",
            "engine": "deterministic-offline",
            "summary": f"Brouillon de FNC à compléter pour la commande {po}.",
            "execution": {
                "method": "POST",
                "path": "/api/bapi/quality-notifications/",
                "json": {"po_number": po, "defect_type": "À préciser", "report_8d_status": "PENDING"},
            },
        }
    return {
        "action_type": "REVIEW_REQUIRED",
        "agent": "transactional",
        "engine": "deterministic-offline",
        "summary": f"Entrée relative à {po} nécessitant une revue manuelle (aucun LLM disponible).",
    }


def transactional_node(state: GraphState) -> dict:
    """Nœud LangGraph : produit un brouillon d'action transactionnelle."""
    llm = get_transactional_llm()
    if llm is None:
        draft = _offline_draft(state)
        return {"draft_action": draft}

    agent = create_react_agent(llm, TRANSACTIONAL_TOOLS, prompt=_SYSTEM_PROMPT)
    result = agent.invoke({"messages": [HumanMessage(build_task_prompt(state))]})
    draft = extract_draft(result["messages"], fallback_agent="transactional")
    return {"messages": result["messages"], "draft_action": draft}

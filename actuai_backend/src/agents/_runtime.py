"""
Utilitaires partagés par les agents spécialistes (Transactionnel & Investigateur).

- `build_task_prompt` : transforme l'état global en une consigne initiale lisible.
- `extract_draft`     : récupère le brouillon structuré produit par un outil `draft_*`
                        dans l'historique de messages de l'agent ReAct.
- `parse_po_number`   : extraction déterministe d'un numéro de commande (mode offline).
"""

import json
import re
from typing import Any, Optional

from langchain_core.messages import AIMessage, ToolMessage

# Outils dont la sortie constitue le brouillon final soumis au HITL.
DRAFT_TOOL_NAMES = {
    "draft_sap_date_update",
    "draft_fnc",
    "reconstruct_traceability",
}

_PO_RE = re.compile(r"PO-\d{4,6}")


def parse_po_number(raw_input: dict[str, Any]) -> Optional[str]:
    """Extrait un numéro de commande (PO-XXXXXX) du payload brut."""
    blob = " ".join(str(v) for v in raw_input.values())
    match = _PO_RE.search(blob)
    return match.group(0) if match else None


def build_task_prompt(state: dict[str, Any]) -> str:
    """Construit la consigne initiale à partir du trigger et de la donnée brute."""
    trigger = state.get("trigger_type", "inconnu")
    raw = json.dumps(state.get("raw_input", {}), ensure_ascii=False, indent=2)
    return (
        f"Type de déclencheur : {trigger}\n"
        f"Donnée brute reçue :\n{raw}\n\n"
        "Analyse cette entrée, utilise tes outils pour rassembler le contexte "
        "nécessaire, puis termine OBLIGATOIREMENT par l'appel d'un outil de "
        "brouillon (draft_*) afin de produire l'action à valider par l'humain."
    )


def extract_draft(messages: list, fallback_agent: str) -> dict[str, Any]:
    """Récupère le dernier brouillon JSON produit par un outil de type draft_*."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) in DRAFT_TOOL_NAMES:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return {"action_type": "RAW", "agent": fallback_agent, "content": str(msg.content)}

    # Repli : aucun brouillon -> on remonte le dernier message texte de l'agent.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {
                "action_type": "INFORMATION",
                "agent": fallback_agent,
                "summary": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
    return {"action_type": "EMPTY", "agent": fallback_agent}

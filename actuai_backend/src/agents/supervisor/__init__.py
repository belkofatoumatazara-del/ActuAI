"""
Agent Superviseur — Routeur sémantique (point d'entrée unique).

Rôle (cf. rapport) : analyser l'État Global et router le flux vers l'agent
spécialiste adéquat. Il n'exécute AUCUNE logique métier et n'accède à AUCUNE base
de données ; il se contente d'un appel LLM rapide d'intent classification.

- Données structurées / actions ERP (emails fournisseurs, discordances de date,
  qualité)            -> agent TRANSACTIONNEL.
- Recherche documentaire / reconstitution d'historique (audits de conformité)
                      -> agent INVESTIGATEUR.

Repli déterministe : si aucun LLM local n'est disponible (mode offline / Ollama
absent), le routage s'effectue par règles sur le type de trigger et des mots-clés.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from actuai_backend.src.agents.llm import get_supervisor_llm
from actuai_backend.src.agents.state import GraphState

_SYSTEM_PROMPT = (
    "You are the Supervisor of the ActuAI multi-agent system for an aerospace "
    "Actuation department. Classify each incoming trigger and route it to exactly "
    "one specialist worker:\n"
    "- 'transactional': structured-data tasks on the SAP datalake — supplier "
    "delivery updates, ERP date discrepancies, and quality non-conformance (FNC). "
    "Missions 1-3.\n"
    "- 'investigative': unstructured-data tasks via document retrieval (RAG) — "
    "finding the right document version and reconstructing component traceability "
    "for compliance audits. Missions 4-5.\n"
    "Respond only with the routing decision."
)

# Mots-clés (FR/EN) indiquant un besoin de recherche documentaire / traçabilité.
_INVESTIGATIVE_HINTS = (
    "audit", "traç", "trace", "traceability", "historique", "history",
    "document", "certificat", "certificate", "version", "dossier",
)


class RouteDecision(BaseModel):
    """Décision de routage structurée renvoyée par le Superviseur."""

    destination: Literal["transactional", "investigative"] = Field(
        description="L'agent spécialiste cible."
    )
    reason: str = Field(description="Justification courte du routage.")


def _rule_based_route(state: GraphState) -> RouteDecision:
    """Routage déterministe de secours (sans LLM)."""
    trigger = state.get("trigger_type")
    if trigger == "compliance_audit":
        return RouteDecision(
            destination="investigative",
            reason="Audit de conformité -> recherche documentaire / traçabilité (règle).",
        )
    if trigger in ("supplier_email", "erp_discrepancy"):
        blob = " ".join(str(v) for v in state.get("raw_input", {}).values()).lower()
        if any(h in blob for h in _INVESTIGATIVE_HINTS):
            return RouteDecision(
                destination="investigative",
                reason="Mots-clés documentaires détectés dans le trigger (règle).",
            )
        return RouteDecision(
            destination="transactional",
            reason="Email/discordance ERP -> action sur données structurées (règle).",
        )
    return RouteDecision(
        destination="transactional",
        reason="Type de trigger inconnu -> agent transactionnel par défaut (règle).",
    )


def supervisor_node(state: GraphState) -> dict:
    """Nœud LangGraph : produit la décision de routage."""
    llm = get_supervisor_llm()

    if llm is None:
        decision = _rule_based_route(state)
    else:
        context = (
            f"Trigger type: {state.get('trigger_type')}\n"
            f"Raw input: {state.get('raw_input')}"
        )
        try:
            structured = llm.with_structured_output(RouteDecision)
            decision = structured.invoke(
                [SystemMessage(_SYSTEM_PROMPT), HumanMessage(context)]
            )
        except Exception as e:  # pragma: no cover - repli si l'appel LLM échoue
            print(f"⚠️  Superviseur : échec LLM ({e}), bascule sur le routage par règles.")
            decision = _rule_based_route(state)

    print(f"🧭 Superviseur -> {decision.destination} ({decision.reason})")
    return {"route": decision.destination, "route_reason": decision.reason}

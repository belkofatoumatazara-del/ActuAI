"""
État Global (Global State) du système multi-agents ActuAI.

Ce dictionnaire partagé sert de mémoire de travail commune à l'ensemble du graphe
LangGraph. Il transporte la donnée brute du trigger, la décision de routage du
Superviseur, le BROUILLON (draft) produit par l'agent spécialiste, puis le statut
et le résultat de la validation humaine (Human-in-the-Loop).

Principe directeur (cf. rapport, couche AI Agent Layer) :
    trigger -> superviseur -> agent spécialiste (brouillon) -> pause HITL -> exécution
Aucun agent ne possède d'accès en écriture autonome au SAP : ils ne produisent
que des brouillons soumis à validation.
"""

from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

# --- Types métier ---------------------------------------------------------

# Les trois goulots d'étranglement opérationnels qui déclenchent le cycle.
TriggerType = Literal["supplier_email", "erp_discrepancy", "compliance_audit"]

# Les deux agents spécialistes vers lesquels le Superviseur peut router.
RouteType = Literal["transactional", "investigative"]

# Statut de la boucle de validation humaine.
HITLStatus = Literal["pending", "approved", "rejected"]


class GraphState(TypedDict, total=False):
    """Schéma de l'état partagé circulant dans le graphe LangGraph."""

    # --- Entrée (déposée par le routeur d'ingestion) ---
    trigger_type: TriggerType
    raw_input: dict[str, Any]  # email, paramètres de discordance ERP, requête d'audit

    # --- Mémoire de travail des agents spécialistes (boucle ReAct) ---
    # Le réducteur add_messages accumule les messages sans les écraser.
    messages: Annotated[list, add_messages]

    # --- Décision du Superviseur (routeur sémantique) ---
    route: Optional[RouteType]
    route_reason: Optional[str]

    # --- Brouillon produit par l'agent spécialiste (objet soumis au HITL) ---
    # Ex : payload de mise à jour SAP, FNC pré-remplie, dossier de traçabilité.
    draft_action: Optional[dict[str, Any]]

    # --- Validation humaine ---
    hitl_status: HITLStatus
    human_comment: Optional[str]

    # --- Résultat de l'exécution réelle (écriture SAP après validation) ---
    execution_result: Optional[dict[str, Any]]

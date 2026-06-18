"""Dépendances FastAPI partagées (injection du graphe compilé)."""

from actuai_backend.src.agents.graph import graph


def get_graph():
    """Fournit le graphe LangGraph compilé (singleton avec checkpointer).

    Le même checkpointer doit être réutilisé entre le déclenchement (pause HITL)
    et la reprise (validation), d'où l'usage d'une instance unique de module.
    """
    return graph

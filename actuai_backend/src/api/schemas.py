"""Schémas Pydantic des requêtes/réponses de l'API ActuAI."""

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Déclencheurs (Triggers) ----------------------------------------------

class EmailTrigger(BaseModel):
    """Email fournisseur reçu via webhook MS Exchange (cf. generators/emails.py)."""

    message_id: str
    sender: str
    subject: str
    date: Optional[str] = None
    body: str


class DiscrepancyTrigger(BaseModel):
    """Alerte interne de discordance entre date SAP et planning Airbus."""

    po_number: str
    detail: Optional[str] = Field(default=None, description="Description de l'écart constaté.")


class AuditTrigger(BaseModel):
    """Requête d'audit de conformité / reconstitution de traçabilité."""

    query: str
    po_number: Optional[str] = None


# --- Validation humaine (HITL) --------------------------------------------

class ValidationRequest(BaseModel):
    """Décision de l'expert humain sur un brouillon en attente."""

    approved: bool
    comment: Optional[str] = None


# --- Réponses --------------------------------------------------------------

class RunResponse(BaseModel):
    """État d'un cycle d'orchestration renvoyé au frontend."""

    thread_id: str
    status: str = Field(description="pending_validation | completed | rejected")
    route: Optional[str] = None
    draft: Optional[dict[str, Any]] = None
    execution_result: Optional[dict[str, Any]] = None

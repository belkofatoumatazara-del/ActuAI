"""
Outils SQL de l'agent Transactionnel (Missions 1 à 3).

Ces outils interrogent le datalake PostgreSQL local (tables miroir de SAP) pour
produire des BROUILLONS d'actions à haute précision logique. Aucun outil n'écrit
dans SAP : les outils `draft_*` ne font que préparer un payload soumis à la
validation humaine (HITL).

Note : l'engine PostgreSQL est importé PARESSEUSEMENT à l'intérieur des fonctions
pour que le module reste importable sans variable d'environnement de base de
données (utile pour les tests et le mode hors-ligne).
"""

import json
from datetime import date, datetime
from typing import Any

from langchain_core.tools import tool
from sqlmodel import Session, select


def _session() -> Session:
    """Ouvre une session sur le datalake (import paresseux de l'engine)."""
    from actuai_backend.src.database.connection import engine

    return Session(engine)


def _po_to_dict(po: Any) -> dict[str, Any]:
    return {
        "po_number": po.po_number,
        "part_reference": po.part_reference,
        "supplier_name": po.supplier_name,
        "quantity": po.quantity,
        "expected_delivery_date": str(po.expected_delivery_date),
        "status": po.status,
        "serial_number_expected": po.serial_number_expected,
    }


@tool
def get_purchase_order(po_number: str) -> dict:
    """Récupère une commande d'achat (Purchase Order) depuis le datalake SAP via son numéro.

    Renvoie les métadonnées : référence pièce, fournisseur, quantité, date de
    livraison prévue, statut et numéro de série attendu. À utiliser pour
    retrouver le contexte d'une commande avant toute action.
    """
    from actuai_backend.src.database.models import DatalakePurchaseOrder

    with _session() as s:
        po = s.exec(
            select(DatalakePurchaseOrder).where(
                DatalakePurchaseOrder.po_number == po_number
            )
        ).first()
        if not po:
            return {"found": False, "po_number": po_number}
        result = _po_to_dict(po)
        result["found"] = True
        return result


@tool
def check_delivery_risk(po_number: str) -> dict:
    """Évalue le risque AOG d'une commande (Mission 2 : coordination planning).

    Compare la date de livraison prévue de la commande à la date limite
    d'assemblage Airbus (« drop-dead date ») de la pièce concernée, puis calcule
    la marge en jours et un niveau de risque (OK / WARNING / CRITICAL). Sert à
    détecter proactivement les blocages de ligne d'assemblage.
    """
    from actuai_backend.src.database.models import (
        DatalakeProductionSchedule,
        DatalakePurchaseOrder,
    )

    with _session() as s:
        po = s.exec(
            select(DatalakePurchaseOrder).where(
                DatalakePurchaseOrder.po_number == po_number
            )
        ).first()
        if not po:
            return {"found": False, "po_number": po_number}

        schedule = s.exec(
            select(DatalakeProductionSchedule).where(
                DatalakeProductionSchedule.part_reference == po.part_reference
            )
        ).first()
        if not schedule:
            return {
                "found": True,
                "po_number": po_number,
                "part_reference": po.part_reference,
                "schedule_found": False,
                "note": "Aucun planning d'assemblage pour cette référence.",
            }

        margin = (schedule.assembly_line_date - po.expected_delivery_date).days
        if margin < 0:
            risk = "CRITICAL"  # livraison après la date d'assemblage -> AOG probable
        elif margin <= 2:
            risk = "WARNING"
        else:
            risk = "OK"

        return {
            "found": True,
            "po_number": po_number,
            "part_reference": po.part_reference,
            "expected_delivery_date": str(po.expected_delivery_date),
            "assembly_line_date": str(schedule.assembly_line_date),
            "margin_days": margin,
            "risk_level": risk,
        }


@tool
def draft_sap_date_update(po_number: str, new_delivery_date: str) -> str:
    """Prépare (SANS l'exécuter) un payload de mise à jour de date de livraison SAP (Mission 1).

    À appeler comme étape FINALE pour la synchronisation d'une commande après
    lecture d'un email de retard/expédition. La date doit être au format
    AAAA-MM-JJ. Renvoie un brouillon JSON qui sera soumis à validation humaine ;
    aucune écriture n'a lieu à ce stade.
    """
    try:
        datetime.strptime(new_delivery_date, "%Y-%m-%d")
    except ValueError:
        return json.dumps(
            {"error": "Format de date invalide, attendu AAAA-MM-JJ.",
             "received": new_delivery_date}
        )

    draft = {
        "action_type": "SAP_DATE_UPDATE",
        "agent": "transactional",
        "mission": "1 - Supply Chain Monitoring",
        "summary": f"Repousser la date de livraison de {po_number} au {new_delivery_date}.",
        "execution": {
            "method": "PUT",
            "path": f"/api/bapi/purchase-orders/{po_number}/update-date",
            "params": {"new_date": new_delivery_date},
        },
    }
    return json.dumps(draft, ensure_ascii=False)


@tool
def draft_fnc(po_number: str, defect_type: str) -> str:
    """Pré-remplit une Fiche de Non-Conformité (FNC) à partir des données SAP (Mission 3).

    À appeler comme étape FINALE pour documenter un défaut qualité. L'outil
    récupère les métadonnées de la commande (référence pièce, fournisseur) dans le
    datalake et construit un brouillon de FNC conforme EN9100/AS9100. Renvoie un
    brouillon JSON soumis à validation humaine ; aucune création n'a lieu à ce stade.
    """
    from actuai_backend.src.database.models import DatalakePurchaseOrder

    with _session() as s:
        po = s.exec(
            select(DatalakePurchaseOrder).where(
                DatalakePurchaseOrder.po_number == po_number
            )
        ).first()

    ncr_number = f"FNC-{date.today():%y}-{abs(hash(po_number)) % 900 + 100}"
    payload = {
        "ncr_number": ncr_number,
        "po_number": po_number,
        "defect_type": defect_type,
        "report_8d_status": "PENDING",
    }
    draft = {
        "action_type": "FNC_CREATION",
        "agent": "transactional",
        "mission": "3 - Quality & Non-Conformity Management",
        "summary": f"Créer la FNC {ncr_number} pour la commande {po_number} ({defect_type}).",
        "prefilled_from_sap": _po_to_dict(po) if po else {"found": False},
        "execution": {
            "method": "POST",
            "path": "/api/bapi/quality-notifications/",
            "json": payload,
        },
    }
    return json.dumps(draft, ensure_ascii=False)


# Liste exposée à l'agent (ordre = priorité pédagogique de découverte).
TRANSACTIONAL_TOOLS = [
    get_purchase_order,
    check_delivery_risk,
    draft_sap_date_update,
    draft_fnc,
]

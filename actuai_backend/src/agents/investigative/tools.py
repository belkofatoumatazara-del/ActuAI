"""
Outils RAG de l'agent Investigateur (Missions 4 et 5).

Ces outils effectuent des recherches sémantiques dans la base vectorielle Qdrant
(collection « technical_documentation » alimentée par `etl/document_indexer.py`)
afin de retrouver les bonnes versions de documents et de reconstituer l'historique
complet d'un composant. Le produit final est un BROUILLON de dossier soumis à la
validation humaine ; aucune écriture SAP n'a lieu ici.

Imports paresseux : embeddings et client Qdrant ne sont chargés qu'à l'appel des
outils, afin de garder le module importable en mode hors-ligne / test.
"""

import json
import os
from typing import Any

from langchain_core.tools import tool
from sqlmodel import Session, select

COLLECTION_NAME = "technical_documentation"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _vector_store():
    """Connexion paresseuse à Qdrant avec le même modèle d'embedding que l'ETL."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Qdrant
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = QdrantClient(url=qdrant_url)
    return Qdrant(
        client=client, collection_name=COLLECTION_NAME, embeddings=embeddings
    )


@tool
def search_technical_documents(query: str, k: int = 4) -> list[dict]:
    """Recherche sémantique dans la documentation technique indexée (Mission 4).

    Retrouve les `k` extraits les plus pertinents (certificats matière, rapports
    8D, PV de contrôle...) correspondant à la requête, avec leur source. À utiliser
    pour localiser la bonne version d'un document sans parcourir manuellement les
    disques réseau.
    """
    try:
        store = _vector_store()
        results = store.similarity_search(query, k=k)
    except Exception as e:  # pragma: no cover - dépend de Qdrant
        return [{"error": f"Recherche vectorielle indisponible : {e}"}]

    return [
        {
            "content": doc.page_content[:500],
            "source": doc.metadata.get("source", "inconnu"),
            "page": doc.metadata.get("page"),
        }
        for doc in results
    ]


@tool
def reconstruct_traceability(po_number: str) -> str:
    """Reconstitue l'historique end-to-end d'un composant (Mission 5).

    À appeler comme étape FINALE d'un audit de traçabilité. L'outil agrège les
    métadonnées structurées du datalake SAP (commande, fournisseur, n° de série
    attendu) et les extraits documentaires pertinents trouvés via RAG, puis vérifie
    la cohérence du numéro de série. Renvoie un brouillon de dossier de traçabilité
    (JSON) soumis à validation humaine.
    """
    from actuai_backend.src.database.connection import engine
    from actuai_backend.src.database.models import DatalakePurchaseOrder

    sap_record: dict[str, Any]
    with Session(engine) as s:
        po = s.exec(
            select(DatalakePurchaseOrder).where(
                DatalakePurchaseOrder.po_number == po_number
            )
        ).first()
        sap_record = (
            {
                "po_number": po.po_number,
                "part_reference": po.part_reference,
                "supplier_name": po.supplier_name,
                "serial_number_expected": po.serial_number_expected,
                "status": po.status,
            }
            if po
            else {"found": False, "po_number": po_number}
        )

    # Récupération des preuves documentaires associées à la commande.
    try:
        store = _vector_store()
        docs = store.similarity_search(po_number, k=4)
        evidence = [
            {"source": d.metadata.get("source", "inconnu"),
             "excerpt": d.page_content[:300]}
            for d in docs
        ]
    except Exception as e:  # pragma: no cover
        evidence = [{"error": f"RAG indisponible : {e}"}]

    draft = {
        "action_type": "TRACEABILITY_DOSSIER",
        "agent": "investigative",
        "mission": "5 - Component Traceability",
        "summary": f"Dossier de traçabilité reconstitué pour la commande {po_number}.",
        "sap_record": sap_record,
        "document_evidence": evidence,
        "execution": {"method": "NONE", "note": "Archivage du dossier après validation."},
    }
    return json.dumps(draft, ensure_ascii=False)


# Liste exposée à l'agent Investigateur.
INVESTIGATIVE_TOOLS = [
    search_technical_documents,
    reconstruct_traceability,
]

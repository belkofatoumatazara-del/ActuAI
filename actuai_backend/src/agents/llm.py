"""
Fabrique de modèles de langage (LLM Factory) — Stratégie hybride ActuAI.

Conformément à la couche "Foundation Model" du rapport, chaque agent reçoit un
modèle adapté à son rôle, en équilibrant performance et contraintes matérielles
locales (serveur edge limité à 8 Go de VRAM) :

    - Superviseur   -> Llama 3.1 8B en local via Ollama (latence quasi nulle,
                       appelé à chaque routage, aucune fuite de donnée).
    - Transactionnel-> Mistral-Nemo (12B) via API cloud sécurisée (excellent
                       suivi d'instructions de tool-calling / génération de JSON).
    - Investigateur -> Llama 3.1 70B / Mistral Large via API cloud (grande fenêtre
                       de contexte pour les gros chunks RAG).

Les imports des fournisseurs sont PARESSEUX (lazy) : le graphe reste importable
et testable même si `langchain-ollama` / `langchain-openai` ne sont pas installés
ou si aucun modèle n'est disponible.

Mode hors-ligne : exporter `ACTUAI_LLM_MODE=offline` force toutes les fabriques à
renvoyer `None`. Les nœuds basculent alors sur des stratégies déterministes
(routage par règles, brouillons construits par simple parsing), ce qui permet de
faire tourner le graphe complet sans aucun modèle ni clé d'API.
"""

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel


def _offline() -> bool:
    return os.getenv("ACTUAI_LLM_MODE", "").lower() == "offline"


def _build_ollama(model: str, temperature: float = 0.0) -> Optional[BaseChatModel]:
    """Instancie un modèle local Ollama (On-Premise)."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        print("⚠️  langchain-ollama non installé : superviseur local indisponible.")
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)
    except Exception as e:  # pragma: no cover - dépend de l'environnement
        print(f"⚠️  Impossible d'initialiser Ollama ({model}) : {e}")
        return None


def _build_cloud(model: str, temperature: float = 0.0) -> Optional[BaseChatModel]:
    """
    Instancie un modèle cloud via une API compatible OpenAI
    (NVIDIA NIM ou Azure), avec politique de zéro rétention de données.
    """
    api_key = os.getenv("ACTUAI_CLOUD_API_KEY")
    base_url = os.getenv("ACTUAI_CLOUD_BASE_URL", "https://integrate.api.nvidia.com/v1")
    if not api_key:
        print(f"⚠️  ACTUAI_CLOUD_API_KEY absente : modèle cloud '{model}' indisponible.")
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("⚠️  langchain-openai non installé : agents cloud indisponibles.")
        return None
    try:
        return ChatOpenAI(
            model=model, base_url=base_url, api_key=api_key, temperature=temperature
        )
    except Exception as e:  # pragma: no cover
        print(f"⚠️  Impossible d'initialiser le modèle cloud ({model}) : {e}")
        return None


def get_supervisor_llm() -> Optional[BaseChatModel]:
    """Modèle léger local pour le routage sémantique (Llama 3.1 8B / Ollama)."""
    if _offline():
        return None
    model = os.getenv("ACTUAI_SUPERVISOR_MODEL", "llama3.1:8b")
    return _build_ollama(model, temperature=0.0)


def get_transactional_llm() -> Optional[BaseChatModel]:
    """Modèle cloud orienté tool-calling/JSON pour l'agent Transactionnel."""
    if _offline():
        return None
    model = os.getenv("ACTUAI_TRANSACTIONAL_MODEL", "mistralai/mistral-nemo-12b-instruct")
    return _build_cloud(model, temperature=0.0)


def get_investigative_llm() -> Optional[BaseChatModel]:
    """Modèle cloud à grand contexte pour l'agent Investigateur (RAG)."""
    if _offline():
        return None
    model = os.getenv("ACTUAI_INVESTIGATIVE_MODEL", "meta/llama-3.1-70b-instruct")
    return _build_cloud(model, temperature=0.1)

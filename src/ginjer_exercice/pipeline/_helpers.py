"""Fonctions utilitaires partagées entre les steps du pipeline.

Ces helpers sont des fonctions pures sans état, testables indépendamment.
Ils centralisent la logique de formatage des inputs LLM pour éviter
la duplication entre les 4 steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..llm.base import LLMMessage
from ..schemas.ad import AdText

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Limite stricte : au-delà, on échantillonne pour éviter les dépassements de contexte.
# Gemini 2.0 Flash supporte ~1M tokens, mais les médias comptent pour beaucoup.
MAX_MEDIA_FILES = 8
MAX_IMAGES = 8
MAX_VIDEOS = 3  # Les vidéos consomment beaucoup plus de tokens que les images


def build_texts_block(texts: list[AdText]) -> str:
    """Concatène tous les textes disponibles d'une pub en un bloc lisible.

    Chaque champ non-vide est préfixé de son label pour aider le LLM
    à différencier titre, corps, légende et URL.

    Args:
        texts: Liste des objets AdText de la publicité.

    Returns:
        Bloc texte multi-lignes prêt à être injecté dans un prompt.
        Retourne ``"(no text available)"`` si tous les champs sont vides.
    """
    lines: list[str] = []

    for i, text in enumerate(texts, start=1):
        prefix = f"[Text {i}] " if len(texts) > 1 else ""
        if text.title:
            lines.append(f"{prefix}Title: {text.title}")
        if text.body_text:
            lines.append(f"{prefix}Body: {text.body_text}")
        if text.caption:
            lines.append(f"{prefix}Caption: {text.caption}")
        if text.url:
            lines.append(f"{prefix}URL: {text.url}")

    if not lines:
        return "(no text available)"

    return "\n".join(lines)


def build_media_messages(media_urls: list[str]) -> list[LLMMessage]:
    """Construit les messages LLM multimodaux à partir des URLs de médias.

    Stratégie de gestion des limites :
        - Images : max ``MAX_IMAGES`` fichiers. Au-delà, on prend les N premiers
          (pas d'échantillonnage aléatoire pour la reproductibilité).
        - Vidéos : max ``MAX_VIDEOS`` fichiers pour les mêmes raisons de coût token.
        - Mix images/vidéos : la limite globale est ``MAX_MEDIA_FILES``.
        - Si aucun média : retourne une liste vide (la step continue avec le texte seul).

    L'URL est passée directement dans ``LLMMessage.media`` — le provider
    (GeminiProvider) est responsable de la conversion gs:// si nécessaire.

    Args:
        media_urls: Liste d'URLs de médias (images ou vidéos).

    Returns:
        Liste de ``LLMMessage`` avec les médias chargés.
        Peut être une liste vide si ``media_urls`` est vide.

    Raises:
        ValueError: Si une URL est vide ou None (corruption de données).
    """
    if not media_urls:
        return []

    images = [u for u in media_urls if not _is_video(u)]
    videos = [u for u in media_urls if _is_video(u)]

    # Appliquer les limites
    if len(images) > MAX_IMAGES:
        logger.warning(
            "Pub avec %d images — limite à %d (les %d premières conservées).",
            len(images), MAX_IMAGES, MAX_IMAGES,
        )
        images = images[:MAX_IMAGES]

    if len(videos) > MAX_VIDEOS:
        logger.warning(
            "Pub avec %d vidéos — limite à %d (les %d premières conservées).",
            len(videos), MAX_VIDEOS, MAX_VIDEOS,
        )
        videos = videos[:MAX_VIDEOS]

    selected = images + videos
    if len(selected) > MAX_MEDIA_FILES:
        logger.warning(
            "Total médias sélectionnés (%d) dépasse la limite globale (%d) — troncature.",
            len(selected), MAX_MEDIA_FILES,
        )
        selected = selected[:MAX_MEDIA_FILES]

    if not selected:
        return []

    # Un seul message multimodal avec tous les médias
    return [LLMMessage(text="", media=selected)]


def _is_video(url: str) -> bool:
    """Détecte si une URL pointe vers une vidéo (heuristique par extension)."""
    return any(url.lower().endswith(ext) for ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"))


def build_llm_messages(
    prompt_text: str,
    media_urls: list[str],
) -> list[LLMMessage]:
    """Assemble les messages LLM finaux : prompt texte + médias.

    Le prompt est dans le premier message. Si des médias sont présents,
    ils sont ajoutés comme message supplémentaire (ou comme parts du même message).

    Args:
        prompt_text: Le prompt compilé (avec variables substituées).
        media_urls: URLs des médias à attacher.

    Returns:
        Liste ordonnée de ``LLMMessage`` prête à passer à ``generate_structured()``.
    """
    text_message = LLMMessage(text=prompt_text, media=[])
    media_messages = build_media_messages(media_urls)

    if not media_messages:
        return [text_message]

    # Fusionner le texte et les médias dans un seul message pour Gemini
    combined = LLMMessage(
        text=prompt_text,
        media=media_messages[0].media if media_messages else [],
    )
    return [combined]

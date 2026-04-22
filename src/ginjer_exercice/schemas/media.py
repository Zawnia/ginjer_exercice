"""Schémas Pydantic pour les contenus médias (images et vidéos).

Utilisé comme modèle de transfert entre le MediaFetcher et les étapes
multimodales du pipeline. Le champ ``content`` contient les octets bruts
téléchargés — il est exclu de ``repr`` pour éviter la pollution des logs.
"""

from enum import Enum

from pydantic import BaseModel, Field


class MediaKind(str, Enum):
    """Type de média supporté par le pipeline."""
    IMAGE = "image"
    VIDEO = "video"


class MediaContent(BaseModel):
    """Contenu d'un asset média téléchargé et validé.

    Attributes:
        url: URL d'origine du média.
        kind: Type de média (image ou vidéo).
        mime_type: Type MIME détecté depuis la réponse HTTP.
        content: Octets bruts du fichier téléchargé.
        size_bytes: Taille du contenu en octets.
    """
    url: str = Field(..., description="URL d'origine du média")
    kind: MediaKind = Field(..., description="Type de média (image ou vidéo)")
    mime_type: str = Field(..., description="Type MIME détecté depuis la réponse HTTP")
    content: bytes = Field(repr=False, description="Octets bruts du fichier téléchargé")
    size_bytes: int = Field(..., ge=0, description="Taille du contenu en octets")

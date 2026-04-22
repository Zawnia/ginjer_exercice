from pydantic import BaseModel, Field
from enum import Enum

class Brand(str, Enum):
    CHANEL = "CHANEL"
    DIOR = "DIOR"
    LOUIS_VUITTON = "LOUIS_VUITTON"
    BALENCIAGA = "BALENCIAGA"
    MFK = "MFK"

class AdText(BaseModel):
    title: str | None = Field(
        default=None, 
        description="Le titre principal de la publicité, s'il existe."
    )
    body_text: str | None = Field(
        default=None, 
        description="Le texte principal (corps) de la publicité."
    )
    caption: str | None = Field(
        default=None, 
        description="La légende associée au média de la pub."
    )
    url: str | None = Field(
        default=None, 
        description="L'URL de redirection ou le lien cliquable de la pub."
    )

class Ad(BaseModel):
    platform_ad_id: str
    brand: Brand
    texts: list[AdText] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)

    def all_text(self) -> str:
        """Concatène tous les textes disponibles de la publicité."""
        parts = []
        for t in self.texts:
            if t.title: parts.append(t.title)
            if t.body_text: parts.append(t.body_text)
            if t.caption: parts.append(t.caption)
            if t.url: parts.append(t.url)
        return "\n".join(parts)
from pydantic import BaseModel, Field, ConfigDict

class ScoreReport(BaseModel):
    """
    Rapport de scores pour une publicité traitée.
    Contient les scores calculés la cohérence taxonomique / la confiance globale / le score du LLM as a judge.
    """
    taxonomy_coherence: float | None = Field(
        default=None, 
        description="Score déterministe de cohérence de l'arbre taxonomique"
    )
    confidence: float | None = Field(
        default=None, 
        description="Confiance moyenne remontée par les étapes du pipeline"
    )
    llm_judge: float | None = Field(
        default=None, 
        description="Score attribué par le LLM évaluateur final"
    )

    model_config = ConfigDict(extra="allow")

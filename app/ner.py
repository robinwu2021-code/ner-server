from gliner import GLiNER
from app.models import Entity


class NERService:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model = GLiNER.from_pretrained(model_name, cache_dir=cache_dir)

    def extract(self, text: str, labels: list[str], threshold: float) -> list[Entity]:
        if not text or not labels:
            return []
        raw = self._model.predict_entities(text, labels, threshold=threshold)
        return [
            Entity(
                text=e["text"],
                label=e["label"],
                score=round(e["score"], 4),
                start=e["start"],
                end=e["end"],
            )
            for e in raw
        ]

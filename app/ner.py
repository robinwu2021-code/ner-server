import httpx
from fastapi import HTTPException

from app.models import Entity


class NERService:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def extract(self, text: str, labels: list[str], threshold: float) -> list[Entity]:
        if not text or not labels:
            return []
        try:
            resp = httpx.post(
                f"{self._base_url}/extract",
                json={"text": text, "labels": labels, "threshold": threshold},
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Upstream NER API error: {e}")
        return [Entity(**e) for e in resp.json()["entities"]]

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import NER_API_BASE_URL
from app.models import ExtractRequest, ExtractResponse
from app.ner import NERService

ner_service: NERService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_service
    ner_service = NERService(NER_API_BASE_URL)
    yield
    ner_service = None


app = FastAPI(title="NER API", lifespan=lifespan)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    entities = ner_service.extract(req.text, req.labels, req.threshold)
    return ExtractResponse(entities=entities)

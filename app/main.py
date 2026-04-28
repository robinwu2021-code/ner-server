from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import MODEL_CACHE_DIR, MODEL_NAME
from app.models import ExtractRequest, ExtractResponse
from app.ner import NERService

ner_service: NERService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_service
    ner_service = NERService(MODEL_NAME, MODEL_CACHE_DIR)
    yield
    ner_service = None


app = FastAPI(title="NER API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    entities = ner_service.extract(req.text, req.labels, req.threshold)
    return ExtractResponse(entities=entities)

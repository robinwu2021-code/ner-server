import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import MODEL_CACHE_DIR, MODEL_NAME
from app.logger import get_logger
from app.models import ExtractRequest, ExtractResponse
from app.ner import NERService

logger = get_logger("ner.api")
ner_service: NERService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_service
    logger.info("Loading model: %s (cache_dir=%s)", MODEL_NAME, MODEL_CACHE_DIR)
    ner_service = NERService(MODEL_NAME, MODEL_CACHE_DIR)
    logger.info("Model ready")
    yield
    ner_service = None


app = FastAPI(title="NER API", lifespan=lifespan)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    logger.info(
        "extract request | text_len=%d labels=%s threshold=%s",
        len(req.text),
        req.labels,
        req.threshold,
    )
    t0 = time.perf_counter()
    entities = ner_service.extract(req.text, req.labels, req.threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "extract response | entities=%s elapsed=%.1fms",
        [{"text": e.text, "label": e.label, "score": e.score} for e in entities],
        elapsed_ms,
    )
    return ExtractResponse(entities=entities)

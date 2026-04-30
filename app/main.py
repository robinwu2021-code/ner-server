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


app = FastAPI(
    title="NER API",
    description=(
        "Zero-shot Named Entity Recognition powered by GLiNER. "
        "Supports English, Chinese, Arabic and mixed-language text. "
        "Labels are optional — omit them to use built-in bilingual defaults."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["NER"])
def extract(req: ExtractRequest):
    logger.info(
        "extract request | text_len=%d labels=%s threshold=%s language=%s",
        len(req.text),
        req.labels or "(default)",
        req.threshold,
        req.language,
    )
    t0 = time.perf_counter()
    entities, labels_used = ner_service.extract(
        req.text,
        req.labels,
        req.threshold,
        language=req.language,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "extract response | entities=%d elapsed=%.1fms labels_used=%d",
        len(entities),
        elapsed_ms,
        len(labels_used),
    )
    return ExtractResponse(entities=entities, labels_used=labels_used)

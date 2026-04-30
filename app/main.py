import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import EN_MODEL_NAME, MODEL_CACHE_DIR, ZH_MODEL_NAME
from app.logger import get_logger
from app.models import ExtractRequest, ExtractResponse
from app.ner import NERService

logger = get_logger("ner.api")
ner_service: NERService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_service
    logger.info(
        "Initializing NER service | en_model=%s zh_model=%s cache=%s",
        EN_MODEL_NAME, ZH_MODEL_NAME, MODEL_CACHE_DIR,
    )
    ner_service = NERService(EN_MODEL_NAME, ZH_MODEL_NAME, MODEL_CACHE_DIR)
    # 预热：启动时同时加载两个模型，首个请求无需等待
    ner_service.warmup()
    logger.info("NER service ready")
    yield
    ner_service = None


app = FastAPI(
    title="NER API",
    description=(
        "Zero-shot Named Entity Recognition powered by GLiNER (EN/AR) "
        "and BERT-Chinese (ZH). "
        "Supports English · Chinese · Arabic · mixed-language text. "
        "Labels are optional — omit them to use built-in bilingual defaults."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["NER"])
def extract(req: ExtractRequest):
    logger.info(
        "extract request | text_len=%d labels=%s threshold=%s language=%s min_entities=%s",
        len(req.text),
        req.labels or "(default)",
        req.threshold,
        req.language,
        req.min_entities,
    )
    t0 = time.perf_counter()
    entities, labels_used = ner_service.extract(
        req.text,
        req.labels,
        req.threshold,
        language=req.language,
        min_entities=req.min_entities,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "extract response | entities=%d elapsed=%.1fms language=%s",
        len(entities),
        elapsed_ms,
        req.language,
    )
    return ExtractResponse(entities=entities, labels_used=labels_used)

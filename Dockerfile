FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 构建时预下载两个模型，冷启动无需联网 ──────────────────────────────────────
# EN / AR 模型：GLiNER 零样本多语言（~1 GB）
RUN python -c "\
from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_multi-v2.1', cache_dir='/app/model_cache')"

# ZH 模型：BERT 专用中文 NER（~400 MB）
RUN python -c "\
from transformers import pipeline; \
pipeline('token-classification', \
         model='shibing624/bert4ner-base-chinese', \
         model_kwargs={'cache_dir': '/app/model_cache'}, \
         aggregation_strategy='simple')"

COPY app/ app/
COPY run.py .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV MODEL_CACHE_DIR=/app/model_cache
ENV EN_MODEL_NAME=urchade/gliner_multi-v2.1
ENV ZH_MODEL_NAME=shibing624/bert4ner-base-chinese

EXPOSE 7860

CMD ["python", "run.py"]

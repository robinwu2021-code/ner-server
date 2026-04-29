FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time so cold-start needs no network access.
# The image will be ~1.5 GB but startup is instant.
RUN python -c "\
from gliner import GLiNER; \
GLiNER.from_pretrained('knowledgator/gliner-multitask-large-v0.5', cache_dir='/app/model_cache')"

COPY app/ app/
COPY run.py .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV MODEL_CACHE_DIR=/app/model_cache

EXPOSE 7860

CMD ["python", "run.py"]

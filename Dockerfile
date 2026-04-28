FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run.py .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV MODEL_CACHE_DIR=/app/model_cache

EXPOSE 7860

CMD ["python", "run.py"]

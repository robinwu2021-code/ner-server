FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run.py .

ENV HOST=0.0.0.0
ENV PORT=4000
ENV NER_API_BASE_URL=http://127.0.0.1:4001

EXPOSE 7860

CMD ["python", "run.py"]

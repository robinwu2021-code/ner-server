---
title: NER Server
emoji: 🏷️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NER Server

Zero-shot Named Entity Recognition HTTP API powered by [GLiNER](https://github.com/urchade/GLiNER).

## API

Base URL: `https://<your-hf-username>-ner-server.hf.space`

### `GET /api/v1/health`

```json
{"status": "ok"}
```

### `POST /api/v1/extract`

**Request**

```json
{
  "text": "Elon Musk founded SpaceX in Hawthorne, California.",
  "labels": ["person", "organization", "location"],
  "threshold": 0.6
}
```

**Response**

```json
{
  "entities": [
    {"text": "Elon Musk",   "label": "person",       "score": 0.98, "start": 0,  "end": 9 },
    {"text": "SpaceX",      "label": "organization", "score": 0.97, "start": 18, "end": 24},
    {"text": "Hawthorne",   "label": "location",     "score": 0.91, "start": 28, "end": 37},
    {"text": "California",  "label": "location",     "score": 0.95, "start": 39, "end": 49}
  ]
}
```

## Environment Variables

| Variable          | Default                       | Description                        |
|-------------------|-------------------------------|------------------------------------|
| `MODEL_NAME`      | `urchade/gliner_medium-v2.1`  | GLiNER model name                  |
| `PORT`            | `7860`                        | Listen port (fixed by HF Spaces)   |
| `MODEL_CACHE_DIR` | `/app/model_cache`            | Model cache path                   |
| `HF_ENDPOINT`     | *(huggingface.co)*            | Override with a custom mirror URL  |

## Interactive Docs

- Swagger UI: `/docs`
- ReDoc: `/redoc`

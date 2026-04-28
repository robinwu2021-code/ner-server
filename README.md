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

### `POST /extract`

```json
{
  "text": "Elon Musk founded SpaceX in 2002.",
  "labels": ["person", "organization"],
  "threshold": 0.5
}
```

**Response**

```json
{
  "entities": [
    {"text": "Elon Musk", "label": "person",       "score": 0.98, "start": 0,  "end": 9},
    {"text": "SpaceX",    "label": "organization", "score": 0.97, "start": 18, "end": 24}
  ]
}
```

### `GET /health`

```json
{"status": "ok"}
```

## Environment Variables

| Variable        | Default                        | Description          |
|-----------------|-------------------------------|----------------------|
| `MODEL_NAME`    | `urchade/gliner_medium-v2.1`  | GLiNER model name    |
| `PORT`          | `7860`                        | Listen port          |
| `MODEL_CACHE_DIR` | `./model_cache`             | Local model cache    |
| `HF_ENDPOINT`   | *(huggingface.co)*            | Custom HF mirror URL |

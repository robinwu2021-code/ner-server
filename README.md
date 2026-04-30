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
Supports **English · Chinese · Arabic · mixed-language** text.

## Quick Start

```bash
# 最简调用：不传 labels，自动使用内置双语标签集
curl -X POST https://robinwu-nerserver.hf.space/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "马云创立了阿里巴巴，总部在杭州。"}'

# 自定义标签（自动补充双语对等标签提升召回）
curl -X POST https://robinwu-nerserver.hf.space/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Elon Musk founded SpaceX.", "labels": ["full name of a person", "company or organization name"]}'
```

## API

Base URL: `https://robinwu-nerserver.hf.space`

### `GET /api/v1/health`

```json
{"status": "ok"}
```

### `POST /api/v1/extract`

**Request**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✓ | — | Input text |
| `labels` | string[] | — | `[]` | Entity type labels. **Empty = use built-in bilingual defaults.** Bilingual pairs are auto-expanded. |
| `threshold` | float [0,1] | — | `0.4` | Minimum confidence. Lower → more entities; higher → more precise. |
| `language` | `auto`/`en`/`zh`/`ar`/`mixed` | — | `auto` | Language hint (auto-detected when omitted). |

**Example — labels omitted (auto bilingual defaults)**

```json
{"text": "张伟加入了 Google 北京研发中心，负责 Android 系统优化。"}
```

```json
{
  "entities": [
    {"text": "张伟",    "label": "人名或姓名",                "score": 0.91, "start": 0,  "end": 2},
    {"text": "Google",  "label": "company or organization name", "score": 0.95, "start": 6,  "end": 12},
    {"text": "Android", "label": "product or technology name",   "score": 0.88, "start": 20, "end": 27}
  ],
  "labels_used": ["full name of a person", "人名或姓名", "company or organization name", "..."]
}
```

**Example — custom labels (bilingual auto-expansion)**

```json
{
  "text": "Elon Musk founded SpaceX in Hawthorne, California.",
  "labels": ["full name of a person", "company or organization name", "geographical location"],
  "threshold": 0.5
}
```

```json
{
  "entities": [
    {"text": "Elon Musk",   "label": "full name of a person",        "score": 0.98, "start": 0,  "end": 9 },
    {"text": "SpaceX",      "label": "company or organization name", "score": 0.97, "start": 18, "end": 24},
    {"text": "Hawthorne",   "label": "geographical location",        "score": 0.91, "start": 28, "end": 37},
    {"text": "California",  "label": "geographical location",        "score": 0.95, "start": 39, "end": 49}
  ],
  "labels_used": ["full name of a person", "人名或姓名", "company or organization name", "..."]
}
```

## Built-in Bilingual Label Pairs

When `labels` is empty, the following label pairs are used automatically:

| English | 中文 |
|---|---|
| full name of a person | 人名或姓名 |
| company or organization name | 公司或组织机构名称 |
| geographical location | 地名或城市 |
| product or technology name | 产品或技术名称 |
| date or year | 日期或年份 |
| hospital or medical institution | 医院或医疗机构名称 |
| university or research institution | 大学或研究机构 |
| project or initiative name | 项目或计划名称 |
| legislation or policy name | 法规或政策名称 |
| monetary amount | 金额或货币 |
| job title or position | 职位或头衔 |
| event name | 事件或活动名称 |

When you provide custom labels, each label is automatically paired with its counterpart
(e.g. passing `"人名或姓名"` also activates `"full name of a person"` internally).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `knowledgator/gliner-multitask-large-v0.5` | GLiNER model |
| `PORT` | `7860` | Listen port (fixed by HF Spaces) |
| `MODEL_CACHE_DIR` | `/app/model_cache` | Model cache path |
| `HF_ENDPOINT` | *(huggingface.co)* | Override with a mirror URL |

## Interactive Docs

- Swagger UI: `/docs`
- ReDoc: `/redoc`

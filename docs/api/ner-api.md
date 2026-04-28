# NER API 接口文档

Base URL: `http://localhost:4000`

---

## GET /health

健康检查。

**响应**

```json
{"status": "ok"}
```

---

## POST /extract

从文本中抽取命名实体。

**请求体**

| 字段        | 类型           | 必填 | 默认值 | 说明                         |
|-------------|----------------|------|--------|------------------------------|
| `text`      | string         | 是   | —      | 待抽取的文本                 |
| `labels`    | array[string]  | 是   | —      | 期望识别的实体类型列表       |
| `threshold` | float [0, 1]   | 否   | `0.5`  | 置信度阈值，低于此值的结果过滤掉 |

**请求示例**

```json
{
  "text": "Elon Musk founded SpaceX in Hawthorne, California.",
  "labels": ["person", "organization", "location"],
  "threshold": 0.5
}
```

**响应体**

| 字段               | 类型          | 说明                         |
|--------------------|---------------|------------------------------|
| `entities`         | array[Entity] | 抽取到的实体列表             |
| `entities[].text`  | string        | 实体原文                     |
| `entities[].label` | string        | 实体类型（与请求 labels 对应）|
| `entities[].score` | float         | 置信度分数，范围 [0, 1]      |
| `entities[].start` | int           | 实体在原文中的起始字符偏移   |
| `entities[].end`   | int           | 实体在原文中的结束字符偏移   |

**响应示例**

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

**错误码**

| HTTP 状态码 | 说明                                      |
|-------------|-------------------------------------------|
| `200`       | 成功，`entities` 为空列表表示未抽取到结果 |
| `422`       | 请求参数校验失败（如 threshold 超出范围） |

---

## 说明

- 模型支持 **zero-shot** 识别，`labels` 可传入任意自定义类型，无需预定义。
- 服务启动时加载一次模型，所有请求复用同一实例。
- 在线文档（服务运行时可访问）：
  - Swagger UI：`http://localhost:4000/docs`
  - ReDoc：`http://localhost:4000/redoc`

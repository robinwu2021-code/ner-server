# TDD-ner-api

状态：已实现
关联需求：docs/requirements/PRD-ner-api.md
创建日期：2026-04-28

## 1. 需求摘要

用 FastAPI 包装 GLiNER 模型，提供 POST /extract 接口，接收文本与实体类型列表，返回抽取结果。

## 2. 方案设计

### 方案选型

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| FastAPI + GLiNER | 轻量、async 支持好、自动生成文档 | — | ✅ 采用 |
| Flask + GLiNER | 更简单 | 无 async，性能差 | ❌ |

### 目录结构

```
ner-server/
├── app/
│   ├── main.py          # FastAPI app 入口，lifespan 加载模型
│   ├── config.py        # 环境变量配置
│   ├── models.py        # Pydantic 请求/响应模型
│   └── ner.py           # GLiNER 封装（NERService）
├── tests/
│   └── test_extract.py
├── requirements.txt
└── .env.example
```

### 核心接口

```
POST /extract
Request:  { "text": str, "labels": list[str], "threshold": float = 0.5 }
Response: { "entities": [{ "text": str, "label": str, "score": float, "start": int, "end": int }] }

GET /health
Response: { "status": "ok" }
```

### 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | urchade/gliner_medium-v2.1 | GLiNER 模型名称 |
| PORT | 8000 | 服务端口 |
| HOST | 0.0.0.0 | 监听地址 |

## 3. 测试策略

- 正常路径：传入文本和标签，返回实体列表
- 空文本：返回空实体列表
- 空标签列表：返回空实体列表
- threshold 过滤：高阈值时过滤低置信度实体

---
确认记录：2026-04-28 用户确认（口头需求）

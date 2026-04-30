import os

# ── 模型配置 ──────────────────────────────────────────────────────────────────
# 英文 / 阿拉伯文 / 混合：轻量 GLiNER 零样本模型
EN_MODEL_NAME: str = os.getenv("EN_MODEL_NAME", "urchade/gliner_multi-v2.1")

# 中文：专用 BERT NER 模型（400MB，~100ms/次，4 种固定实体类型）
ZH_MODEL_NAME: str = os.getenv("ZH_MODEL_NAME", "shibing624/bert4ner-base-chinese")

# 兼容旧版环境变量（若设置 MODEL_NAME，则覆盖 EN_MODEL_NAME）
_legacy = os.getenv("MODEL_NAME")
if _legacy:
    EN_MODEL_NAME = _legacy

MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./model_cache")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "4000"))

# 可选：国内镜像（留空则使用 huggingface.co）
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

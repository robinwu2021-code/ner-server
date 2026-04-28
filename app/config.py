import os

MODEL_NAME: str = os.getenv("MODEL_NAME", "urchade/gliner_medium-v2.1")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "4000"))
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# Only override HF_ENDPOINT when explicitly set (local mirror) so HF Spaces
# can reach huggingface.co directly without forcing the mirror.
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

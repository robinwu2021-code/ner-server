import os

NER_API_BASE_URL: str = os.getenv("NER_API_BASE_URL", "http://127.0.0.1:4000")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "4000"))

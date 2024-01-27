import os

api_models = os.environ.get("API_MODELS", "api/models")
bind = os.environ.get("BIND_IP", "127.0.0.1:8000")

raw_env = [
    "PROXY=true",
    f"API_MODELS={api_models}",
]
timeout = 120

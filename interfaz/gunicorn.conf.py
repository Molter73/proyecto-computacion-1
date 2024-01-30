import os

bind = os.environ.get("BIND_IP", "127.0.0.1:8050")
raw_env = [
    "API_URL=http://api:8000",
    "UI_PREFIX=/classifiers/",
]

import multiprocessing

bind = "127.0.0.1:8000"
raw_env = [
    "PROXY=true",
    "API_MODELS=api/models",
]
timeout = 120
workers = multiprocessing.cpu_count() * 2 + 1

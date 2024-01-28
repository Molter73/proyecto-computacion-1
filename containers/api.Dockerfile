FROM python:3.11-slim AS base

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

FROM base

RUN mkdir /app
COPY models /app/models
COPY app.py /app
COPY gunicorn.conf.py /app
COPY __init__.py /app

ENV API_MODELS=/app/models
ENV BIND_IP="0.0.0.0:8000"
EXPOSE 8000

ENTRYPOINT ["gunicorn"]
CMD ["-c", "/app/gunicorn.conf.py", "app:create_app()"]


FROM python:3.11-slim AS base

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

FROM base

RUN mkdir /app
COPY assets/ /app/assets/
COPY interfaz.py /app
COPY gunicorn.conf.py /app

ENV BIND_IP="0.0.0.0:8000"
ENV APP_BASENAME="/classifiers/"
EXPOSE 8000

ENTRYPOINT ["gunicorn"]
CMD ["-c", "/app/gunicorn.conf.py", "app.interfaz:server"]

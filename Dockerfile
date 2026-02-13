FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

RUN pip install --no-cache-dir soprano-tts uvicorn gradio[all] gunicorn

FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
EXPOSE 8000 7860

CMD ["uvicorn", "soprano.server:app", "--host", "0.0.0.0", "--port", "8000"]

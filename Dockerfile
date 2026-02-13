FROM python:3.11-slim

RUN pip install --no-cache-dir soprano-tts uvicorn gradio[all] gunicorn

WORKDIR /app
EXPOSE 8000 7860

CMD ["uvicorn", "soprano.server:app", "--host", "0.0.0.0", "--port", "8000"]

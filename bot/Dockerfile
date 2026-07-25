# PolyglotBot image. Build from the REPO ROOT (the bot imports langbench):
#   docker build -f bot/Dockerfile -t polyglotbot .
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
COPY config ./config
COPY bot ./bot

RUN uv pip install --system --no-cache ".[bot]"

ENV PYTHONPATH=/app/bot
ENV PYTHONUNBUFFERED=1

# /metrics
EXPOSE 9100

CMD ["python", "-m", "polyglotbot.main"]

# Langbench harness image (the bot has its own: bot/Dockerfile).
#   docker build -t langbench .
#   docker run --env-file .env -v $PWD/data:/app/data langbench \
#       python scripts/run_eval.py --dry-run
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts

RUN uv pip install --system --no-cache .

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/run_eval.py", "--dry-run"]

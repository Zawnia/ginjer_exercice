FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY prompts /app/prompts
COPY data /app/data
COPY product_categorisation.json /app/product_categorisation.json

RUN uv sync --frozen

CMD ["python", "-m", "ginjer_exercice.cli", "--help"]

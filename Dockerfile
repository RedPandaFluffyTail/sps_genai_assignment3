FROM python:3.12-slim-bookworm

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock /code/

RUN uv sync --frozen

COPY main.py /code/
COPY app /code/app
COPY helper_lib /code/helper_lib

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
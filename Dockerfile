# Projector demo — container image for any Docker host (Render, Cloud Run, a VPS…).
#
# Serves the FastAPI backend + web front on the synthetic `demo` source. Only the
# `api` extra is installed (FastAPI + uvicorn + numpy) — no desktop/Qt deps, so the
# image stays small. Binds 0.0.0.0 on $PORT (hosts inject their own; default 7860).
#
#   docker build -t projector .
#   docker run --rm -p 7860:7860 projector   # -> http://localhost:7860
FROM python:3.12-slim

# uv (pinned) for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

# Hugging Face Spaces runs containers as uid 1000; create a matching non-root user
# so the venv, caches and session sidecars live in a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Resolve and install dependencies first (cached unless pyproject/uv.lock change),
# without the project itself. README is referenced by pyproject metadata.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --extra api --frozen --no-dev --no-install-project

# Then the source, and install the project (web assets are packaged under projector/web).
COPY --chown=user projector ./projector
RUN uv sync --extra api --frozen --no-dev

# $PORT is injected by the host (Render, Cloud Run, …); 7860 is the local default.
ENV PORT=7860
EXPOSE 7860
CMD ["/bin/sh", "-c", "exec /app/.venv/bin/projector demo --serve --host 0.0.0.0 --port ${PORT}"]

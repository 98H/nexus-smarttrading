# Multi-Stage Production Dockerfile for Core Services
# Stage 1: Build & Dependencies
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md* ./
COPY src/ ./src/
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels . && \
    python3 -c "import tomllib, pathlib, glob, os; p = pathlib.Path('pyproject.toml'); name = tomllib.loads(p.read_text()).get('project', {}).get('name', '').replace('-', '_') if p.exists() else ''; [os.remove(f) for f in glob.glob(f'/build/wheels/{name}-*.whl')] if name else None" 2>/dev/null || true

# Stage 2: Minimal Production Runtime
FROM python:3.11-slim-bookworm AS production

ARG APP_USER=core
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CORE_ENV=production \
    CORE_HOST=0.0.0.0 \
    CORE_PORT=8000 \
    CORE_WORKERS=2 \
    CORE_DEBUG=false \
    CORE_LOG_LEVEL=INFO \
    PYTHONPATH="/app" \
    PATH="/home/${APP_USER}/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g "${APP_GID}" "${APP_USER}" \
    && useradd -u "${APP_UID}" -g "${APP_GID}" -s /bin/bash -m -d /home/"${APP_USER}" "${APP_USER}" \
    && mkdir -p /run/core /app \
    && chown -R "${APP_UID}:${APP_GID}" /run/core /app

COPY --from=builder /build/wheels /tmp/wheels
RUN if [ -d /tmp/wheels ] && [ -n "$(ls -A /tmp/wheels/*.whl 2>/dev/null)" ]; then \
        pip install --no-cache-dir /tmp/wheels/*.whl; \
    fi && rm -rf /tmp/wheels

COPY --chown=${APP_UID}:${APP_GID} src/ /app/src/

USER ${APP_UID}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-m", "src.core.healthcheck"]

ENTRYPOINT ["python", "-m", "src.core.runtime_config"]
CMD ["start"]

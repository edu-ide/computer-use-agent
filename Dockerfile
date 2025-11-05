# Stage 1: Builder
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY cua2-front/package*.json ./

RUN npm ci

COPY cua2-front/ ./

RUN npm run build

# Stage 2: Production image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY cua2-core/ ./cua2-core/

RUN pip install --no-cache-dir uv

RUN cd /app && uv sync --frozen

COPY --from=frontend-builder /app/frontend/dist /app/static

COPY nginx.conf /etc/nginx/nginx.conf

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

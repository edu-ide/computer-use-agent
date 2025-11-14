# Stage 1: Builder
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY cua2-front/package*.json ./

RUN npm ci

COPY cua2-front/ ./

RUN npm run build

# Stage 2: Production image
FROM python:3.11-slim

# Install system packages as root
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Create necessary directories with proper permissions for nginx
RUN mkdir -p /var/log/nginx /var/lib/nginx /var/cache/nginx /run \
    && chown -R user:user /var/log/nginx /var/lib/nginx /var/cache/nginx /run \
    && chmod -R 755 /var/log/nginx /var/lib/nginx /var/cache/nginx /run

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Upgrade pip as user
RUN pip install --no-cache-dir --upgrade pip

# Install uv as user
RUN pip install --no-cache-dir uv

# Copy the project files with proper ownership
COPY --chown=user:user pyproject.toml ./
COPY --chown=user:user cua2-core/ ./cua2-core/
COPY --chown=user:user .gitattributes ./
COPY --chown=user:user .gitattributes ./cua2-core/.gitattributes

# Install Python dependencies
RUN uv sync --all-extras

# Copy frontend build with proper ownership
COPY --chown=user:user --from=frontend-builder /app/frontend/dist ./static

# Copy nginx config (user needs read access)
COPY --chown=user:user nginx.conf ./nginx.conf

# Copy entrypoint script with proper ownership and make it executable
COPY --chown=user:user entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

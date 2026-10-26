FROM python:3.11-slim

# Node.js runs generated apps. Playwright installs its own matching Chromium
# build and browser system dependencies below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep these versions aligned with the multiturn project requirements.
RUN pip install --no-cache-dir \
    "httpx>=0.27" \
    "openai>=1.0" \
    "playwright>=1.60" \
    && playwright install --with-deps chromium

WORKDIR /app
RUN git init /app

# syntax=docker/dockerfile:1.4

########################
# Frontend Build Stage #
########################
FROM --platform=$BUILDPLATFORM node:18-alpine AS frontend-builder

WORKDIR /frontend

# 1. Copy only dependency definitions first to leverage cache
COPY frontend/package.json frontend/yarn.lock ./

# 2. Install dependencies with cache mount
RUN --mount=type=cache,target=/usr/local/share/.cache/yarn \
    yarn install --frozen-lockfile

# 3. Copy the rest of the frontend source code
COPY frontend/ ./
# Make addon version metadata available for Vite in container builds.
COPY config.json ./config.json

# 4. Build
RUN yarn build

####################################
# Final Runtime Stage
####################################
FROM python:3.12-slim-bookworm

WORKDIR /docker-app

# 1. Install system dependencies:
#    - build-essential: for Python C extensions
#    - ESP-IDF prerequisites: git, cmake, ninja-build, flex, bison, gperf, ccache,
#      dfu-util, libusb-1.0-0, python3-venv (needed by ESP-IDF's own venv)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git cmake ninja-build flex bison gperf ccache \
    dfu-util libusb-1.0-0 python3-venv

# 2. Install ESP-IDF (shallow clone, esp32 target only)
#    Change IDF_VERSION to pin a specific release, e.g. v5.3.2
ARG IDF_VERSION=v5.4.1
ENV IDF_PATH=/opt/esp-idf

RUN --mount=type=cache,target=/root/.espressif \
    git clone --depth=1 --branch ${IDF_VERSION} --recurse-submodules \
        https://github.com/espressif/esp-idf.git ${IDF_PATH} \
    && cd ${IDF_PATH} \
    && ./install.sh esp32 \
    && rm -rf ${IDF_PATH}/.git

# 3. Copy python requirements first to leverage cache
COPY requirements.txt .

# 4. Install Python requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy backend source code (ignoring files in .dockerignore)
COPY . .

# 4. Copy built frontend assets from the builder stage
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

EXPOSE 8070

CMD ["python", "run.py", "--setup"]

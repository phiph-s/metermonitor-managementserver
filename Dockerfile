# syntax=docker/dockerfile:1.4

########################
# Frontend Build Stage #
########################
FROM --platform=$BUILDPLATFORM node:18-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/yarn.lock ./
RUN --mount=type=cache,target=/usr/local/share/.cache/yarn \
    yarn install --frozen-lockfile

COPY frontend/ ./
COPY config.json ./config.json
RUN yarn build

####################################
# ESP-IDF Base Stage
#
# Isolated so Docker only re-runs this when IDF_VERSION changes.
# For cross-machine caching, push this stage and use --cache-from:
#   docker build --target idf-base -t registry/mm-idf-base:v5.4.1 .
#   docker push registry/mm-idf-base:v5.4.1
#   docker build --cache-from registry/mm-idf-base:v5.4.1 .
####################################
FROM python:3.12-slim-bookworm AS idf-base

ARG IDF_VERSION=v5.4.1
ENV IDF_PATH=/opt/esp-idf

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build flex bison gperf ccache \
    dfu-util libusb-1.0-0 python3-venv

# Tools are installed to /root/.espressif — no cache mount here so they
# land in the image layer and are available at runtime.
RUN git clone --depth=1 --branch ${IDF_VERSION} --recurse-submodules \
        https://github.com/espressif/esp-idf.git ${IDF_PATH} \
    && cd ${IDF_PATH} \
    && ./install.sh esp32 \
    && rm -rf ${IDF_PATH}/.git

####################################
# Final Runtime Stage
####################################
FROM idf-base

WORKDIR /docker-app

# build-essential is only needed to compile Python C extensions
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

EXPOSE 8070

CMD ["python", "run.py", "--setup"]

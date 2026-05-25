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
# Only rebuilds when IDF_VERSION changes. For cross-machine cache sharing:
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

# --shallow-submodules makes submodule clones shallow too (--depth alone doesn't).
# Everything is cleaned up in the same RUN so deleted bytes never land in a layer.
RUN git clone --depth=1 --shallow-submodules --branch ${IDF_VERSION} \
        --recurse-submodules \
        https://github.com/espressif/esp-idf.git ${IDF_PATH} \
    && cd ${IDF_PATH} \
    && ./install.sh esp32 \
    # Replace full git history with a bare init so ESP-IDF cmake scripts
    # (gdbinit.cmake etc.) can still resolve `git rev-parse --show-toplevel`
    # without the hundreds-of-MB history being present at runtime.
    && rm -rf ${IDF_PATH}/.git \
    && git -C ${IDF_PATH} init -q \
    \
    # ── strip tools not needed for compilation ─────────────────────────────
    # Debuggers and JTAG interface
    && rm -rf /root/.espressif/tools/openocd-esp32 \
    && rm -rf /root/.espressif/tools/xtensa-esp-elf-gdb \
    && rm -rf /root/.espressif/tools/riscv32-esp-elf-gdb \
    # RISC-V compiler (ESP32 classic uses Xtensa only)
    && rm -rf /root/.espressif/tools/riscv32-esp-elf \
    # Downloaded archives – already extracted, not needed at runtime
    && rm -rf /root/.espressif/dist \
    \
    # ── strip IDF source content not needed at runtime ─────────────────────
    && rm -rf ${IDF_PATH}/examples \
    && rm -rf ${IDF_PATH}/docs \
    && find ${IDF_PATH}/components -type d \( -name "test" -o -name "test_apps" \) \
         -exec rm -rf {} + 2>/dev/null || true

####################################
# Final Runtime Stage
####################################
FROM idf-base

WORKDIR /docker-app

# build-essential only needed to compile Python C extensions; purged afterwards
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

EXPOSE 8070
CMD ["python", "run.py", "--setup"]

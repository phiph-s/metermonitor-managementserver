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
# Final Runtime Stage
####################################
FROM python:3.12-slim-bookworm

WORKDIR /docker-app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /frontend/dist /docker-app/frontend/dist

EXPOSE 8070
CMD ["python", "run.py", "--setup"]

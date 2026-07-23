# Serves the Bedrock Converse API on :8080, backed by llama.cpp and a baked-in model.
# Nothing external is required at runtime.
#
#   docker build -t fakerock .
#   docker run -p 8099:8080 fakerock
#
# Swap the model at build time:
#   docker build --build-arg MODEL_URL=<gguf url> -t fakerock .
#
# Build args exist so a network without internet access can point each source at a mirror:
#   GO_IMAGE, CURL_IMAGE, BASE_IMAGE, MODEL_URL
# Use a CUDA BASE_IMAGE with LLAMA_NGL>0 to run on a GPU; llama.cpp does not offload by default.
#
# LLAMA_REASONING is off because Converse has nowhere to put thinking: the model would spend its
# budget in a reasoning field that is then discarded, and answer worse for it.

ARG GO_IMAGE=golang:1.25-alpine
ARG CURL_IMAGE=curlimages/curl:latest
ARG BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:server

FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper
ARG TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$TARGETARCH \
    go build -ldflags="-s -w" -o /fakerock ./cmd/fakerock

FROM --platform=$BUILDPLATFORM ${CURL_IMAGE} AS model
ARG MODEL_URL=https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf
RUN curl -fSL --retry 5 --retry-all-errors --retry-delay 5 -o /tmp/model.gguf "$MODEL_URL"

FROM ${BASE_IMAGE}
COPY --from=wrapper /fakerock /usr/local/bin/fakerock
COPY --from=model /tmp/model.gguf /models/model.gguf

ENV LISTEN_ADDR=:8080 \
    BACKEND_BASE_URL=http://127.0.0.1:8081/v1 \
    BACKEND_EMBEDDING_BASE_URL=http://127.0.0.1:8082/v1 \
    BACKEND_MODEL=local \
    LLAMA_CTX=32768 \
    LLAMA_NGL=0 \
    LLAMA_REASONING=off \
    LLAMA_EMBEDDINGS=off

EXPOSE 8080

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

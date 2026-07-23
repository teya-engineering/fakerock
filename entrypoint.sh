#!/bin/bash
set -euo pipefail

# Chat server. --jinja is what makes tool calling work; without it llama.cpp ignores the tool
# definitions and the agent loop ends after one turn with no error.
/app/llama-server \
  --model /models/model.gguf \
  --jinja \
  --reasoning "${LLAMA_REASONING}" \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size "${LLAMA_CTX}" \
  --n-gpu-layers "${LLAMA_NGL}" &
chat_pid=$!

trap 'kill "${chat_pid}" "${embed_pid:-}" 2>/dev/null || true' TERM INT

wait_for() {
  local port=$1 pid=$2 name=$3
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do
    kill -0 "${pid}" 2>/dev/null || { echo "${name} exited before becoming ready" >&2; exit 1; }
    sleep 1
  done
  echo "${name} ready"
}

# Embeddings are off by default: they load the bundled model a second time (llama-server runs in
# chat or embedding mode per process, not both). Set LLAMA_EMBEDDINGS=on to serve /v1/embeddings.
if [ "${LLAMA_EMBEDDINGS:-off}" = "on" ]; then
  /app/llama-server \
    --model /models/model.gguf \
    --embeddings \
    --pooling mean \
    --host 127.0.0.1 \
    --port 8082 \
    --ctx-size "${LLAMA_CTX}" \
    --n-gpu-layers "${LLAMA_NGL}" &
  embed_pid=$!
fi

wait_for 8081 "${chat_pid}" "chat llama-server"
if [ -n "${embed_pid:-}" ]; then
  wait_for 8082 "${embed_pid}" "embeddings llama-server"
fi

exec /usr/local/bin/fakerock

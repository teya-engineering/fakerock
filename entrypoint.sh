#!/bin/bash
set -euo pipefail

# The bundled llama-servers exist only to back the default BACKEND_BASE_URL and
# BACKEND_EMBEDDING_BASE_URL. Pointing either at an external backend - host ollama, a shared
# endpoint - makes that server dead weight: it still loads the model and a full KV cache, and with
# LLAMA_NGL=0 that sits in host RAM. Start each one only when it is what the wrapper routes to.
serves_bundled() {
  case "$1/" in
    *://127.0.0.1:"$2"/*|*://localhost:"$2"/*|*://\[::1\]:"$2"/*) return 0 ;;
    *) return 1 ;;
  esac
}

chat_pid=""
embed_pid=""

trap 'kill "${chat_pid:-}" "${embed_pid:-}" 2>/dev/null || true' TERM INT

# Chat server. --jinja is what makes tool calling work; without it llama.cpp ignores the tool
# definitions and the agent loop ends after one turn with no error.
if serves_bundled "${BACKEND_BASE_URL}" 8081; then
  /app/llama-server \
    --model /models/model.gguf \
    --jinja \
    --reasoning "${LLAMA_REASONING}" \
    --host 127.0.0.1 \
    --port 8081 \
    --ctx-size "${LLAMA_CTX}" \
    --n-gpu-layers "${LLAMA_NGL}" &
  chat_pid=$!
else
  echo "chat: serving ${BACKEND_BASE_URL}, bundled llama-server not started"
fi

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
  if serves_bundled "${BACKEND_EMBEDDING_BASE_URL:-}" 8082; then
    /app/llama-server \
      --model /models/model.gguf \
      --embeddings \
      --pooling mean \
      --host 127.0.0.1 \
      --port 8082 \
      --ctx-size "${LLAMA_CTX}" \
      --n-gpu-layers "${LLAMA_NGL}" &
    embed_pid=$!
  else
    echo "embeddings: serving ${BACKEND_EMBEDDING_BASE_URL:-unset}, bundled llama-server not started"
  fi
fi

if [ -n "${chat_pid}" ]; then
  wait_for 8081 "${chat_pid}" "chat llama-server"
fi
if [ -n "${embed_pid}" ]; then
  wait_for 8082 "${embed_pid}" "embeddings llama-server"
fi

# Warmup: one tiny completion through the chat template before the Bedrock API opens.
# It absorbs first-inference setup costs and proves the server can actually generate,
# so a broken model or template fails here in the logs, not in the first caller.
# Because fakerock only starts listening afterwards, a TCP readiness probe on the API
# port waits for the warmup too. Set LLAMA_WARMUP=off to skip it.
# Only the bundled server is warmed: an external backend owns its own model lifecycle, and the
# placeholder model id below does not resolve there.
if [ -n "${chat_pid}" ] && [ "${LLAMA_WARMUP:-on}" = "on" ]; then
  echo "warmup: sending a one-token completion"
  warmup_start=$(date +%s)
  if curl -sS -X POST "http://127.0.0.1:8081/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"warmup","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
      -o /dev/null; then
    echo "warmup: done in $(( $(date +%s) - warmup_start ))s"
  else
    echo "warmup: request failed, starting anyway" >&2
  fi
fi

exec /usr/local/bin/fakerock

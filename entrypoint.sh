#!/bin/bash
set -euo pipefail

# --jinja is what makes tool calling work; without it llama.cpp ignores the tool
# definitions and the agent loop ends after one turn with no error.
/app/llama-server \
  --model /models/model.gguf \
  --jinja \
  --reasoning "${LLAMA_REASONING}" \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size "${LLAMA_CTX}" \
  --n-gpu-layers "${LLAMA_NGL}" &
llama_pid=$!
trap 'kill "${llama_pid}" 2>/dev/null || true' TERM INT

until curl -sf http://127.0.0.1:8081/health >/dev/null; do
  kill -0 "${llama_pid}" 2>/dev/null || { echo "llama-server exited before becoming ready" >&2; exit 1; }
  sleep 1
done
echo "llama-server ready"

exec /usr/local/bin/fakerock

# Shared entrypoint helpers. Sourced by entrypoint.sh and entrypoint_test.sh.

# True when url points at the bundled llama-server on port (loopback hosts only).
serves_bundled() {
  case "$1/" in
    *://127.0.0.1:"$2"/*|*://localhost:"$2"/*|*://\[::1\]:"$2"/*) return 0 ;;
    *) return 1 ;;
  esac
}

chat_uses_bundled() {
  serves_bundled "${BACKEND_BASE_URL:-}" 8081
}

embed_uses_bundled() {
  serves_bundled "${BACKEND_EMBEDDING_BASE_URL:-}" 8082
}

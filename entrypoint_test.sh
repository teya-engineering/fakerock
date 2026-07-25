#!/usr/bin/env bash
# Tests entrypoint_lib.sh. Run from the repo root: ./entrypoint_test.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=entrypoint_lib.sh
source "${root}/entrypoint_lib.sh"

failures=0

pass() {
  echo "PASS: $1"
}

fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

# want: 0 = serves_bundled true, 1 = false
assert_serves_bundled() {
  local name=$1 url=$2 port=$3 want=$4
  if serves_bundled "$url" "$port"; then
    got=0
  else
    got=1
  fi
  if [ "$got" -eq "$want" ]; then
    pass "$name"
  else
    fail "$name (url=$url port=$port want=$want got=$got)"
  fi
}

assert_chat_bundled() {
  local name=$1 want=$2
  BACKEND_BASE_URL=$3
  if chat_uses_bundled; then
    got=0
  else
    got=1
  fi
  if [ "$got" -eq "$want" ]; then
    pass "$name"
  else
    fail "$name (BACKEND_BASE_URL=$3 want=$want got=$got)"
  fi
}

echo "entrypoint_lib"

assert_serves_bundled "127.0.0.1 bundled chat" "http://127.0.0.1:8081/v1" 8081 0
assert_serves_bundled "127.0.0.1 bundled chat trailing slash" "http://127.0.0.1:8081/" 8081 0
assert_serves_bundled "localhost bundled chat" "http://localhost:8081/v1" 8081 0
assert_serves_bundled "ipv6 loopback bundled chat" "http://[::1]:8081/v1" 8081 0
assert_serves_bundled "external ollama" "http://host.docker.internal:11434/v1" 8081 1
assert_serves_bundled "wrong port" "http://127.0.0.1:8082/v1" 8081 1
assert_serves_bundled "empty url" "" 8081 1
assert_serves_bundled "remote host" "http://192.168.1.10:8081/v1" 8081 1

assert_chat_bundled "default chat url" 0 "http://127.0.0.1:8081/v1"
assert_chat_bundled "external chat url" 1 "http://host.docker.internal:11434/v1"

if bash -n "${root}/entrypoint.sh"; then
  pass "entrypoint.sh syntax"
else
  fail "entrypoint.sh syntax"
fi

if [ "$failures" -eq 0 ]; then
  echo "All entrypoint tests passed."
  exit 0
fi

echo "${failures} entrypoint test(s) failed."
exit 1

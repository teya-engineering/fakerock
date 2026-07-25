#!/usr/bin/env bash
# Tests entrypoint flag gating. Run from the repo root: ./entrypoint_test.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failures=0

pass() {
  echo "PASS: $1"
}

fail() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

# want: 0 = chat enabled, 1 = disabled
assert_llama_chat() {
  local name=$1 want=$2 value=$3
  if [ -n "$value" ]; then
    LLAMA_CHAT=$value
  else
    unset LLAMA_CHAT
  fi
  if [ "${LLAMA_CHAT:-on}" = "on" ]; then
    got=0
  else
    got=1
  fi
  if [ "$got" -eq "$want" ]; then
    pass "$name"
  else
    fail "$name (LLAMA_CHAT=${LLAMA_CHAT-<unset>} want=$want got=$got)"
  fi
}

echo "entrypoint flags"

assert_llama_chat "default on when unset" 0 ""
assert_llama_chat "explicit on" 0 "on"
assert_llama_chat "explicit off" 1 "off"

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

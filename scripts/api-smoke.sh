#!/usr/bin/env bash
# api-smoke.sh - start `jev-router serve` and exercise the HTTP API end to end.
#
# Usage:
#   scripts/api-smoke.sh [--config PATH] [--addr HOST:PORT] [--no-complete] [--help]
#
#   --config PATH    configuration file for `serve` (default: $JEV_ROUTER_CONFIG,
#                    else ./jev-router.local.yaml when that file exists, else
#                    environment-only mode)
#   --addr HOST:PORT listen address (default: 127.0.0.1:8199); passed to the
#                    server as JEV_ROUTER_SERVER_ADDR, overriding the config
#   --no-complete    skip the billed completion check (all other checks run)
#
# The script builds ./cmd/jev-router into a temp dir, starts the server, waits
# for GET /healthz to answer, then calls:
#   1. GET  /healthz        -> 200 {"status":"ok"}
#   2. GET  /nope           -> 404 {"code":"not_found"}
#   3. GET  /v1/route       -> 405 with Allow: POST
#   4. POST /v1/route       -> 200 decision with a model and candidate chain
#   5. POST /v1/complete    -> 200 full JSON result (one small BILLED request;
#                              also sends a legacy "stream":true field, which
#                              must be ignored and answered with plain JSON)
#   6. POST /v1/route {}    -> 400 {"code":"invalid_request"}
#   7. GET  /v1/models      -> 200 OpenAI list containing "auto"
#   8. POST /v1/chat/completions stream:true -> 400 OpenAI error envelope
#   9. POST /v1/chat/completions -> 200 chat.completion (BILLED; skipped with --no-complete)
#
# Exit status is 0 when every check passes, 1 otherwise; on failure the server
# log and temp dir are kept for debugging.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${JEV_ROUTER_CONFIG:-}"
ADDR="127.0.0.1:8199"
NO_COMPLETE=0

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    --addr) ADDR="${2:?--addr needs HOST:PORT}"; shift 2 ;;
    --no-complete) NO_COMPLETE=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown flag: $1" >&2; usage 1 ;;
  esac
done

if [[ -z "$CONFIG" && -f "$REPO_ROOT/jev-router.local.yaml" ]]; then
  CONFIG="$REPO_ROOT/jev-router.local.yaml"
fi

for cmd in go curl python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "missing required command: $cmd" >&2; exit 1; }
done

TMP_DIR="$(mktemp -d)"
BIN="$TMP_DIR/jev-router"
LOG="$TMP_DIR/server.log"
BODY="$TMP_DIR/body.json"
HDRS="$TMP_DIR/headers.txt"
SERVER_PID=""
PASSED=0
FAILED=0
KEEP=0

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$KEEP" == 0 ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT INT TERM

pass() { PASSED=$((PASSED + 1)); printf '  PASS  %s\n' "$1"; }
fail() { FAILED=$((FAILED + 1)); printf '  FAIL  %s\n' "$1"; }

# jsonq 'd["key"]' reads JSON on stdin and prints the Python expression.
jsonq() {
  python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('')
    sys.exit(0)
try:
    v = $1
except Exception:
    v = ''
print('' if v is None else v)
"
}

# request METHOD PATH [curl args...] fills $BODY/$HDRS and sets $HTTP_CODE.
HTTP_CODE=""
request() {
  local method="$1" path="$2"
  shift 2
  : > "$BODY"
  if ! HTTP_CODE=$(curl -sS --max-time 30 -o "$BODY" -D "$HDRS" -w '%{http_code}' \
      -X "$method" "http://$ADDR$path" "$@"); then
    HTTP_CODE="000"
  fi
}

header_value() { # header_value Name -> first value, CR stripped
  awk -v name="$1" 'BEGIN{IGNORECASE=1} index($0, name ":") == 1 {sub(/\r$/, ""); print substr($0, length(name) + 3); exit}' "$HDRS"
}

echo '== jev-router API smoke test =='
echo "config: ${CONFIG:-<environment-only defaults>}"
echo "addr:   $ADDR"
echo "note:   checks 5 and 9 each perform one small billed completion request (skip with --no-complete)"
echo

echo "building ./cmd/jev-router ..."
if ! go build -o "$BIN" ./cmd/jev-router; then
  echo 'build failed' >&2
  exit 1
fi

serve_args=(serve)
[[ -n "$CONFIG" ]] && serve_args+=(--config "$CONFIG")
JEV_ROUTER_SERVER_ADDR="$ADDR" "$BIN" "${serve_args[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!

echo "waiting for http://$ADDR/healthz ..."
READY=0
for _ in $(seq 1 100); do
  if curl -fsS --max-time 2 "http://$ADDR/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.2
done
if [[ "$READY" != 1 ]]; then
  echo 'server did not become ready; log:' >&2
  tail -20 "$LOG" >&2 || true
  KEEP=1
  exit 1
fi
echo 'server is up'
echo

TOTAL=9
N=0
next() { N=$((N + 1)); printf '[%d/%d] %s\n' "$N" "$TOTAL" "$1"; }

# 1. healthz
next 'GET /healthz'
request GET /healthz
if [[ "$HTTP_CODE" == 200 && "$(jsonq 'd["status"]' < "$BODY")" == ok ]]; then
  pass 'healthz answered {"status":"ok"}'
else
  fail "GET /healthz -> $HTTP_CODE ($(cat "$BODY"))"
fi

# 2. unknown path -> JSON 404
next 'GET /nope (JSON 404)'
request GET /nope
if [[ "$HTTP_CODE" == 404 && "$(jsonq 'd["code"]' < "$BODY")" == not_found ]]; then
  pass 'unknown path answered 404 {"code":"not_found"}'
else
  fail "GET /nope -> $HTTP_CODE ($(cat "$BODY"))"
fi

# 3. wrong method -> 405 with Allow header
next 'GET /v1/route (405 + Allow: POST)'
request GET /v1/route
if [[ "$HTTP_CODE" == 405 && "$(header_value Allow)" == POST ]]; then
  pass 'wrong method answered 405 with Allow: POST'
else
  fail "GET /v1/route -> $HTTP_CODE (Allow: $(header_value Allow))"
fi

# 4. routing decision
next 'POST /v1/route'
request POST /v1/route -H 'Content-Type: application/json' \
  -d '{"prompt":"Fix this off-by-one bug in a Go loop that skips the last element."}'
ROUTE_MODEL="$(jsonq 'd["model_id"]' < "$BODY")"
ROUTE_SOURCE="$(jsonq 'd["source"]' < "$BODY")"
ROUTE_CHAIN="$(jsonq 'len(d["candidates"])' < "$BODY")"
if [[ "$HTTP_CODE" == 200 && -n "$ROUTE_MODEL" && -n "$ROUTE_CHAIN" && "$ROUTE_CHAIN" -ge 1 ]]; then
  pass "routed to $ROUTE_MODEL (source: $ROUTE_SOURCE, candidates: $ROUTE_CHAIN)"
else
  fail "POST /v1/route -> $HTTP_CODE ($(head -c 300 "$BODY"))"
fi

# 5. completion - full JSON result; legacy "stream":true must be ignored
next 'POST /v1/complete (billed)'
if [[ "$NO_COMPLETE" == 1 ]]; then
  printf '  SKIP  --no-complete\n'
else
  request POST /v1/complete -H 'Content-Type: application/json' \
    -d '{"prompt":"What is the capital of France? Answer in one word.","max_tokens":16,"stream":true}'
  CTYPE="$(header_value Content-Type)"
  COMPLETE_MODEL="$(jsonq 'd["routing_decision"]["model_id"]' < "$BODY")"
  COMPLETE_CONTENT="$(jsonq 'd.get("content") or ""' < "$BODY")"
  COMPLETE_USAGE="$(jsonq 'd.get("usage")' < "$BODY")"
  if [[ "$HTTP_CODE" == 200 && "$CTYPE" == application/json* && -n "$COMPLETE_MODEL" && ( -n "$COMPLETE_CONTENT" || -n "$COMPLETE_USAGE" ) ]]; then
    pass "served by $COMPLETE_MODEL, Content-Type: $CTYPE (legacy stream field ignored), content: ${COMPLETE_CONTENT:0:40}"
  else
    fail "POST /v1/complete -> $HTTP_CODE, Content-Type: $CTYPE ($(head -c 300 "$BODY"))"
  fi
fi

# 6. invalid request -> JSON 400
next 'POST /v1/route {} (400)'
request POST /v1/route -H 'Content-Type: application/json' -d '{}'
if [[ "$HTTP_CODE" == 400 && "$(jsonq 'd["code"]' < "$BODY")" == invalid_request ]]; then
  pass 'missing prompt answered 400 {"code":"invalid_request"}'
else
  fail "POST /v1/route {} -> $HTTP_CODE ($(head -c 200 "$BODY"))"
fi

# 7. OpenAI-compatible model list
next 'GET /v1/models'
request GET /v1/models
if [[ "$HTTP_CODE" == 200 && "$(jsonq 'd["object"]' < "$BODY")" == list && "$(jsonq 'd["data"][0]["id"]' < "$BODY")" == auto ]]; then
  pass 'model list answered in OpenAI shape with the auto model'
else
  fail "GET /v1/models -> $HTTP_CODE ($(head -c 200 "$BODY"))"
fi

# 8. streaming is rejected in OpenAI's error envelope
next 'POST /v1/chat/completions stream:true (400)'
request POST /v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"hi"}]}'
if [[ "$HTTP_CODE" == 400 && "$(jsonq 'd["error"]["code"]' < "$BODY")" == streaming_not_supported ]]; then
  pass 'stream:true answered 400 with an OpenAI error envelope'
else
  fail "POST /v1/chat/completions stream:true -> $HTTP_CODE ($(head -c 200 "$BODY"))"
fi

# 9. OpenAI-compatible completion (billed)
next 'POST /v1/chat/completions (billed)'
if [[ "$NO_COMPLETE" == 1 ]]; then
  printf '  SKIP  --no-complete\n'
else
  request POST /v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"auto","max_tokens":16,"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}'
  CHAT_OBJECT="$(jsonq 'd["object"]' < "$BODY")"
  CHAT_CONTENT="$(jsonq 'd["choices"][0]["message"]["content"]' < "$BODY")"
  if [[ "$HTTP_CODE" == 200 && "$CHAT_OBJECT" == chat.completion && -n "$CHAT_CONTENT" ]]; then
    pass "chat.completion from $(jsonq 'd["model"]' < "$BODY"): ${CHAT_CONTENT:0:40}"
  else
    fail "POST /v1/chat/completions -> $HTTP_CODE ($(head -c 300 "$BODY"))"
  fi
fi

echo
echo "== $PASSED passed, $FAILED failed =="
if [[ "$FAILED" != 0 ]]; then
  KEEP=1
  echo "server log kept at: $LOG"
  tail -20 "$LOG" || true
  exit 1
fi

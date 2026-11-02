#!/usr/bin/env bash
# run_aiperf_harness.sh — run vLLM + kvbench benchmark with aiperf
#
# Sequential cases use --shared-system-prompt-length 256 so all requests
# share a 256-token prefix (1 LMCache chunk), triggering io.write on
# request 1 and io.read on all subsequent requests.
#
# Multi-turn uses 512 input + 64 output per turn (2 chunks per turn).
#
# Usage:
#   bash scripts/run_aiperf_harness.sh [OUTPUT_DIR]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/.venv"
AIPERF="/home/admin/aditya/ai-perf/bin/aiperf"
LMCACHE_CFG="$REPO_DIR/run/lmcache.yaml"
LMCACHE_CACHE="$REPO_DIR/run/lmcache-cache"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT=8000
URL="http://localhost:$PORT"
HF_TOKEN="${HF_TOKEN:-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}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$REPO_DIR/traces/aiperf_run_$TS}"
mkdir -p "$OUT"

# ── Sequential experiment params ──────────────────────────────────────────────
SEQ_SHARED_PREFIX=256      # 1 LMCache chunk — same for all requests
SEQ_UNIQUE=1792            # unique part per request (total = 2048)
SEQ_OUTPUT=64

# ── Multi-turn experiment params ──────────────────────────────────────────────
MT_INPUT=512               # input tokens per turn (2 chunks)
MT_OUTPUT=64               # output tokens per turn

echo "══════════════════════════════════════════════════════════"
echo "  aiperf harness — vLLM + kvbench"
echo "  output: $OUT"
echo "  sequential total input: $((SEQ_SHARED_PREFIX + SEQ_UNIQUE)) tokens"
echo "          (shared prefix=$SEQ_SHARED_PREFIX + unique=$SEQ_UNIQUE)"
echo "  multi-turn: ${MT_INPUT} input / ${MT_OUTPUT} output per turn"
echo "══════════════════════════════════════════════════════════"

# ── Helpers ───────────────────────────────────────────────────────────────────
VLLM_PID=""
KB_PID=""
OTLP_PID=""

wait_server() {
    local url="$1"; local timeout="${2:-120}"
    echo -n "  waiting for server ..."
    for i in $(seq 1 $timeout); do
        curl -sf "$url/health" >/dev/null 2>&1 && { echo " ready"; return 0; }
        echo -n "."; sleep 2
    done
    echo " TIMEOUT"; return 1
}

stop_vllm() {
    [ -n "$VLLM_PID" ] || return 0
    local pid="$VLLM_PID"; VLLM_PID=""
    echo "  stopping vLLM ($pid)..."
    # Kill process group and all children immediately
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
    sleep 5
    kill -9 "$pid" 2>/dev/null || true
    pkill -9 -P "$pid" 2>/dev/null || true
    # Also kill any stray vllm processes
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "EngineCore\|VLLM" 2>/dev/null || true
    sleep 2
    echo "  vLLM stopped"
}

stop_kb() {
    [ -n "$KB_PID" ] || return
    kill "$KB_PID" 2>/dev/null; sleep 2; kill -9 "$KB_PID" 2>/dev/null || true
    KB_PID=""
}

stop_otlp() {
    [ -n "$OTLP_PID" ] || return
    kill "$OTLP_PID" 2>/dev/null || true; OTLP_PID=""
}

clean_lmcache() {
    rm -rf "$LMCACHE_CACHE"; mkdir -p "$LMCACHE_CACHE"
    echo "  lmcache cleared"
}

trap 'stop_vllm; stop_kb; stop_otlp' EXIT

# ── aiperf runner ─────────────────────────────────────────────────────────────
run_aiperf_seq() {
    local tag="$1" requests="$2" concurrency="$3"
    local artifact_dir="$OUT/$tag/aiperf"
    mkdir -p "$artifact_dir"
    echo "  running aiperf: $tag ($requests requests, conc=$concurrency)"
    "$AIPERF" profile \
        --model "$MODEL" --url "$URL" \
        --endpoint-type chat --streaming \
        --concurrency "$concurrency" \
        --request-count "$requests" \
        --shared-system-prompt-length "$SEQ_SHARED_PREFIX" \
        --prompt-input-tokens-mean "$SEQ_UNIQUE" \
        --prompt-input-tokens-stddev 0 \
        --prompt-output-tokens-mean "$SEQ_OUTPUT" \
        --prompt-output-tokens-stddev 0 \
        --output-artifact-dir "$artifact_dir" \
        2>&1 | tee "$OUT/$tag/aiperf.log" | tail -5
}

run_aiperf_mt() {
    local tag="$1" conversations="$2" turns="$3"
    local artifact_dir="$OUT/$tag/aiperf"
    mkdir -p "$artifact_dir"
    echo "  running aiperf: $tag ($conversations conversations × $turns turns)"
    "$AIPERF" profile \
        --model "$MODEL" --url "$URL" \
        --endpoint-type chat --streaming \
        --concurrency 1 \
        --conversation-num "$conversations" \
        --conversation-turn-mean "$turns" \
        --prompt-input-tokens-mean "$MT_INPUT" \
        --prompt-input-tokens-stddev 0 \
        --prompt-output-tokens-mean "$MT_OUTPUT" \
        --prompt-output-tokens-stddev 0 \
        --output-artifact-dir "$artifact_dir" \
        2>&1 | tee "$OUT/$tag/aiperf.log" | tail -5
}

# ── vLLM phase ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: vLLM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_vllm_case() {
    local TAG="$1"; shift
    echo ""
    echo "── vLLM | $TAG ──"
    local CASE_DIR="$OUT/$TAG"; mkdir -p "$CASE_DIR"
    clean_lmcache

    KV_CFG="{\"kv_connector\":\"TracingLMCacheConnector\",\"kv_connector_module_path\":\"kvbench.vllm_trace.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"trace_file\":\"$CASE_DIR/vllm-kv-trace.jsonl\"}}"

    "$VENV/bin/kvbench" otlp-receive \
        --trace-file "$CASE_DIR/vllm-request-trace.jsonl" --port 4318 &
    OTLP_PID=$!; sleep 1

    OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
    LMCACHE_CONFIG_FILE="$LMCACHE_CFG" \
    HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    "$VENV/bin/vllm" serve "$MODEL" \
        --otlp-traces-endpoint "http://localhost:4318/v1/traces" \
        --kv-transfer-config "$KV_CFG" \
        --gpu-memory-utilization 0.5 \
        --no-enable-prefix-caching --enforce-eager --port "$PORT" \
        >"$CASE_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!

    wait_server "$URL" || { stop_vllm; stop_otlp; return 1; }
    "$@"
    stop_vllm; sleep 2; stop_otlp; sleep 1
    echo "  traces: $(ls -lh $CASE_DIR/*.jsonl 2>/dev/null | awk '{print $5,$9}' | tr '\n' '  ')"
}

run_vllm_case "seq_100"      run_aiperf_seq seq_100       100  1
run_vllm_case "seq_1000"     run_aiperf_seq seq_1000     1000  1
run_vllm_case "conc_5"       run_aiperf_seq conc_5        100  5
run_vllm_case "conc_10"      run_aiperf_seq conc_10       100 10
run_vllm_case "mt_10c_5t"    run_aiperf_mt  mt_10c_5t     10   5
run_vllm_case "mt_50c_5t"    run_aiperf_mt  mt_50c_5t     50   5

# ── kvbench phase ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: kvbench"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_kb_case() {
    local TAG="$1"; shift
    echo ""
    echo "── kvbench | $TAG ──"
    local CASE_DIR="$OUT/$TAG"; mkdir -p "$CASE_DIR"
    clean_lmcache

    "$VENV/bin/kvbench" serve \
        --calibrated-timing --gpu GB10 --model llama-3.2-1b \
        --lmcache-config "$LMCACHE_CFG" \
        --trace-file "$CASE_DIR/kvbench-trace.jsonl" \
        --port "$PORT" --type combined &
    KB_PID=$!

    wait_server "$URL" 30 || { stop_kb; return 1; }
    "$@"
    stop_kb; sleep 1
    echo "  trace: $(ls -lh $CASE_DIR/kvbench-trace.jsonl 2>/dev/null | awk '{print $5,$9}')"
}

run_kb_case "seq_100"      run_aiperf_seq seq_100       100  1
run_kb_case "seq_1000"     run_aiperf_seq seq_1000     1000  1
run_kb_case "conc_5"       run_aiperf_seq conc_5        100  5
run_kb_case "conc_10"      run_aiperf_seq conc_10       100 10
run_kb_case "mt_10c_5t"    run_aiperf_mt  mt_10c_5t     10   5
run_kb_case "mt_50c_5t"    run_aiperf_mt  mt_50c_5t     50   5

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  All done. Output: $OUT"
echo "══════════════════════════════════════════════════════════"

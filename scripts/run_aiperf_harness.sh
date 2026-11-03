#!/usr/bin/env bash
# run_aiperf_harness.sh — run vLLM + kvbench benchmark with aiperf
#
# All paths are relative to the repo root (REPO_DIR).
# VLLM_VENV defaults to ../vllm/.venv (sibling of the repo) and can be
# overridden by setting VLLM_VENV in the environment.
#
# Usage:
#   bash scripts/run_aiperf_harness.sh [OPTIONS] [OUTPUT_DIR]
#
# Options:
#   --cases  CASE[,CASE,...]   Comma-separated list of cases to run.
#                              Available: seq_100 seq_1000 conc_5 conc_10
#                                         mt_10c_5t mt_50c_5t
#                              Default: all
#   --phase  vllm|kvbench|both  Which phases to run (default: both)
#   --gpu    PROFILE           kvbench GPU profile (default: RTX_PRO_6000_BW)
#   --gpu-util FLOAT           vLLM --gpu-memory-utilization (default: 0.9)
#
# Examples:
#   bash scripts/run_aiperf_harness.sh --cases seq_100,seq_1000
#   bash scripts/run_aiperf_harness.sh --cases seq_100 --phase kvbench
#   bash scripts/run_aiperf_harness.sh --gpu H100_SXM --cases seq_1000

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Paths (all relative to REPO_DIR or overridable via env) ──────────────────
KB_VENV="$REPO_DIR/.venv"
VLLM_VENV="${VLLM_VENV:-$(dirname "$REPO_DIR")/vllm/.venv}"
AIPERF="${AIPERF:-$KB_VENV/bin/aiperf}"
LMCACHE_CFG="$REPO_DIR/run/lmcache.yaml"
LMCACHE_CACHE="$REPO_DIR/run/lmcache-cache"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN="${HF_TOKEN:-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}"
PORT=8000

# ── Defaults ──────────────────────────────────────────────────────────────────
OPT_CASES="all"
OPT_PHASE="both"
OPT_GPU="RTX_PRO_6000_BW"
OPT_GPU_UTIL="0.9"
OPT_OUT=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)   OPT_CASES="$2";    shift 2 ;;
        --phase)   OPT_PHASE="$2";    shift 2 ;;
        --gpu)     OPT_GPU="$2";      shift 2 ;;
        --gpu-util) OPT_GPU_UTIL="$2"; shift 2 ;;
        --*)       echo "Unknown option: $1"; exit 1 ;;
        *)         OPT_OUT="$1";      shift ;;
    esac
done

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OPT_OUT:-$REPO_DIR/traces/aiperf_run_$TS}"
mkdir -p "$OUT"

# ── Sequential experiment params ──────────────────────────────────────────────
SEQ_SHARED_PREFIX=256
SEQ_UNIQUE=1792
SEQ_OUTPUT=64

# ── Multi-turn experiment params ──────────────────────────────────────────────
MT_INPUT=512
MT_OUTPUT=64

# ── Validate paths ────────────────────────────────────────────────────────────
if [[ ! -f "$AIPERF" ]]; then
    echo "ERROR: aiperf not found at $AIPERF"
    echo "       Override with: AIPERF=/path/to/aiperf $0"
    exit 1
fi
if [[ "$OPT_PHASE" != "kvbench" && ! -f "$VLLM_VENV/bin/vllm" ]]; then
    echo "ERROR: vLLM not found at $VLLM_VENV/bin/vllm"
    echo "       Override with: VLLM_VENV=/path/to/vllm/.venv $0"
    exit 1
fi

echo "══════════════════════════════════════════════════════════"
echo "  aiperf harness"
echo "  output:    $OUT"
echo "  cases:     $OPT_CASES"
echo "  phase:     $OPT_PHASE"
echo "  kvbench gpu: $OPT_GPU"
echo "  vllm venv: $VLLM_VENV"
echo "  aiperf:    $AIPERF"
echo "══════════════════════════════════════════════════════════"

# ── Process case filter ────────────────────────────────────────────────────────
ALL_CASES=(seq_100 seq_1000 conc_5 conc_10 mt_10c_5t mt_50c_5t)

case_enabled() {
    local c="$1"
    [[ "$OPT_CASES" == "all" ]] && return 0
    echo "$OPT_CASES" | tr ',' '\n' | grep -qx "$c"
}

# ── PIDs ──────────────────────────────────────────────────────────────────────
VLLM_PID=""
KB_PID=""
OTLP_PID=""

wait_server() {
    local url="$1"; local timeout="${2:-180}"
    echo -n "  waiting for server"
    for i in $(seq 1 $timeout); do
        curl -sf "$url/health" >/dev/null 2>&1 && { echo " ready (${i}s)"; return 0; }
        echo -n "."; sleep 1
    done
    echo " TIMEOUT"; return 1
}

stop_vllm() {
    [ -z "$VLLM_PID" ] && return 0
    local pid="$VLLM_PID"; VLLM_PID=""
    echo "  stopping vLLM ($pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    kill -9 "$pid" 2>/dev/null || true
    # Kill any EngineCore still holding GPU memory
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 3
    echo "  vLLM stopped"
}

stop_kb() {
    [ -z "$KB_PID" ] && return
    kill "$KB_PID" 2>/dev/null; sleep 2; kill -9 "$KB_PID" 2>/dev/null || true
    KB_PID=""
}

stop_otlp() {
    [ -z "$OTLP_PID" ] && return
    kill "$OTLP_PID" 2>/dev/null || true; OTLP_PID=""
}

clean_lmcache() {
    rm -rf "$LMCACHE_CACHE"; mkdir -p "$LMCACHE_CACHE"
    echo "  lmcache cleared"
}

trap 'stop_vllm; stop_kb; stop_otlp' EXIT

# ── aiperf runners ────────────────────────────────────────────────────────────
run_aiperf_seq() {
    local tag="$1" requests="$2" concurrency="$3"
    local art_dir="$OUT/$tag/aiperf"
    mkdir -p "$art_dir"
    echo "  aiperf: $tag  requests=$requests  concurrency=$concurrency"
    HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    "$AIPERF" profile \
        --model "$MODEL" --url "http://localhost:$PORT" \
        --endpoint-type chat --streaming \
        --concurrency "$concurrency" \
        --request-count "$requests" \
        --shared-system-prompt-length "$SEQ_SHARED_PREFIX" \
        --prompt-input-tokens-mean "$SEQ_UNIQUE" \
        --prompt-input-tokens-stddev 0 \
        --prompt-output-tokens-mean "$SEQ_OUTPUT" \
        --prompt-output-tokens-stddev 0 \
        --output-artifact-dir "$art_dir" \
        2>&1 | tee "$OUT/$tag/aiperf.log" | tail -3
}

run_aiperf_mt() {
    local tag="$1" conversations="$2" turns="$3"
    local art_dir="$OUT/$tag/aiperf"
    mkdir -p "$art_dir"
    echo "  aiperf: $tag  conversations=$conversations  turns=$turns"
    HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    "$AIPERF" profile \
        --model "$MODEL" --url "http://localhost:$PORT" \
        --endpoint-type chat --streaming \
        --concurrency 1 \
        --conversation-num "$conversations" \
        --conversation-turn-mean "$turns" \
        --prompt-input-tokens-mean "$MT_INPUT" \
        --prompt-input-tokens-stddev 0 \
        --prompt-output-tokens-mean "$MT_OUTPUT" \
        --prompt-output-tokens-stddev 0 \
        --output-artifact-dir "$art_dir" \
        2>&1 | tee "$OUT/$tag/aiperf.log" | tail -3
}

# ── vLLM phase ────────────────────────────────────────────────────────────────
if [[ "$OPT_PHASE" == "vllm" || "$OPT_PHASE" == "both" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 1: vLLM  (TracingLMCacheConnector)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_vllm_case() {
        local TAG="$1"; shift
        case_enabled "$TAG" || return 0
        echo ""
        echo "── vLLM | $TAG ──"
        local CASE_DIR="$OUT/$TAG"; mkdir -p "$CASE_DIR"
        clean_lmcache

        > "$CASE_DIR/vllm-request-trace.jsonl"
        > "$CASE_DIR/vllm-kv-trace.jsonl"

        "$KB_VENV/bin/kvbench" otlp-receive \
            --trace-file "$CASE_DIR/vllm-request-trace.jsonl" --port 4318 \
            > "$CASE_DIR/otlp.log" 2>&1 &
        OTLP_PID=$!; sleep 1

        local KV_CFG
        KV_CFG='{"kv_connector":"TracingLMCacheConnector","kv_connector_module_path":"kvbench.vllm_trace.connector","kv_role":"kv_both","kv_connector_extra_config":{"trace_file":"'"$CASE_DIR"'/vllm-kv-trace.jsonl"}}'

        CUDA_VISIBLE_DEVICES=0 \
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
        LMCACHE_CONFIG_FILE="$LMCACHE_CFG" \
        HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        PATH="$VLLM_VENV/bin:$PATH" \
        "$VLLM_VENV/bin/vllm" serve "$MODEL" \
            --otlp-traces-endpoint "http://localhost:4318/v1/traces" \
            --kv-transfer-config "$KV_CFG" \
            --gpu-memory-utilization "$OPT_GPU_UTIL" \
            --no-enable-prefix-caching --enforce-eager --port "$PORT" \
            > "$CASE_DIR/vllm.log" 2>&1 &
        VLLM_PID=$!

        wait_server "http://localhost:$PORT" || { stop_vllm; stop_otlp; return 1; }
        "$@"
        stop_vllm; stop_otlp
        echo "  traces: $(ls -lh "$CASE_DIR"/*.jsonl 2>/dev/null | awk '{print $5,$9}' | tr '\n' '  ')"
    }

    run_vllm_case "seq_100"   run_aiperf_seq seq_100    100  1
    run_vllm_case "seq_1000"  run_aiperf_seq seq_1000  1000  1
    run_vllm_case "conc_5"    run_aiperf_seq conc_5     100  5
    run_vllm_case "conc_10"   run_aiperf_seq conc_10    100 10
    run_vllm_case "mt_10c_5t" run_aiperf_mt  mt_10c_5t  10  5
    run_vllm_case "mt_50c_5t" run_aiperf_mt  mt_50c_5t  50  5
fi

# ── kvbench phase ─────────────────────────────────────────────────────────────
if [[ "$OPT_PHASE" == "kvbench" || "$OPT_PHASE" == "both" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 2: kvbench  (calibrated, $OPT_GPU)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_kb_case() {
        local TAG="$1"; shift
        case_enabled "$TAG" || return 0
        echo ""
        echo "── kvbench | $TAG ──"
        local CASE_DIR="$OUT/$TAG"; mkdir -p "$CASE_DIR"
        clean_lmcache

        "$KB_VENV/bin/kvbench" serve \
            --calibrated-timing --gpu "$OPT_GPU" --model llama-3.2-1b \
            --lmcache-config "$LMCACHE_CFG" \
            --trace-file "$CASE_DIR/kvbench-trace.jsonl" \
            --port "$PORT" --type combined \
            > "$CASE_DIR/kvbench.log" 2>&1 &
        KB_PID=$!

        wait_server "http://localhost:$PORT" 60 || { stop_kb; return 1; }
        "$@"
        stop_kb
        echo "  trace: $(ls -lh "$CASE_DIR/kvbench-trace.jsonl" 2>/dev/null | awk '{print $5,$9}')"
    }

    run_kb_case "seq_100"   run_aiperf_seq seq_100    100  1
    run_kb_case "seq_1000"  run_aiperf_seq seq_1000  1000  1
    run_kb_case "conc_5"    run_aiperf_seq conc_5     100  5
    run_kb_case "conc_10"   run_aiperf_seq conc_10    100 10
    run_kb_case "mt_10c_5t" run_aiperf_mt  mt_10c_5t  10  5
    run_kb_case "mt_50c_5t" run_aiperf_mt  mt_50c_5t  50  5
fi

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating comparison report..."
"$KB_VENV/bin/python" "$REPO_DIR/scripts/compare_report.py" \
    --vllm-dir    "$OUT" \
    --kvbench-dir "$OUT" \
    --output      "$OUT/comparison_report.pdf" \
    2>&1 | tail -2

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done. Output:  $OUT"
echo "  Report:        $OUT/comparison_report.pdf"
echo "══════════════════════════════════════════════════════════"

#!/usr/bin/env bash
# Drive an already-running V1 server. A failed benchmark is a failed command.
set -euo pipefail
command -v genai-perf >/dev/null || { echo 'Install your chosen GenAI-Perf version first.' >&2; exit 1; }
exec genai-perf profile \
  -m "${KVBENCH_MODEL:-llama-3.2-1b}" \
  --service-kind openai --endpoint-type chat \
  --url "${KVBENCH_ENDPOINT:-http://localhost:8000}" --streaming \
  --synthetic-input-tokens-mean "${KVBENCH_INPUT_TOKENS:-2048}" \
  --output-tokens-mean "${KVBENCH_OUTPUT_TOKENS:-256}" \
  --concurrency "${KVBENCH_CONCURRENCY:-4}" \
  --request-count "${KVBENCH_REQUEST_COUNT:-100}" "$@"

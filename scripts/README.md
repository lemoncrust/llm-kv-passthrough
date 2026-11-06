# Benchmarking Scripts

Two scripts for running a head-to-head comparison between a real vLLM server
and a kvbench emulated server: a harness that drives both systems through
identical workloads, and a report generator that turns the resulting traces
into a PDF comparison.

## run_aiperf_harness.sh

Runs vLLM and kvbench sequentially against the same aiperf workloads, then
generates a comparison report from the collected traces.

### Requirements

- `HF_TOKEN` set in the environment (Hugging Face access token)
- A vLLM virtual environment (default: `../vllm/.venv` relative to repo root,
  or set `VLLM_VENV=/path/to/venv`)
- `aiperf` binary (default: `.venv/bin/aiperf`, or set `AIPERF=/path/to/aiperf`)
- An LMCache config file — use `examples/lmcache-local-disk.yaml` as a
  starting point and adjust the disk path and tier sizes for your setup

### Usage

```bash
export HF_TOKEN=<token>

bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    [OPTIONS] [OUTPUT_DIR]
```

If `OUTPUT_DIR` is omitted the script creates `traces/aiperf_run_<timestamp>/`.

### Options

| Option | Default | Description |
|---|---|---|
| `--cases CASE[,...]` | all | Cases to run: `seq_100` `seq_1000` `conc_5` `conc_10` `mt_10c_5t` `mt_50c_5t` |
| `--phase vllm\|kvbench\|both` | both | Which system(s) to run |
| `--gpu PROFILE` | RTX\_PRO\_6000\_BW | kvbench GPU profile for calibration |
| `--hardware LABEL` | RTX PRO 6000 Blackwell (GPU 0) | Hardware label in the PDF |
| `--model-name LABEL` | meta-llama/Llama-3.2-1B-Instruct | Model label in the PDF |
| `--lmcache-config FILE` | required | LMCache config file (see `examples/lmcache-local-disk.yaml`) |
| `--lmcache-cache DIR` | required | Directory where LMCache writes chunk files |
| `--gpu-util FLOAT` | 0.9 | vLLM `--gpu-memory-utilization` |

### Examples

```bash
# Sequential cases only
bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    --cases          seq_100,seq_1000

# kvbench phase only (vLLM traces already collected)
bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    --phase kvbench /path/to/existing/traces

# Different hardware
bash scripts/run_aiperf_harness.sh \
    --lmcache-config /path/to/lmcache.yaml \
    --lmcache-cache  /nvme/lmcache-cache \
    --gpu      H100_SXM \
    --hardware "NVIDIA H100 SXM" \
    --cases    seq_100,seq_1000,conc_5,conc_10
```

### Output layout

```
OUTPUT_DIR/
  seq_100/
    vllm-kv-trace.<pid>.jsonl    real vLLM KV spans
    vllm-request-trace.jsonl     OTLP request spans
    kvbench-trace.jsonl          emulated server spans
    aiperf/                      aiperf artifacts
  seq_1000/   ...
  comparison_<gpu_profile>.pdf   generated report
```

The report filename is derived from the GPU profile:
`RTX_PRO_6000_BW` produces `comparison_rtx_pro_6000_bw.pdf`.

## compare_report.py

Generates the PDF report from two trace directories. Can be run standalone
against any existing traces without re-running the servers.

### Usage

```bash
python scripts/compare_report.py \
    --vllm-dir    /path/to/traces \
    --kvbench-dir /path/to/traces \
    --output      report.pdf
```

Both `--vllm-dir` and `--kvbench-dir` point to the same directory when the
harness wrote both systems' traces together (the default setup).

### Options

| Option | Description |
|---|---|
| `--cases CASE[,...]` | Subset of cases to include in the report |
| `--output FILE` | Output PDF path (default: `comparison_<timestamp>.pdf` in current dir) |
| `--hardware LABEL` | Hardware string shown in the page header |
| `--model-name LABEL` | Model string shown in the page header |
| `--kvbench-gpu PROFILE` | GPU profile shown on the cover page |
| `--no-stddev` | Omit standard deviation from mean cells |
| `--name STEM` | Stem for the default output filename |

### Examples

```bash
# Sequential cases only
python scripts/compare_report.py \
    --vllm-dir    traces/aiperf_run_20261106 \
    --kvbench-dir traces/aiperf_run_20261106 \
    --output      seq_comparison.pdf \
    --cases       seq_100,seq_1000

# All cases with custom hardware label
python scripts/compare_report.py \
    --vllm-dir    traces/run_a \
    --kvbench-dir traces/run_b \
    --hardware    "NVIDIA H200 SXM" \
    --kvbench-gpu H200_SXM \
    --output      h200_comparison.pdf
```

### Metrics in the report

Each table row is one metric, summed per request then averaged across all
requests in that case. The three column groups show mean, p50, and p90 with
Δ = (kvbench - vLLM) / vLLM.

| Metric | vLLM source | kvbench source |
|---|---|---|
| Prefill | OTLP prefill minus kv.load\_init and kv.store | `prefill.compute` span |
| Decode per token | OTLP decode / completion\_tokens | `decode.step` spans |
| kv.lookup | `kv.lookup` span | `kv.lookup` span |
| kv.store | `kv.store` span | `kv.store` span |
| gpu.copy\_out | `gpu.copy_out` spans summed | `gpu.copy_out` spans summed |
| gpu.alloc | not present | `gpu.alloc` span (emulator artifact) |
| io.write | `io.write` spans summed | `io.write` spans summed |
| kv.retrieve | `kv.load_init` span | `rank.retrieve` span |
| gpu.copy\_in | `gpu.copy_in` span | `gpu.copy_in` span |
| io.read | `io.read` spans summed | `io.read` spans summed |

Prefill on the vLLM side subtracts kv.load\_init and kv.store from the OTLP
prefill window so that KV storage work is not charged to GPU compute.

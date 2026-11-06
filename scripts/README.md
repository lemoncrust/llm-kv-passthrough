# Benchmarking Scripts

Three scripts for comparing a real vLLM server against a kvbench emulated server:

- `run_aiperf_harness.sh` runs both servers through identical workloads and collects traces
- `compare_report.py` turns the traces into a PDF comparison report
- `compare_table.py` prints the same numbers in the terminal without generating a PDF

---

## Step 1: Collect traces with the harness

The harness starts vLLM, runs aiperf against it, then does the same with kvbench.
Both servers use the same LMCache configuration so the storage layer is identical.

### What you need before running

| Requirement | How to provide |
|---|---|
| Hugging Face token | `export HF_TOKEN=<token>` |
| LMCache config file | Pass with `--lmcache-config`. Use `examples/lmcache-local-disk.yaml` as a starting point. |
| Cache directory | Pass with `--lmcache-cache`. Point it at a fast local disk. |
| vLLM virtual environment | Default: `../vllm/.venv`. Override with `VLLM_VENV=/path/to/venv`. |
| aiperf binary | Default: `.venv/bin/aiperf`. Override with `AIPERF=/path/to/aiperf`. |

### Basic usage

```bash
export HF_TOKEN=<token>

bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    /path/to/output
```

If you omit the output directory the script creates `traces/aiperf_run_<timestamp>/`.

### All options

| Option | Default | Description |
|---|---|---|
| `--lmcache-config FILE` | required | LMCache config file |
| `--lmcache-cache DIR` | required | Directory where LMCache writes chunk files |
| `--cases CASE[,...]` | all | Cases to run. Available: `seq_100` `seq_1000` `conc_5` `conc_10` `mt_10c_5t` `mt_50c_5t` |
| `--phase vllm/kvbench/both` | both | Run only vLLM, only kvbench, or both |
| `--gpu PROFILE` | RTX\_PRO\_6000\_BW | kvbench GPU profile used for timing calibration |
| `--hardware LABEL` | RTX PRO 6000 Blackwell (GPU 0) | Hardware label shown in the PDF |
| `--model-name LABEL` | meta-llama/Llama-3.2-1B-Instruct | Model label shown in the PDF |
| `--gpu-util FLOAT` | 0.9 | vLLM `--gpu-memory-utilization` |

### Examples

```bash
# Sequential cases only
bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    --cases          seq_100,seq_1000

# Run only the kvbench phase against vLLM traces you already collected
bash scripts/run_aiperf_harness.sh \
    --lmcache-config examples/lmcache-local-disk.yaml \
    --lmcache-cache  /tmp/lmcache-cache \
    --phase kvbench \
    /path/to/existing/traces

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
  seq_1000/
    ...
  comparison_<gpu_profile>.pdf   generated report
```

The report filename is derived from the GPU profile. For example,
`RTX_PRO_6000_BW` produces `comparison_rtx_pro_6000_bw.pdf`.

---

## Step 2: Generate a PDF report

The harness generates the report automatically at the end. If you want to
regenerate it from existing traces, run compare_report.py directly.

```bash
python scripts/compare_report.py \
    --vllm-dir    /path/to/traces \
    --kvbench-dir /path/to/traces \
    --output      report.pdf
```

Both `--vllm-dir` and `--kvbench-dir` can point to the same directory when
the harness wrote both systems' traces together, which is the default.

### Options

| Option | Description |
|---|---|
| `--cases CASE[,...]` | Only include these cases in the report (default: all found) |
| `--output FILE` | Output PDF path (default: `comparison_<timestamp>.pdf` in current directory) |
| `--hardware LABEL` | Hardware string shown in the page header |
| `--model-name LABEL` | Model string shown in the page header |
| `--kvbench-gpu PROFILE` | GPU profile shown on the cover page |
| `--no-stddev` | Omit standard deviation from mean cells |

### Example

```bash
# Sequential cases only, with custom labels
python scripts/compare_report.py \
    --vllm-dir    traces/aiperf_run_20261106 \
    --kvbench-dir traces/aiperf_run_20261106 \
    --output      seq_comparison.pdf \
    --cases       seq_100,seq_1000 \
    --hardware    "NVIDIA H100 SXM" \
    --kvbench-gpu H100_SXM
```

---

## Step 3: Inspect numbers in the terminal

compare_table.py prints the same metrics as the PDF report but in plain text,
which is useful for quick checks without generating a file.

```bash
python scripts/compare_table.py /path/to/traces
```

It automatically finds every case sub-directory that contains traces and
shows them one after another.

### Options

| Option | Description |
|---|---|
| `--cases CASE[,...]` | Only show these cases |
| `--kvbench-dir DIR` | Separate directory for kvbench traces (default: same as run dir) |

### Examples

```bash
# Show all cases in a run directory
python scripts/compare_table.py traces/aiperf_run_20261106

# Show only sequential cases
python scripts/compare_table.py traces/aiperf_run_20261106 --cases seq_100,seq_1000

# vLLM and kvbench traces in separate directories
python scripts/compare_table.py /path/to/vllm-traces \
    --kvbench-dir /path/to/kvbench-traces
```

---

## What the metrics mean

Each row in the table or PDF is one metric, summed per request and then
averaged across all requests in that case.

| Metric | vLLM source | kvbench source |
|---|---|---|
| Prefill | OTLP prefill minus kv.load\_init and kv.store | `prefill.compute` span |
| Decode per token | OTLP decode / completion\_tokens | `decode.step` spans |
| kv.lookup | `kv.lookup` span | `kv.lookup` span |
| kv.store | `kv.store` span | `kv.store` span |
| gpu.copy\_out | `gpu.copy_out` spans summed per request | `gpu.copy_out` spans summed |
| gpu.alloc | not present in real vLLM | `gpu.alloc` span (emulator artifact) |
| io.write | `io.write` spans summed per request | `io.write` spans summed |
| kv.retrieve | `kv.load_init` span | `rank.retrieve` span |
| gpu.copy\_in | `gpu.copy_in` span | `gpu.copy_in` span |
| io.read | `io.read` spans summed per request | `io.read` spans summed |

The delta column shows `(kvbench - vLLM) / vLLM`. Green means kvbench
over-predicts. Red means kvbench under-predicts. No colour means within 5%.

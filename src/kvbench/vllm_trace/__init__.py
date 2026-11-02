"""
KV I/O and step tracing for a real vLLM server.

``TracingLMCacheConnector`` (``kvbench.vllm_trace.connector``) is a drop-in
vLLM KV connector that reuses kvbench's LMCache hooks, so a ``vllm serve``
run produces span JSONL that ``kvbench trace2fio`` already understands.
Importing this package does not import vLLM.
"""

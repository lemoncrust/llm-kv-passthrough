import pytest

from kvbench.benchmark.blocktrace import summarize_issues, validate_trace_log


def test_issue_order_sequential_runs_and_interleaving(tmp_path):
    path = tmp_path / "issues.txt"
    path.write_text("""1.0 D WS 100 8
1.1 D WS 108 8
1.2 C WS 100 8
1.3 D R 5 8
1.4 D WS 10000 16
1.5 D W 116 8
1.6 D WS 124 8
1.7 D WF 0 0
""")
    result = summarize_issues(path)
    assert result["write_ops"] == 5
    assert result["write_bytes"] == 48 * 512
    assert result["adjacent_write_transitions"] == 2
    assert result["sequential_transition_fraction"] == 0.5
    assert result["backward_write_transitions"] == 1
    assert result["max_contiguous_run_bytes"] == 8192


def test_dropped_or_failed_trace_is_not_a_valid_lba_sample():
    validate_trace_log("Total: 123 events (dropped 0)")
    for text in ("Total: 123 events (dropped 2)", "FAILED to start thread", "no drop accounting"):
        with pytest.raises(RuntimeError):
            validate_trace_log(text)

"""The CLI must select explicit rates and reject misleading legacy controls."""

import os

import pytest
from typer.testing import CliRunner

from kvbench.cli.main import app
from kvbench.servers.app import load_config


def test_help_prioritizes_supported_v1_controls():
    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--prefill-tps" in result.stdout and "--decode-tps" in result.stdout
    assert "--gpu-batching" not in result.stdout


@pytest.mark.parametrize("flags", [["--simple-timing"], ["--roofline-timing"],
                                   ["--no-gpu-batching"], ["--eff-bw", "0.9"]])
def test_legacy_flags_fail_even_when_value_equals_old_default(flags):
    result = CliRunner().invoke(app, ["serve", *flags])
    assert result.exit_code == 2
    assert "Legacy timing flags" in result.output


def test_cli_rates_reach_worker_configuration(monkeypatch):
    import uvicorn

    for name in list(os.environ):
        if name.startswith("KVBENCH_"):
            monkeypatch.delenv(name)
    for name in ("KVBENCH_CONFIG_FILE", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    captured = []

    def run(*args, **kwargs):  # noqa: ARG001 - uvicorn boundary spy
        captured.append(load_config())

    monkeypatch.setattr(uvicorn, "run", run)
    result = CliRunner().invoke(app, ["serve", "--prefill-tps", "4321", "--decode-tps", "123"])
    assert result.exit_code == 0, result.output
    assert captured[0].scheduler.prefill_tokens_per_second == 4321
    assert captured[0].scheduler.decode_tokens_per_second == 123

"""Deployment examples render supported configurations, not historical modes."""

import json
from pathlib import Path

import pytest
import yaml

from kvbench.core.config import KVBenchConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("example", ["rate-v1.yaml", "aiperf-validation.yaml"])
def test_example_run_configuration(example):
    cfg = KVBenchConfig.from_yaml(ROOT / "examples" / example)
    cfg.validate_serving()
    assert cfg.kv.lmcache_config_file.is_absolute()
    assert cfg.scheduler.prefill_tokens_per_second == 10000


def test_ansible_templates_render_valid_rate_config(tmp_path):
    jinja = pytest.importorskip("jinja2")
    variables = yaml.safe_load((ROOT / "deployment/ansible/inventory/example.yml").read_text())["all"]["vars"]
    variables.update(inventory_hostname="test-host", kvbench_config_dir=str(tmp_path),
                     ansible_date_time={"iso8601": "2026-09-29T00:00:00"})
    environment = jinja.Environment(undefined=jinja.StrictUndefined)
    environment.filters["to_json"] = json.dumps
    folder = ROOT / "deployment/ansible/templates"
    for name in ("lmcache", "config"):
        rendered = environment.from_string((folder / f"{name}.yaml.j2").read_text()).render(**variables)
        (tmp_path / f"{name}.yaml").write_text(rendered)
    cfg = KVBenchConfig.from_yaml(tmp_path / "config.yaml")
    cfg.validate_serving()
    service = environment.from_string((folder / "kvbench.service.j2").read_text()).render(**variables)
    assert "MemoryMax=2G" in service
    assert "kvbench_storage" not in service


@pytest.mark.parametrize("filename", ["docker-compose.yml", "docker-compose.distributed.yml"])
def test_compose_engines_use_supported_configuration(filename, monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("KVBENCH_"):
            monkeypatch.delenv(key)
    data = yaml.safe_load((ROOT / "deployment/docker" / filename).read_text())
    engines = 0
    for service in data["services"].values():
        env = service.get("environment", {})
        if not any(key.startswith("KVBENCH_") for key in env):
            continue
        engines += 1
        with monkeypatch.context() as context:
            for key, value in env.items():
                context.setenv(key, str(value))
            KVBenchConfig.from_env().validate_serving()
        assert service["mem_limit"] == "2g"
    assert engines == (2 if "distributed" in filename else 1)

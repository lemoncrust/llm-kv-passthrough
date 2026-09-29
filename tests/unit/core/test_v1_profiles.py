"""Declarative sizing and prefix identity contracts for V1."""

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.core.models import get_model_profile
from kvbench.core.synthetic import SyntheticTokenizer
from kvbench.servers.openai_compat import ChatMessage


def test_catalog_override_is_scoped_and_resolved_relative_to_config(tmp_path):
    (tmp_path / "catalog.yaml").write_text("""version: 1
models:
  custom:
    name: Test model
    layers: 4
    hidden: 128
    kv_heads: 2
    num_heads: 2
    intermediate: 256
    kv_dtype: fp8
""")
    (tmp_path / "run.yaml").write_text("profiles_file: catalog.yaml\nserver:\n  model_profile: custom\n")
    cfg = KVBenchConfig.from_yaml(tmp_path / "run.yaml")
    assert cfg.model.total_kv_cache_bytes_per_token == 1024
    assert cfg.profiles_file.is_absolute()
    with pytest.raises(KeyError):
        get_model_profile("custom")
    cfg.to_yaml(tmp_path / "resolved.yaml")
    assert KVBenchConfig.from_yaml(tmp_path / "resolved.yaml").model == cfg.model


@pytest.mark.parametrize("bad", [{"typo": 1}, {"storage": {"backend_type": "redis"}},
                                  {"scheduler": {"decode_tokens_per_second": -1}},
                                  {"scheduler": {"decode_tokens_per_second": float("nan")}},
                                  {"gpu": {"tp_size": 3}}])
def test_invalid_config_fails(bad):
    with pytest.raises(ValueError):
        KVBenchConfig.model_validate(bad)


def test_rank_partition_and_replication():
    m = get_model_profile("llama-3.1-8b")
    baseline = m.total_kv_cache_bytes_per_token
    layout = m.rank_layout(4, 2)
    assert len(layout) == 8
    assert sum(r["bytes_per_token"] for r in layout) == baseline
    assert all(r["bytes_per_token"] == baseline // 8 for r in layout)
    assert sum(r["bytes_per_token"] for r in m.rank_layout(16)) == 2 * baseline
    mla = get_model_profile("deepseek-v3.2")
    assert sum(r["bytes_per_token"] for r in mla.rank_layout(8)) == 8 * mla.total_kv_cache_bytes_per_token


def test_kv_precision_is_independent_of_weight_precision():
    m = get_model_profile("gpt-oss-120b")
    assert m.bytes_per_element == 1
    assert m.kv_bytes_per_element == 2
    assert m.total_kv_cache_bytes_per_token == 36 * 2 * 8 * 64 * 2


@pytest.mark.parametrize("ratio", [1.1, 3.84, 4, 4.1, 4.4, 7.8, 8])
def test_generated_output_roundtrips_into_next_turn(ratio):
    tokenizer = SyntheticTokenizer(ratio)
    messages = [ChatMessage(role="user", content="hello 世界 " * 8)]
    prompt = tokenizer.prompt(messages)
    ids = tokenizer.encode(prompt)
    content = ""
    for _ in range(50):
        text, token = tokenizer.sample(len(ids))
        ids.append(token)
        content += text
    following = tokenizer.prompt([*messages, ChatMessage(role="assistant", content=content),
                                  ChatMessage(role="user", content="continue")])
    assert tokenizer.encode(following)[:len(ids)] == ids
    assert len(ids) == pytest.approx(len((prompt + content).encode()) / ratio, abs=1)

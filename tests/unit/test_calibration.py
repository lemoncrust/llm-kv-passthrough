"""Unit tests for the calibrated timing strategy."""

from __future__ import annotations

import json

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.timing import create_timing_strategy
from kvbench.timing.calibration import (
    BUILTIN_ANCHORS,
    CalibratedTimingStrategy,
    TimingAnchor,
    load_anchors,
)
from kvbench.timing.roofline import RooflineTimingStrategy


def _backbone(gpu: str, model: str, tp: int = 1) -> RooflineTimingStrategy:
    return RooflineTimingStrategy(
        gpu=gpu,
        model=model,
        tp_size=tp,
        efficiency=1.0,
        include_tp_communication=tp > 1,
        include_pp_communication=False,
    )


def test_exact_anchor_scales_to_measurement() -> None:
    """With an exact (model, gpu) anchor, predictions at the anchor's shape
    reproduce the measured values."""
    anchor = TimingAnchor(
        model="llama-3.2-1b", gpu="GB10", isl=2048, osl=64,
        ttft_ms=79.6, tpot_ms=12.5,
    )
    strat = CalibratedTimingStrategy(
        gpu="GB10", model="llama-3.2-1b",
        include_tp_communication=False, include_pp_communication=False,
        anchors=[anchor],
    )
    assert strat.prefill_latency(2048).total_ms == pytest.approx(79.6, rel=0.01)
    assert strat.decode_latency(2048).total_ms == pytest.approx(12.5, rel=0.01)


def test_scales_are_at_least_one_for_real_anchors() -> None:
    """Measured latency can't beat the theoretical envelope: builtin
    experiment anchors must produce scales >= 1."""
    strat = CalibratedTimingStrategy(gpu="GB10", model="llama-3.2-1b")
    assert strat.prefill_scale >= 1.0
    assert strat.decode_scale >= 1.0


def test_gpu_tier_beats_model_tier() -> None:
    """A same-gpu anchor is preferred over a same-model anchor."""
    gpu_anchor = TimingAnchor(
        model="llama-3.2-3b", gpu="GB10", isl=1024, osl=64, tpot_ms=20.0,
    )
    model_anchor = TimingAnchor(
        model="llama-3.2-1b", gpu="H100_SXM", isl=1024, osl=64, tpot_ms=5.0,
    )
    strat = CalibratedTimingStrategy(
        gpu="GB10", model="llama-3.2-1b",
        include_tp_communication=False, include_pp_communication=False,
        anchors=[gpu_anchor, model_anchor],
    )
    expected = 20.0 / _backbone("GB10", "llama-3.2-3b").decode_latency(1024).total_ms
    assert strat.decode_scale == pytest.approx(expected, rel=1e-6)


def test_unknown_profile_anchor_is_skipped() -> None:
    bad = TimingAnchor(model="no-such-model", gpu="GB10", isl=64, osl=8, tpot_ms=1.0)
    good = TimingAnchor(model="llama-3.2-1b", gpu="GB10", isl=64, osl=8, tpot_ms=15.0)
    strat = CalibratedTimingStrategy(
        gpu="GB10", model="llama-3.2-1b",
        include_tp_communication=False, include_pp_communication=False,
        anchors=[bad, good],
    )
    expected = 15.0 / _backbone("GB10", "llama-3.2-1b").decode_latency(64).total_ms
    assert strat.decode_scale == pytest.approx(expected, rel=1e-6)


def test_no_anchor_fallback_is_default_efficiency() -> None:
    strat = CalibratedTimingStrategy(
        gpu="H200_SXM", model="llama-3.1-8b",
        include_tp_communication=False, include_pp_communication=False,
        anchors=[],
    )
    assert strat.prefill_scale == pytest.approx(1.0 / 0.7)
    assert strat.decode_scale == pytest.approx(1.0 / 0.7)


def test_load_anchors_json(tmp_path) -> None:
    f = tmp_path / "anchors.json"
    f.write_text(json.dumps([
        {"model": "llama-3.1-8b", "gpu": "H100_SXM", "isl": 4096, "osl": 128,
         "ttft_ms": 300.0, "tpot_ms": 9.0, "source": "experiment",
         "citation": "test"},
    ]))
    anchors = load_anchors(f)
    assert len(anchors) == 1
    assert anchors[0].gpu == "H100_SXM"
    assert anchors[0].tpot_ms == 9.0


def test_builtin_anchors_all_usable() -> None:
    """Every builtin anchor references resolvable profiles."""
    strat = CalibratedTimingStrategy(gpu="B200_SXM", model="gpt-oss-120b")
    assert len(strat._scored) == len(BUILTIN_ANCHORS)


def test_factory_creates_calibrated_strategy() -> None:
    cfg = KVBenchConfig()
    cfg = cfg.model_copy(update={"timing": cfg.timing.model_copy(update={"calibrated": True})})
    strat = create_timing_strategy(cfg)
    assert isinstance(strat, CalibratedTimingStrategy)

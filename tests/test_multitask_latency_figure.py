"""Keep the 80-input ratio panel distinct from the matched ten-input timing panel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def latency_renderer(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("latency_panels", ROOT / "scripts/render_multitask_latency.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_panels_use_their_own_frozen_samples(latency_renderer) -> None:
    scaling, matched = latency_renderer.load_panels()
    assert np.array_equal(scaling.k, [1, 2, 4, 8, 16])
    assert np.allclose(scaling.mean_ms.iloc[[0, -1]], [68.6316994625, 73.3366664375])
    assert np.isclose(scaling.increase_pct.iloc[-1], 6.855384628164085)
    assert np.allclose(scaling.iloc[-1][["ci_low_pct", "ci_high_pct"]], [5.550336850276594, 7.801510280055779])
    assert np.allclose(matched.shared_ms.iloc[[0, -1]], [61.833902, 62.6979777])
    assert np.isclose(matched.repeated_ms.iloc[-1], 956.7061062999999)
    assert not np.isclose(scaling.mean_ms.iloc[-1], matched.shared_ms.iloc[-1])


def test_latency_axes_are_equal_sized_linear_and_have_no_inset(latency_renderer) -> None:
    scaling, matched = latency_renderer.load_panels()
    figure = latency_renderer.make_figure(scaling, matched)
    assert len(figure.axes) == 2
    left, right = figure.axes
    assert np.isclose(left.get_position().width, right.get_position().width)
    assert np.isclose(left.get_position().height, right.get_position().height)
    for axis in figure.axes:
        assert axis.get_xscale() == axis.get_yscale() == "linear"
        assert np.array_equal(axis.lines[0].get_xdata(), [1, 2, 4, 8, 16])
        assert not axis.child_axes
    assert "n=80" in left.get_title(loc="left")
    assert "n=10" in right.get_title(loc="left")
    assert all("%" in label.get_text() for label in left.get_yticklabels())
    assert "(%)" in left.get_ylabel()
    assert right.get_ylim()[0] == 0
    latency_renderer.plt.close(figure)


def test_depth_text_distinguishes_midpoint_from_window_coverage() -> None:
    source = (ROOT / "research/multitask-activation-benchmark/paper/DRAFT.md").read_text()
    text = " ".join(source.split())
    assert "window midpoint divided by the final output index, 24" in text
    assert "midpoints range from output 3 to 19" in text
    assert "windows themselves span output indices 2--20" in text
    assert "25 hidden-state outputs, indexed 0--24" in text

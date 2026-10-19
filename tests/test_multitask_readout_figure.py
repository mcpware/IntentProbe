"""Check that the paper figure encodes selected manifest windows, not performance."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def readout_renderer(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("readout_figure", ROOT / "scripts/render_multitask_readouts.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readout_matrix_matches_every_selected_window(readout_renderer) -> None:
    registry = json.loads(readout_renderer.REGISTRY.read_text())
    heads = readout_renderer.readout_frame(registry)
    matrix = readout_renderer.selection_matrix(heads)
    assert matrix.shape == (16, 25)
    assert not matrix[:, 24].any()
    assert np.array_equal(np.flatnonzero(matrix[0]), np.array([18, 19, 20]))
    assert np.array_equal(np.flatnonzero(matrix[1]), np.array([5, 6, 7]))
    assert np.array_equal(np.count_nonzero(matrix, axis=1), heads.width)
    assert set(np.unique(matrix)) == {0, 1, 2}


def test_readout_manifest_rejects_malformed_windows(readout_renderer) -> None:
    registry = json.loads(readout_renderer.REGISTRY.read_text())
    invalid = deepcopy(registry)
    invalid["heads"][0]["end_layer"] = 25
    with pytest.raises(AssertionError):
        readout_renderer.readout_frame(invalid)
    invalid = deepcopy(registry)
    invalid["heads"][0]["width"] = 2
    with pytest.raises(AssertionError):
        readout_renderer.readout_frame(invalid)


def test_readout_figure_has_accessible_pooling_and_output_labels(readout_renderer) -> None:
    registry = json.loads(readout_renderer.REGISTRY.read_text())
    heads = readout_renderer.readout_frame(registry)
    figure = readout_renderer.make_figure(heads)
    axes = figure.axes[0]
    assert "24 = final" in axes.get_xlabel()
    assert "outputs 2–20" in axes.get_title(loc="left")
    assert len(axes.get_yticklabels()) == 16
    assert len(axes.child_axes[0].get_yticklabels()) == 16
    readout_renderer.plt.close(figure)

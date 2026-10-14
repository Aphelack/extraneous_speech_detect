"""Unit tests for metrics computation and 50 ms boundary tolerance collar."""

import numpy as np
import pytest

from src.config import COLLAR_FRAMES, FRAME_DURATION_SEC
from src.metrics import (
    compute_boundary_collar_mask,
    cosine_similarity,
    evaluate_channels,
    frame_mask_to_intervals,
    intervals_to_frame_matrix,
    precision_recall_f1,
)


def test_boundary_collar_forgives_small_offsets():
    """Verify that prediction boundary jitter within 50 ms (5 frames) is forgiven."""
    num_frames = 300  # 3.0 seconds at 10 ms grid
    # Ground truth segment: from 1.00s to 2.00s (frames 100 to 200)
    gt_intervals = [{"start": 1.00, "end": 2.00, "label": "own"}]
    gt_masks = intervals_to_frame_matrix(gt_intervals, num_frames=num_frames, frame_sec=0.010)
    y_true = gt_masks["own"]

    # 1. Prediction starts 30 ms earlier (0.97s, frame 97) and ends 30 ms later (2.03s, frame 203)
    # The offset is 3 frames (30 ms), which is strictly within 5 frames (50 ms collar).
    pred_intervals_within = [{"start": 0.97, "end": 2.03, "label": "own"}]
    pred_masks_within = intervals_to_frame_matrix(pred_intervals_within, num_frames=num_frames, frame_sec=0.010)
    y_pred_within = pred_masks_within["own"]

    # In 'forgive' mode with collar=5 (50 ms), the 30 ms discrepancy at boundaries must not penalize F1
    res_within = precision_recall_f1(y_true, y_pred_within, collar_frames=5, mode="forgive")
    assert res_within["f1"] == 1.0, f"Expected perfect F1 within collar, got: {res_within}"
    assert res_within["precision"] == 1.0
    assert res_within["recall"] == 1.0
    assert res_within["fp"] == 0
    assert res_within["fn"] == 0

    # 2. Without collar (collar_frames=0), these 30 ms boundary errors MUST lower precision
    res_no_collar = precision_recall_f1(y_true, y_pred_within, collar_frames=0)
    assert res_no_collar["precision"] < 1.0, "Without collar, boundary discrepancy must count as false positive"
    assert res_no_collar["fp"] > 0

    # 3. If offset exceeds 50 ms (e.g. 100 ms jitter: starts at 0.90s, ends at 2.10s)
    pred_intervals_outside = [{"start": 0.90, "end": 2.10, "label": "own"}]
    pred_masks_outside = intervals_to_frame_matrix(pred_intervals_outside, num_frames=num_frames, frame_sec=0.010)
    y_pred_outside = pred_masks_outside["own"]

    res_outside = precision_recall_f1(y_true, y_pred_outside, collar_frames=5, mode="forgive")
    assert res_outside["precision"] < 1.0, "Exceeding 50 ms collar must penalize precision"
    assert res_outside["fp"] > 0


def test_boundary_collar_mask_shape_and_indices():
    """Verify compute_boundary_collar_mask marks exact collar radius around transitions."""
    y = np.zeros(50, dtype=bool)
    # Segment from frame 20 to 30
    y[20:30] = True

    collar = compute_boundary_collar_mask(y, collar_frames=5)
    assert len(collar) == 50
    # Boundary 1 is at 20 -> frames 15..25 must be in collar
    assert np.all(collar[15:26])
    # Boundary 2 is at 30 -> frames 25..35 must be in collar
    assert np.all(collar[25:36])
    # Distant frames must not be in collar
    assert not collar[5]
    assert not collar[45]


def test_intervals_conversion_roundtrip():
    """Verify converting intervals to binary mask and back preserves intervals."""
    num_frames = 200
    orig_intervals = [
        {"start": 0.50, "end": 1.00, "label": "own"},
        {"start": 1.20, "end": 1.70, "label": "own"},
    ]
    masks = intervals_to_frame_matrix(orig_intervals, num_frames=num_frames, frame_sec=0.010)
    reconstructed = frame_mask_to_intervals(masks["own"], label="own", frame_sec=0.010)

    assert len(reconstructed) == 2
    assert abs(reconstructed[0]["start"] - 0.50) < 1e-4
    assert abs(reconstructed[0]["end"] - 1.00) < 1e-4
    assert abs(reconstructed[1]["start"] - 1.20) < 1e-4
    assert abs(reconstructed[1]["end"] - 1.70) < 1e-4


def test_cosine_similarity():
    """Verify speaker embedding cosine similarity math."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    v4 = np.array([-1.0, 0.0, 0.0])

    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert pytest.approx(cosine_similarity(v1, v4)) == -1.0


def test_evaluate_channels():
    """Verify multi-channel multi-label evaluation output structure."""
    num_frames = 100
    gt = {
        "A": [{"start": 0.10, "end": 0.40, "label": "own"}],
        "B": [{"start": 0.30, "end": 0.80, "label": "own"}],
    }
    pred = {
        "A": [{"start": 0.10, "end": 0.40, "label": "own"}],
        "B": [{"start": 0.30, "end": 0.80, "label": "own"}],
    }
    metrics = evaluate_channels(gt, pred, num_frames=num_frames)
    assert "macro_f1" in metrics
    assert metrics["macro_f1"] == 1.0
    assert metrics["by_channel"]["A"]["own"]["f1"] == 1.0
    assert metrics["by_channel"]["B"]["own"]["f1"] == 1.0


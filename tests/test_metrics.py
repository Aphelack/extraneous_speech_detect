"""Unit tests for metrics computation and 50 ms boundary tolerance collar."""

import numpy as np
import pytest

from src.config import COLLAR_FRAMES, FRAME_DURATION_SEC
from src.metrics import (
    FrameCountAccumulator,
    compute_boundary_collar_mask,
    cosine_similarity,
    count_frame_errors,
    evaluate_channels,
    f1_from_counts,
    frame_mask_to_intervals,
    intervals_to_frame_matrix,
    precision_recall_f1,
)


def test_boundary_collar_excludes_frames_entirely():
    """Frames within 50 ms of a ground-truth boundary are excluded from TP/FP/FN."""
    num_frames = 300  # 3.0 seconds at 10 ms grid
    gt_intervals = [{"start": 1.00, "end": 2.00, "label": "own"}]
    gt_masks = intervals_to_frame_matrix(gt_intervals, num_frames=num_frames, frame_sec=0.010)
    y_true = gt_masks["own"]

    # Prediction is 30 ms wider on both sides: the extra frames lie inside the
    # collar, so they must not be counted as FP at all.
    pred_intervals_within = [{"start": 0.97, "end": 2.03, "label": "own"}]
    pred_masks_within = intervals_to_frame_matrix(
        pred_intervals_within, num_frames=num_frames, frame_sec=0.010
    )
    y_pred_within = pred_masks_within["own"]

    res_within = precision_recall_f1(y_true, y_pred_within, collar_frames=5)
    assert res_within["f1"] == 1.0, f"Expected perfect F1 within collar, got: {res_within}"
    assert res_within["fp"] == 0
    assert res_within["fn"] == 0

    # Without collar the same boundary discrepancy must be penalized.
    res_no_collar = precision_recall_f1(y_true, y_pred_within, collar_frames=0)
    assert res_no_collar["precision"] < 1.0
    assert res_no_collar["fp"] > 0

    # A 100 ms offset extends beyond the collar and must be penalized.
    pred_intervals_outside = [{"start": 0.90, "end": 2.10, "label": "own"}]
    pred_masks_outside = intervals_to_frame_matrix(
        pred_intervals_outside, num_frames=num_frames, frame_sec=0.010
    )
    res_outside = precision_recall_f1(y_true, pred_masks_outside["own"], collar_frames=5)
    assert res_outside["precision"] < 1.0
    assert res_outside["fp"] > 0


def test_collar_exclusion_removes_tp_inside_collar():
    """TP inside the collar must NOT be counted (strict exclusion, not 'forgive')."""
    y_true = np.zeros(100, dtype=bool)
    y_true[20:60] = True
    y_pred = np.zeros(100, dtype=bool)
    y_pred[20:60] = True

    tp, fp, fn = count_frame_errors(y_true, y_pred, collar_frames=5)
    # Segment spans frames 20..59 (40 frames). The collar removes frames
    # 20..25 and 55..59 (11 frames inside the segment), leaving 29 TP.
    assert tp == 29
    assert fp == 0
    assert fn == 0

    # The same frames must be excluded from FP/FN as well.
    y_pred_shifted = np.zeros(100, dtype=bool)
    y_pred_shifted[20:60] = True
    y_pred_shifted[60:63] = True  # 3 frames just past the segment end
    tp2, fp2, fn2 = count_frame_errors(y_true, y_pred_shifted, collar_frames=5)
    assert tp2 == 29
    assert fp2 == 0, "Frames inside the collar must not be counted as FP"
    assert fn2 == 0


def test_global_micro_f1_does_not_inflate_absent_classes():
    """A class absent from most records must not get F1=1.0 by per-record averaging."""
    acc = FrameCountAccumulator()

    # 9 records with no `third` at all and no third predictions.
    for _ in range(9):
        gt = {"A": [{"start": 0.0, "end": 1.0, "label": "own"}], "B": []}
        pred = {"A": [{"start": 0.0, "end": 1.0, "label": "own"}], "B": []}
        acc.add(gt, pred, num_frames=200)

    # 1 record where `third` exists but is completely missed.
    gt = {"A": [{"start": 0.0, "end": 1.0, "label": "third"}], "B": []}
    pred = {"A": [], "B": []}
    acc.add(gt, pred, num_frames=200)

    result = acc.finalize()
    # Per-record averaging would give (9 * 1.0 + 0.0) / 10 = 0.9.
    assert result["by_label"]["third"]["f1"] == 0.0
    assert result["by_label"]["third"]["fn"] > 0


def test_f1_from_counts_zero_division():
    """Empty counts must yield 0.0, not a spurious perfect score."""
    assert f1_from_counts(0, 0, 0)["f1"] == 0.0
    assert f1_from_counts(0, 5, 0)["f1"] == 0.0
    assert f1_from_counts(0, 0, 5)["f1"] == 0.0


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
    # `own` is perfect; `leak` and `third` have no positives and no
    # predictions, so their F1 is 0.0 (not a spurious 1.0).
    assert metrics["by_label"]["own"]["f1"] == 1.0
    assert metrics["by_label"]["leak"]["f1"] == 0.0
    assert metrics["by_label"]["third"]["f1"] == 0.0
    assert metrics["by_channel"]["A"]["own"]["tp"] > 0
    assert metrics["by_channel"]["B"]["own"]["tp"] > 0


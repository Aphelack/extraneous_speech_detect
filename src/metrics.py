"""Metrics module for extraneous speech detection.

Includes:
- Conversion between interval JSON format and 10 ms frame-level binary masks
- Frame-level Precision, Recall, and F1 with boundary tolerance (collar = 50 ms)
- Multi-label, multi-channel evaluation
- Cosine similarity for speaker embeddings
- Single-thread Real-Time Factor (RTF) benchmarking
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.config import (
    COLLAR_FRAMES,
    COLLAR_SEC,
    EVAL_LABELS,
    FRAME_DURATION_SEC,
    MAX_RTF_LIMIT,
)


def intervals_to_frame_matrix(
    intervals: Sequence[Dict[str, Any]],
    num_frames: int,
    frame_sec: float = FRAME_DURATION_SEC,
    labels: Sequence[str] = EVAL_LABELS,
) -> Dict[str, np.ndarray]:
    """Convert interval list to binary frame masks (shape [num_frames] for each label).

    Parameters
    ----------
    intervals : list of dict
        Each dict has 'start', 'end', and 'label'.
    num_frames : int
        Total number of frames in the recording.
    frame_sec : float
        Frame duration in seconds (default 0.010 = 10 ms).
    labels : sequence of str
        Labels to extract.

    Returns
    -------
    dict of {label: np.ndarray(bool)}
    """
    masks = {lbl: np.zeros(num_frames, dtype=bool) for lbl in labels}

    for seg in intervals:
        lbl = seg.get("label")
        if lbl not in masks:
            continue
        start_sec = max(0.0, float(seg["start"]))
        end_sec = max(start_sec, float(seg["end"]))

        # Frame indices: frame i spans [i * frame_sec, (i + 1) * frame_sec)
        # We activate frame i if its midpoint falls within [start_sec, end_sec)
        start_frame = max(0, int(math.floor(start_sec / frame_sec)))
        end_frame = min(num_frames, int(math.ceil(end_sec / frame_sec)))

        for f in range(start_frame, end_frame):
            frame_mid = (f + 0.5) * frame_sec
            if start_sec <= frame_mid < end_sec:
                masks[lbl][f] = True

    return masks


def frame_mask_to_intervals(
    mask: np.ndarray,
    label: str,
    frame_sec: float = FRAME_DURATION_SEC,
    scores: Optional[np.ndarray] = None,
    min_duration_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    """Convert a 1D boolean frame mask back to contiguous interval list.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Binary frame activity mask.
    label : str
        Label name for the segments.
    frame_sec : float
        Frame duration in seconds.
    scores : np.ndarray (float), optional
        Frame confidence scores.
    min_duration_sec : float
        Minimum segment length to retain.

    Returns
    -------
    list of dict with keys: 'start', 'end', 'label', and optional 'score'.
    """
    intervals: List[Dict[str, Any]] = []
    if len(mask) == 0:
        return intervals

    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant", constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        start_sec = round(s * frame_sec, 4)
        end_sec = round(e * frame_sec, 4)
        if (end_sec - start_sec) >= min_duration_sec:
            item: Dict[str, Any] = {
                "start": start_sec,
                "end": end_sec,
                "label": label,
            }
            if scores is not None and len(scores) >= e:
                seg_scores = scores[s:e]
                item["score"] = round(float(np.mean(seg_scores)), 4)
            intervals.append(item)

    return intervals


def compute_boundary_collar_mask(
    y_true: np.ndarray,
    collar_frames: int = COLLAR_FRAMES,
) -> np.ndarray:
    """Compute a boolean collar mask around segment boundaries of ground truth.

    A frame is within collar if its distance to any ground truth transition
    (0 -> 1 or 1 -> 0) is <= collar_frames.

    Parameters
    ----------
    y_true : np.ndarray (bool or binary int)
        Ground truth frame mask.
    collar_frames : int
        Collar radius in frames (e.g. 5 frames = 50 ms for 10 ms grid).

    Returns
    -------
    collar_mask : np.ndarray (bool)
        True for frames within collar tolerance of any boundary.
    """
    n = len(y_true)
    if n == 0 or collar_frames <= 0:
        return np.zeros(n, dtype=bool)

    # Detect boundary frame indices
    padded = np.pad(y_true.astype(np.int8), (1, 1), mode="constant", constant_values=0)
    diff = np.diff(padded)
    # Starts are at index `s`, ends are at index `e` in original coordinates
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    boundaries = np.concatenate([starts, ends])

    collar_mask = np.zeros(n, dtype=bool)
    for b in boundaries:
        left = max(0, b - collar_frames)
        right = min(n, b + collar_frames + 1)
        collar_mask[left:right] = True

    return collar_mask


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    collar_frames: int = COLLAR_FRAMES,
    mode: str = "forgive",
) -> Dict[str, float]:
    """Calculate frame-level Precision, Recall, and F1 with boundary collar tolerance.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth frames (shape [N]).
    y_pred : np.ndarray
        Binary predicted frames (shape [N]).
    collar_frames : int
        Tolerance window on boundaries (default 5 frames = 50 ms).
    mode : str, {"forgive", "exclude"}
        - "forgive" (recommended for speech segmentation): True Positives count
          everywhere. False Positives and False Negatives occurring inside
          the boundary collar are forgiven (not penalized).
        - "exclude": All frames within the collar of ground truth boundaries
          are completely excluded from precision and recall computation.

    Returns
    -------
    dict with keys: 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    y_t = np.asarray(y_true, dtype=bool)
    y_p = np.asarray(y_pred, dtype=bool)

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})"
        )

    if len(y_t) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    # If collar is 0, standard frame-level classification metrics
    if collar_frames <= 0:
        tp = int(np.sum(y_t & y_p))
        fp = int(np.sum((~y_t) & y_p))
        fn = int(np.sum(y_t & (~y_p)))
    else:
        collar = compute_boundary_collar_mask(y_t, collar_frames=collar_frames)
        if mode == "exclude":
            eval_mask = ~collar
            tp = int(np.sum(y_t & y_p & eval_mask))
            fp = int(np.sum((~y_t) & y_p & eval_mask))
            fn = int(np.sum(y_t & (~y_p) & eval_mask))
        elif mode == "forgive":
            # True Positives always counted
            tp = int(np.sum(y_t & y_p))
            # Errors inside the boundary collar are forgiven
            fp = int(np.sum((~y_t) & y_p & (~collar)))
            fn = int(np.sum(y_t & (~y_p) & (~collar)))
        else:
            raise ValueError(f"Unknown collar mode: {mode}. Must be 'forgive' or 'exclude'.")

    # Precision
    if (tp + fp) == 0:
        precision = 1.0 if fp == 0 else 0.0
    else:
        precision = float(tp / (tp + fp))

    # Recall
    if (tp + fn) == 0:
        recall = 1.0 if fn == 0 else 0.0
    else:
        recall = float(tp / (tp + fn))

    # F1
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_channels(
    ground_truth: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, List[Dict[str, Any]]],
    num_frames: int,
    frame_sec: float = FRAME_DURATION_SEC,
    collar_frames: int = COLLAR_FRAMES,
    labels: Sequence[str] = EVAL_LABELS,
    mode: str = "forgive",
) -> Dict[str, Any]:
    """Evaluate multi-label predictions across channels 'A' and 'B'.

    Parameters
    ----------
    ground_truth : dict
        {"A": [...], "B": [...]} with segment dicts.
    predictions : dict
        {"A": [...], "B": [...]} with segment dicts.
    num_frames : int
        Total frames in evaluation recording.
    frame_sec : float
        Frame step in seconds (0.010 = 10 ms).
    collar_frames : int
        Collar radius (5 frames = 50 ms).
    labels : sequence of str
        Labels to evaluate ("own", "leak", "third").
    mode : str
        Collar handling mode ("forgive" or "exclude").

    Returns
    -------
    dict with detailed per-channel, per-label metrics and macro F1 scores.
    """
    results: Dict[str, Any] = {
        "by_channel": {},
        "by_label": {},
        "macro_f1": 0.0,
    }

    label_f1s: Dict[str, List[float]] = {lbl: [] for lbl in labels}

    for ch in ("A", "B"):
        gt_intervals = ground_truth.get(ch, [])
        pred_intervals = predictions.get(ch, [])

        gt_masks = intervals_to_frame_matrix(gt_intervals, num_frames, frame_sec=frame_sec, labels=labels)
        pred_masks = intervals_to_frame_matrix(pred_intervals, num_frames, frame_sec=frame_sec, labels=labels)

        results["by_channel"][ch] = {}
        for lbl in labels:
            m = precision_recall_f1(
                gt_masks[lbl],
                pred_masks[lbl],
                collar_frames=collar_frames,
                mode=mode,
            )
            results["by_channel"][ch][lbl] = m
            label_f1s[lbl].append(m["f1"])

    # Aggregate across channels for each label
    all_f1s = []
    for lbl in labels:
        mean_f1 = float(np.mean(label_f1s[lbl]))
        results["by_label"][lbl] = {"f1": mean_f1}
        all_f1s.append(mean_f1)

    results["macro_f1"] = float(np.mean(all_f1s)) if all_f1s else 0.0
    return results


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Calculate cosine similarity between two voice embedding vectors.

    cos_sim = (a . b) / (||a|| * ||b||)
    """
    a = np.asarray(embedding_a, dtype=np.float64).flatten()
    b = np.asarray(embedding_b, dtype=np.float64).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def measure_rtf(
    process_fn,
    *args,
    audio_duration_sec: float,
    num_threads: int = 1,
    **kwargs,
) -> Tuple[Any, float, bool]:
    """Measure Real-Time Factor (RTF) of a processing function strictly on single CPU core.

    Parameters
    ----------
    process_fn : callable
        Function to benchmark.
    *args, **kwargs :
        Arguments passed to process_fn.
    audio_duration_sec : float
        Length of processed audio in seconds.
    num_threads : int
        Number of CPU threads to enforce (default 1).

    Returns
    -------
    result : Any
        Result of process_fn(*args, **kwargs).
    rtf : float
        Elapsed CPU time divided by audio duration.
    passed_limit : bool
        True if rtf <= MAX_RTF_LIMIT (1/15 ≈ 0.0667).
    """
    try:
        import torch

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        has_torch = True
    except ImportError:
        has_torch = False
        prev_threads = 1

    start_time = time.perf_counter()
    try:
        result = process_fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start_time
        if has_torch:
            torch.set_num_threads(prev_threads)

    rtf = elapsed / max(audio_duration_sec, 1e-6)
    passed_limit = rtf <= MAX_RTF_LIMIT

    return result, rtf, passed_limit


"""Training script for the contextual 1D-TCN extraneous speech detector.

Trains a dilated residual 1D-CNN (TCN) on realistic physical telephone dialogues
synthesized with real OpenSLR 28 RIRs and disjoint training speakers from
Russian speech + VoxCeleb1 datasets.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import torch.nn as nn

from src.data_loader import RealSpeechLoader
from src.detector import _DetectorNet
from src.features import FeatureExtractor
from src.metrics import intervals_to_frame_matrix, precision_recall_f1
from src.synthesizer import TelephoneDialogueSynthesizer


def train_detector(
    weights_path: str = "models/detector_weights.pt",
    dialogues_per_scenario: int = 12,
    epochs: int = 150,
    lr: float = 0.005,
    seed: int = 42,
) -> None:
    print("=" * 68)
    print("Training 1D-TCN Extraneous Speech Detector on Realistic Audio & RIRs")
    print("=" * 68)

    loader = RealSpeechLoader()
    if hasattr(loader._loader, "get_disjoint_splits"):
        train_spks = loader._loader.get_disjoint_splits()["train"]
    else:
        all_spks = list(loader.discover_speakers().keys())
        train_spks = all_spks[: len(all_spks) // 2]

    print(f"Discovered {len(train_spks)} training speakers: {train_spks[:6]}...")
    rng = np.random.RandomState(seed)
    synth = TelephoneDialogueSynthesizer(seed=seed)
    fe = FeatureExtractor()

    scenarios = ["normal", "double_talk", "high_leak_lt_10db", "path_drift", "third_speaker"]
    
    # 12 dialogues per scenario = 60 total dialogues.
    scenario_counts = {sc: dialogues_per_scenario for sc in scenarios}

    dialogues_x: List[np.ndarray] = []
    dialogues_y: List[np.ndarray] = []

    print(f"Synthesizing dialogues with OpenSLR 28 RIRs (80% Russian speaker preference)...")
    for sc, n_d in scenario_counts.items():
        print(f"  Generating {n_d} dialogues for scenario: {sc}")
        for _ in range(n_d):
            sa, sb, st = loader.sample_dialogue_speakers("train", russian_prob=0.80, rng=rng)

            pool = loader.build_utterance_pool(sa, sb, st, max_per_speaker=8)
            dur = float(synth.rng.uniform(7.0, 11.0))
            res = synth.synthesize_dialogue(
                utterance_pool=pool,
                duration_sec=dur,
                scenario=sc,
                third_speaker_prob=1.0 if sc == "third_speaker" else 0.3,
            )

            feats = fe.extract_features(res.audio)

            for ch in ("A", "B"):
                x_ch = feats[ch]  # Shape: (T, 36)
                gt = intervals_to_frame_matrix(res.ground_truth_intervals[ch], num_frames=len(x_ch))
                y_ch = np.column_stack([gt["own"], gt["leak"], gt["third"]]).astype(np.float32)

                dialogues_x.append(x_ch)
                dialogues_y.append(y_ch)

    print(f"Collected {len(dialogues_x)} channel sequences.")
    all_x = np.vstack(dialogues_x)
    mean = np.mean(all_x, axis=0, keepdims=True)
    std = np.std(all_x, axis=0, keepdims=True) + 1e-6
    print(f"Feature dimension: {all_x.shape[1]}")

    in_dim = all_x.shape[1]
    model = _DetectorNet(in_features=in_dim, hidden_dim=48, num_classes=3)

    # Class balancing weights: pos_weight = [1.0, 1.2, 1.3]
    all_y = np.vstack(dialogues_y)
    n_own = int(np.sum(all_y[:, 0]))
    n_leak = int(np.sum(all_y[:, 1]))
    n_third = int(np.sum(all_y[:, 2]))
    print(f"Class frame distribution -> Own: {n_own}, Leak: {n_leak}, Third: {n_third}")

    pos_weight = torch.tensor([1.0, 1.2, 1.3], dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Convert sequences to normalized tensors
    norm_sequences = []
    for x_seq, y_seq in zip(dialogues_x, dialogues_y):
        x_norm = (x_seq - mean) / std
        # (T, C) -> (1, C, T)
        t_x = torch.tensor(x_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        # (T, 3) -> (1, 3, T)
        t_y = torch.tensor(y_seq, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        norm_sequences.append((t_x, t_y))

    split = max(1, int(0.8 * len(norm_sequences)))
    order = rng.permutation(len(norm_sequences))
    train_indices = order[:split]
    val_indices = order[split:]
    best_macro = -1.0
    best_state = None

    print(f"Training 1D-TCN for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(train_indices)
        epoch_loss = 0.0

        for idx in perm:
            t_x, t_y = norm_sequences[idx]
            optimizer.zero_grad()
            logits = model(t_x)

            loss = nn.functional.binary_cross_entropy_with_logits(logits, t_y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        val_scores = []
        with torch.no_grad():
            for idx in val_indices:
                t_x, t_y = norm_sequences[idx]
                pred = (torch.sigmoid(model(t_x)).squeeze(0).transpose(0, 1).numpy() >= 0.5)
                truth = t_y.squeeze(0).transpose(0, 1).numpy().astype(bool)
                val_scores.append([precision_recall_f1(truth[:, col], pred[:, col])["f1"] for col in range(3)])
        val_macro = float(np.mean(val_scores)) if val_scores else 0.0
        if val_macro > best_macro:
            best_macro = val_macro
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.train()

        if (ep + 1) % 10 == 0 or (ep + 1) == epochs:
            avg_loss = epoch_loss / len(train_indices)
            print(
                f"  Epoch {ep+1:3d}/{epochs} | Train Loss: {avg_loss:.4f} "
                f"| Val Macro F1: {val_macro:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation Macro F1: {best_macro:.4f}")

    # Save model weights
    out_dir = Path(weights_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_dim": in_dim,
        "feature_names": fe.extract_features(np.zeros((2, 160), dtype=np.float32))["feature_names"],
        "architecture": "1D-TCN",
        "dilations": [1, 2, 4, 8],
    }
    torch.save(save_dict, weights_path)
    print(f"\nTrained 1D-TCN detector successfully saved to: {weights_path}")


if __name__ == "__main__":
    train_detector(dialogues_per_scenario=12, epochs=150)

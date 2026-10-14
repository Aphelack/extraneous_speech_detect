"""Training script for the lightweight neural extraneous speech detector.

Trains a compact multi-label neural detector on diverse physical scenarios
synthesized from real speech utterances, and saves calibrated model weights.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.data_loader import RealSpeechLoader
from src.features import FeatureExtractor
from src.metrics import intervals_to_frame_matrix, precision_recall_f1
from src.synthesizer import TelephoneDialogueSynthesizer


class LightweightDetectorNet(nn.Module):
    """Compact multi-label feedforward network (RTF < 0.001)."""

    def __init__(self, in_features: int = 9, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_detector(
    weights_path: str = "models/detector_weights.pt",
    num_dialogues_per_scenario: int = 4,
    epochs: int = 160,
    lr: float = 0.015,
) -> None:
    print("=" * 60)
    print("Training Extraneous Speech Detector on Real Speech Data")
    print("=" * 60)

    loader = RealSpeechLoader(data_root="data/cmuarctic")
    speakers = loader.discover_speakers()
    print(f"Discovered speakers: {list(speakers.keys())}")

    speaker_a = "aew" if "aew" in speakers else list(speakers.keys())[0]
    speaker_b = "slt" if "slt" in speakers else list(speakers.keys())[1]
    speaker_3 = "bdl" if "bdl" in speakers else None

    pool = loader.build_utterance_pool(speaker_a, speaker_b, speaker_3, max_per_speaker=30)
    synth = TelephoneDialogueSynthesizer(seed=42)
    fe = FeatureExtractor()

    scenarios = ["normal", "double_talk", "high_leak_lt_10db", "path_drift", "third_speaker"]
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    print(f"Generating training dialogues across scenarios: {scenarios}...")
    for sc in scenarios:
        for _ in range(num_dialogues_per_scenario):
            dur = float(synth.rng.uniform(6.0, 10.0))
            res = synth.synthesize_dialogue(
                utterance_pool=pool,
                duration_sec=dur,
                scenario=sc,
                third_speaker_prob=1.0 if sc == "third_speaker" else 0.2,
            )
            feats = fe.extract_features(res.audio)

            for ch in ("A", "B"):
                other_ch = "B" if ch == "A" else "A"
                # Combine local channel features + counterpart energy
                other_energy = feats[other_ch][:, 0:1]
                x_ch = np.hstack([feats[ch], other_energy])

                gt = intervals_to_frame_matrix(res.ground_truth_intervals[ch], num_frames=len(x_ch))
                y_ch = np.column_stack([gt["own"], gt["leak"], gt["third"]]).astype(np.float32)

                X_list.append(x_ch)
                y_list.append(y_ch)

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.vstack(y_list).astype(np.float32)

    # Feature standardization
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-6
    X_train_norm = (X_train - mean) / std

    print(f"Dataset compiled: {len(X_train)} frames, {X_train.shape[1]} features.")
    print(f"Positive counts -> Own: {int(np.sum(y_train[:, 0]))}, Leak: {int(np.sum(y_train[:, 1]))}, Third: {int(np.sum(y_train[:, 2]))}")

    # PyTorch Tensors
    X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Pos weights for balanced class training (capped to prevent spurious false positives)
    pos_weight_tensor = torch.tensor([1.0, 1.2, 1.5], dtype=torch.float32)

    model = LightweightDetectorNet(in_features=X_train.shape[1], hidden_dim=32, num_classes=3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (ep + 1) % 40 == 0:
            print(f"Epoch {ep+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    # Save model and normalization constants
    out_dir = Path(weights_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "feature_names": fe.extract_features(np.zeros((2, 160), dtype=np.float32))["feature_names"] + ["other_db_energy"],
        "pos_weights": [1.0, 1.2, 1.5],
    }
    torch.save(save_dict, weights_path)
    print(f"Trained model saved to: {weights_path}")


if __name__ == "__main__":
    train_detector()

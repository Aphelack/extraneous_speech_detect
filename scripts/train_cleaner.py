"""Train the delay-aligned 32-band IRM cleaner."""

import sys
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn

from src.cleaner import CleanerConfig, TelephoneAudioCleaner, _CleanerNet
from src.config import FRAME_SAMPLES, SAMPLE_RATE
from src.data_loader import RealSpeechLoader
from src.detector import ExtraneousSpeechDetector
from src.synthesizer import TelephoneDialogueSynthesizer


def train_cleaner(
    weights_path: str = "models/cleaner_weights.pt",
    dialogues_per_scenario: int = 12,
    epochs: int = 100,
    lr: float = 0.005,
    seed: int = 42,
) -> None:
    loader = RealSpeechLoader()
    rng = np.random.RandomState(seed)
    synth = TelephoneDialogueSynthesizer(seed=seed)
    detector = ExtraneousSpeechDetector()
    cleaner = TelephoneAudioCleaner(detector=detector)
    config = CleanerConfig()
    window = signal.windows.hann(config.n_fft)
    scenarios = ["normal", "double_talk", "high_leak_lt_10db", "path_drift", "third_speaker"]

    features: List[np.ndarray] = []
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    frame_weights: List[np.ndarray] = []

    for scenario in scenarios:
        for _ in range(dialogues_per_scenario):
            sa, sb, st = loader.sample_dialogue_speakers("train", russian_prob=0.80, rng=rng)
            pool = loader.build_utterance_pool(sa, sb, st, max_per_speaker=8)
            result = synth.synthesize_dialogue(
                utterance_pool=pool,
                duration_sec=float(synth.rng.uniform(7.0, 11.0)),
                scenario=scenario,
                third_speaker_prob=1.0 if scenario == "third_speaker" else 0.3,
            )
            probabilities = detector.predict_probabilities(result.audio, result.sample_rate)
            for channel, other, target_key in (("A", "B", "A"), ("B", "A", "B")):
                input_audio = result.audio[0 if channel == "A" else 1]
                other_audio = result.audio[0 if other == "A" else 1]
                target_audio = result.target_audio[target_key] if result.target_audio else result.clean_references[target_key]
                length = min(len(input_audio), len(other_audio), len(target_audio))
                _, _, input_spec = signal.stft(input_audio[:length], fs=result.sample_rate, window=window, nperseg=config.n_fft, noverlap=config.n_fft - FRAME_SAMPLES)
                _, _, other_spec = signal.stft(other_audio[:length], fs=result.sample_rate, window=window, nperseg=config.n_fft, noverlap=config.n_fft - FRAME_SAMPLES)
                _, _, target_spec = signal.stft(target_audio[:length], fs=result.sample_rate, window=window, nperseg=config.n_fft, noverlap=config.n_fft - FRAME_SAMPLES)
                leak_spec = input_spec - target_spec
                probs = probabilities[channel]
                mag_input = np.abs(input_spec)
                delays = cleaner.estimate_delay_frames(mag_input, np.abs(other_spec))
                aligned_other = cleaner.align_spectrum(other_spec, delays)
                p_own = np.resize(probs["own"], input_spec.shape[1])
                p_leak = np.resize(probs["leak"], input_spec.shape[1])
                leak_estimate = cleaner._estimate_leakage(mag_input, np.abs(aligned_other), p_own, p_leak)
                x = cleaner.extract_features(input_spec, other_spec, p_own, p_leak, np.resize(probs["third"], input_spec.shape[1]), aligned_other, leak_estimate)
                target_mag = np.abs(target_spec).astype(np.float32)
                leak_mag = np.abs(leak_spec).astype(np.float32)
                irm = np.sqrt(target_mag ** 2 / (target_mag ** 2 + leak_mag ** 2 + 1e-8)).astype(np.float32)
                features.append(x)
                inputs.append(mag_input.astype(np.float32))
                targets.append(target_mag)
                frame_weights.append(np.full(input_spec.shape[1], 2.0 if scenario == "double_talk" else 1.0, dtype=np.float32))

    all_features = np.hstack(features)
    mean = np.mean(all_features, axis=1, keepdims=True)
    std = np.std(all_features, axis=1, keepdims=True) + 1e-6
    bank = torch.tensor(cleaner.filterbank.T, dtype=torch.float32)
    dataset = [
        (
            torch.tensor((x - mean) / std, dtype=torch.float32),
            torch.tensor(input_mag, dtype=torch.float32),
            torch.tensor(target_mag, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )
        for x, input_mag, target_mag, weight in zip(features, inputs, targets, frame_weights)
    ]

    model = _CleanerNet(in_channels=all_features.shape[0], hidden_dim=48, num_bands=config.num_bands)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for index in rng.permutation(len(dataset)):
            x, input_mag, target_mag, weight = dataset[index]
            optimizer.zero_grad()
            predicted_bands = model(x.unsqueeze(0)).squeeze(0)
            predicted_mask = torch.clamp(bank @ predicted_bands, 0.0, 1.0)
            irm = torch.sqrt(target_mag ** 2 / (target_mag ** 2 + (input_mag - target_mag) ** 2 + 1e-8))
            frame_weight = weight.unsqueeze(0)
            mse = ((predicted_mask - irm) ** 2 * frame_weight).sum() / frame_weight.sum() / predicted_mask.shape[0]
            spectral = torch.abs((input_mag * predicted_mask).clamp_min(1e-8).pow(0.3) - target_mag.clamp_min(1e-8).pow(0.3)).mean()
            loss = mse + spectral
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        if (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1:3d}/{epochs}: {total_loss / len(dataset):.5f}")

    output = Path(weights_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "in_channels": all_features.shape[0],
        "num_bands": config.num_bands,
        "hidden_dim": 48,
        "architecture": "delay-aligned-irm-tcn",
    }, output)
    print(f"Saved cleaner weights to {output}")


if __name__ == "__main__":
    train_cleaner()

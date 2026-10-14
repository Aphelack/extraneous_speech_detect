"""Configuration constants and dataclasses for signal simulation, detection, and evaluation."""

from dataclasses import dataclass, field
from typing import Tuple


# --- Core Signal Specifications ---
SAMPLE_RATE: int = 8000
FRAME_DURATION_SEC: float = 0.010  # 10 ms grid
FRAME_SAMPLES: int = int(SAMPLE_RATE * FRAME_DURATION_SEC)  # 80 samples

# Boundary evaluation tolerance (collar)
COLLAR_SEC: float = 0.050  # 50 ms collar
COLLAR_FRAMES: int = int(round(COLLAR_SEC / FRAME_DURATION_SEC))  # 5 frames

# Channels and Labels
CHANNELS: Tuple[str, str] = ("A", "B")
LABEL_SILENCE: str = "silence"
LABEL_OWN: str = "own"
LABEL_LEAK: str = "leak"
LABEL_THIRD: str = "third"
EVAL_LABELS: Tuple[str, ...] = (LABEL_OWN, LABEL_LEAK, LABEL_THIRD)
ALL_LABELS: Tuple[str, ...] = (LABEL_SILENCE, LABEL_OWN, LABEL_LEAK, LABEL_THIRD)

# --- Telephone Signal Model Parameters ---
# Narrowband telephone transmission: 300 - 3400 Hz
BANDPASS_LOW_HZ: float = 300.0
BANDPASS_HIGH_HZ: float = 3400.0

# Leakage path parameters
LEAK_DELAY_MIN_SEC: float = 0.020  # 20 ms
LEAK_DELAY_MAX_SEC: float = 0.300  # 300 ms
LEAK_DELAY_MIN_SAMPLES: int = int(SAMPLE_RATE * LEAK_DELAY_MIN_SEC)  # 160 samples
LEAK_DELAY_MAX_SAMPLES: int = int(SAMPLE_RATE * LEAK_DELAY_MAX_SEC)  # 2400 samples

LEAK_ATTENUATION_MIN_DB: float = 6.0
LEAK_ATTENUATION_MAX_DB: float = 40.0

# Packet loss chunk duration (G.711 / VoIP packetization)
PACKET_LOSS_CHUNK_SEC: float = 0.020  # 20 ms
PACKET_LOSS_CHUNK_SAMPLES: int = int(SAMPLE_RATE * PACKET_LOSS_CHUNK_SEC)  # 160 samples

# Baseline simple rule threshold (+6 dB)
BASELINE_THRESHOLD_DB: float = 6.0

# RTF (Real-Time Factor) requirement: <= 1/15 on single CPU core (<= 4s per 60s audio)
MAX_RTF_LIMIT: float = 1.0 / 15.0


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    frame_sec: float = FRAME_DURATION_SEC
    frame_samples: int = FRAME_SAMPLES
    channels: Tuple[str, str] = CHANNELS


@dataclass
class BaselineConfig:
    threshold_db: float = BASELINE_THRESHOLD_DB
    frame_sec: float = FRAME_DURATION_SEC
    vad_energy_threshold_db: float = -45.0  # relative dB threshold for speech/silence
    min_speech_duration_sec: float = 0.05   # merge/clean tiny spurious detections


@dataclass
class SynthesisConfig:
    sample_rate: int = SAMPLE_RATE
    bandpass_low: float = BANDPASS_LOW_HZ
    bandpass_high: float = BANDPASS_HIGH_HZ
    leak_delay_range_sec: Tuple[float, float] = (LEAK_DELAY_MIN_SEC, LEAK_DELAY_MAX_SEC)
    leak_attenuation_range_db: Tuple[float, float] = (LEAK_ATTENUATION_MIN_DB, LEAK_ATTENUATION_MAX_DB)
    packet_loss_chunk_sec: float = PACKET_LOSS_CHUNK_SEC
    packet_loss_prob: float = 0.01  # 1% packet loss
    apply_alaw: bool = True
    apply_drift: bool = False
    drift_speed_ms_per_sec: float = 2.0
    apply_speaker_saturation: bool = True
    third_speaker_attenuation_db: float = 12.0
    snr_db_range: Tuple[float, float] = (15.0, 30.0)


@dataclass
class EvaluationConfig:
    collar_sec: float = COLLAR_SEC
    frame_sec: float = FRAME_DURATION_SEC
    collar_frames: int = COLLAR_FRAMES
    eval_labels: Tuple[str, ...] = EVAL_LABELS


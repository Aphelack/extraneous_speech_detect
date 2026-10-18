"""Telephone signal and dialogue synthesizer.

Implements the physical transmission and leakage model described in the specification:
- Dialogue between two participants (A and B) with consistent speaker_id per channel
- Episodic third speaker (more distant/reverberant RIR, lower amplitude)
- Leakage path h_BA:
  - Acoustic coupling / RIR
  - Delay tau in [20, 300] ms with optional drift tau(t)
  - Attenuation in [6, 40] dB
  - Loudspeaker non-linearity phi (soft clipping / saturation)
- Narrowband telephone channel T:
  - Bandpass filtering 300 - 3400 Hz
  - G.711 A-law companding & 8-bit quantization
  - Random 20 ms packet loss (PLC simulation)
  - Additive ambient noise n(t) at target SNR
- Ground truth generation: exact 10 ms frame masks, interval lists, and clean owner references.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal as signal

from src.config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    COLLAR_SEC,
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LABEL_LEAK,
    LABEL_OWN,
    LABEL_THIRD,
    LEAK_ATTENUATION_MAX_DB,
    LEAK_ATTENUATION_MIN_DB,
    LEAK_DELAY_MAX_SEC,
    LEAK_DELAY_MIN_SEC,
    PACKET_LOSS_CHUNK_SAMPLES,
    PACKET_LOSS_CHUNK_SEC,
    SAMPLE_RATE,
    SynthesisConfig,
)
from src.metrics import frame_mask_to_intervals


def generate_synthetic_rir(
    sample_rate: int = SAMPLE_RATE,
    t60_sec: float = 0.20,
    direct_to_reverberant_ratio_db: float = 12.0,
    length_sec: float = 0.30,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate physically-motivated room impulse response (RIR) using exponentially decaying noise."""
    if rng is None:
        rng = np.random.RandomState()

    num_samples = int(sample_rate * length_sec)
    t = np.arange(num_samples) / sample_rate

    # Reverberation decay rate: decay by 60 dB in t60_sec
    # A(t) = exp(-6.908 * t / t60)
    decay = np.exp(-6.908 * t / max(t60_sec, 0.01))
    noise = rng.randn(num_samples).astype(np.float32)
    reverb = noise * decay

    # Direct path impulse at t=0
    direct = np.zeros(num_samples, dtype=np.float32)
    direct[0] = 1.0

    # Scale direct vs reverberant energy
    dr_linear = 10.0 ** (direct_to_reverberant_ratio_db / 20.0)
    reverb_norm = np.linalg.norm(reverb) + 1e-10
    rir = direct + (reverb / reverb_norm) * (1.0 / dr_linear)

    # Normalize peak to 1.0
    rir = rir / (np.max(np.abs(rir)) + 1e-10)
    return rir.astype(np.float32)


class RIRManager:
    """Manages real and simulated RIRs and real noises from OpenSLR 28.

    Falls back to generate_synthetic_rir if the dataset directory is absent.
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "data/rirs_noises/RIRS_NOISES",
        target_sr: int = SAMPLE_RATE,
    ):
        self.target_sr = target_sr
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            repo_root = Path(__file__).resolve().parent.parent
            self.data_root = repo_root / data_root

        self.available = self.data_root.exists()
        self._owner_rirs: List[Path] = []
        self._third_rirs: List[Path] = []
        self._leak_rirs: List[Path] = []
        self._noise_files: List[Path] = []

        if self.available:
            self._index_files()

    def _index_files(self):
        root = self.data_root
        real_dir = root / "real_rirs_isotropic_noises"
        sim_dir = root / "simulated_rirs"
        noise_dir = root / "pointsource_noises"

        if real_dir.exists():
            self._owner_rirs.extend(list(real_dir.glob("*ane*.wav")))
            self._third_rirs.extend(list(real_dir.glob("*ofc*.wav")))
            self._third_rirs.extend(list(real_dir.glob("*e2b*.wav")))
            self._leak_rirs.extend(list(real_dir.glob("RWCP_type*.wav")))
            self._noise_files.extend(list(real_dir.glob("*noise*.wav")))

        if sim_dir.exists():
            small = list((sim_dir / "smallroom").glob("*/*.wav"))
            med = list((sim_dir / "mediumroom").glob("*/*.wav"))
            large = list((sim_dir / "largeroom").glob("*/*.wav"))
            self._owner_rirs.extend(small[:300])
            self._third_rirs.extend(med[:300] + large[:300])
            self._leak_rirs.extend(small[300:500] if len(small) > 500 else small[:200])

        if noise_dir.exists():
            self._noise_files.extend(list(noise_dir.glob("*.wav"))[:100])

    def _load_wav_as_rir(self, path: Path, max_len_sec: float = 0.5) -> np.ndarray:
        import soundfile as sf
        data, orig_sr = sf.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        if orig_sr != self.target_sr:
            data = signal.resample_poly(data, self.target_sr // 1000, orig_sr // 1000)
        max_samples = int(self.target_sr * max_len_sec)
        if len(data) > max_samples:
            data = data[:max_samples]
        peak = np.max(np.abs(data))
        if peak > 1e-6:
            data = data / peak
        return data.astype(np.float32)

    def get_owner_rir(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if self.available and self._owner_rirs:
            if rng is None:
                rng = np.random.RandomState()
            p = self._owner_rirs[rng.randint(len(self._owner_rirs))]
            try:
                rir = self._load_wav_as_rir(p, max_len_sec=0.25)
                # Boost direct path to guarantee high DRR for close-talk microphone
                direct_idx = int(np.argmax(np.abs(rir)))
                rir[direct_idx] = np.sign(rir[direct_idx]) * max(abs(rir[direct_idx]), 1.5)
                return (rir / np.max(np.abs(rir))).astype(np.float32)
            except Exception:
                pass
        return generate_synthetic_rir(self.target_sr, t60_sec=0.18, direct_to_reverberant_ratio_db=15.0, rng=rng)

    def get_third_speaker_rir(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if self.available and self._third_rirs:
            if rng is None:
                rng = np.random.RandomState()
            p = self._third_rirs[rng.randint(len(self._third_rirs))]
            try:
                return self._load_wav_as_rir(p, max_len_sec=0.50)
            except Exception:
                pass
        return generate_synthetic_rir(self.target_sr, t60_sec=0.45, direct_to_reverberant_ratio_db=3.0, rng=rng)

    def get_leakage_rir(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if self.available and self._leak_rirs:
            if rng is None:
                rng = np.random.RandomState()
            p = self._leak_rirs[rng.randint(len(self._leak_rirs))]
            try:
                return self._load_wav_as_rir(p, max_len_sec=0.30)
            except Exception:
                pass
        return generate_synthetic_rir(self.target_sr, t60_sec=0.22, direct_to_reverberant_ratio_db=8.0, rng=rng)

    def add_realistic_noise(
        self,
        x: np.ndarray,
        snr_db: float = 25.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        if self.available and self._noise_files:
            if rng is None:
                rng = np.random.RandomState()
            p = self._noise_files[rng.randint(len(self._noise_files))]
            try:
                import soundfile as sf
                info = sf.info(str(p))
                needed_samples = int(len(x) * (info.samplerate / self.target_sr))
                max_start = max(0, info.frames - needed_samples)
                start_frame = rng.randint(0, max_start + 1) if max_start > 0 else 0
                noise_slice, n_sr = sf.read(str(p), start=start_frame, frames=needed_samples)
                if noise_slice.ndim > 1:
                    noise_slice = noise_slice[:, 0]
                if n_sr != self.target_sr:
                    noise_slice = signal.resample_poly(noise_slice, self.target_sr // 1000, n_sr // 1000)
                if len(noise_slice) < len(x):
                    noise_slice = np.pad(noise_slice, (0, len(x) - len(noise_slice)), mode="wrap")
                else:
                    noise_slice = noise_slice[:len(x)]

                # Bandpass filter noise to telephone channel
                noise_slice = apply_bandpass_filter(noise_slice, sample_rate=self.target_sr)
                sig_p = np.mean(x ** 2) + 1e-12
                noise_p = np.mean(noise_slice ** 2) + 1e-12
                target_np = sig_p / (10.0 ** (snr_db / 10.0))
                scaled = noise_slice * np.sqrt(target_np / noise_p)
                return (x + scaled).astype(np.float32)
            except Exception:
                pass
        return add_colored_noise(x, snr_db=snr_db, sample_rate=self.target_sr, rng=rng)


def apply_speaker_nonlinearity(
    x: np.ndarray,
    saturation_factor: float = 1.8,
) -> np.ndarray:
    """Model loudspeaker membrane non-linearity (soft clipping / harmonic distortion).

    phi(x) = tanh(alpha * x) / tanh(alpha)
    """
    alpha = max(1.0, float(saturation_factor))
    norm_factor = np.tanh(alpha)
    return (np.tanh(alpha * x) / norm_factor).astype(np.float32)


def apply_alaw_compression(x: np.ndarray, a_param: float = 87.6) -> np.ndarray:
    """Simulate G.711 A-law 8-bit companding and quantization.

    F(x) = sgn(x) * (A * |x|) / (1 + ln A), for |x| < 1/A
    F(x) = sgn(x) * (1 + ln(A * |x|)) / (1 + ln A), for 1/A <= |x| <= 1
    """
    orig_shape = x.shape
    flat_x = np.clip(x.flatten(), -1.0, 1.0)
    abs_x = np.abs(flat_x)
    sgn_x = np.sign(flat_x)
    denom = 1.0 + np.log(a_param)

    # Compression curve
    low_mask = abs_x < (1.0 / a_param)
    compressed = np.zeros_like(flat_x)
    compressed[low_mask] = (a_param * abs_x[low_mask]) / denom

    high_mask = ~low_mask
    safe_high = np.maximum(a_param * abs_x[high_mask], 1e-12)
    compressed[high_mask] = (1.0 + np.log(safe_high)) / denom

    # 8-bit quantization (128 discrete magnitude levels per sign)
    quantized = np.round(compressed * 127.0) / 127.0

    # Expansion curve
    expanded = np.zeros_like(flat_x)
    low_q = quantized < (1.0 / denom)
    expanded[low_q] = (quantized[low_q] * denom) / a_param

    high_q = ~low_q
    expanded[high_q] = np.exp(quantized[high_q] * denom - 1.0) / a_param

    out = sgn_x * expanded
    return out.reshape(orig_shape).astype(np.float32)


def apply_bandpass_filter(
    x: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Apply telephone transmission bandpass filter (300 - 3400 Hz)."""
    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return signal.sosfilt(sos, x).astype(np.float32)


def apply_packet_loss(
    x: np.ndarray,
    chunk_samples: int = PACKET_LOSS_CHUNK_SAMPLES,
    drop_prob: float = 0.01,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Randomly drop / zero-out 20 ms transmission packets (VoIP packet loss)."""
    if drop_prob <= 0.0:
        return x.copy()
    if rng is None:
        rng = np.random.RandomState()

    out = x.copy()
    n_chunks = len(out) // chunk_samples
    for i in range(n_chunks):
        if rng.rand() < drop_prob:
            out[i * chunk_samples : (i + 1) * chunk_samples] = 0.0

    return out


def apply_time_varying_delay(
    x: np.ndarray,
    delays_samples: np.ndarray,
) -> np.ndarray:
    """Apply fractional delay with arbitrary sample-by-sample drift tau[n].

    Uses linear interpolation between adjacent samples.
    """
    n = len(x)
    m = np.arange(n, dtype=np.float32) - delays_samples
    i = np.floor(m).astype(np.int64)
    frac = m - i

    valid = (i >= 0) & (i < n - 1)
    out = np.zeros(n, dtype=np.float32)

    valid_indices = np.where(valid)[0]
    i_val = i[valid_indices]
    f_val = frac[valid_indices]

    out[valid_indices] = (1.0 - f_val) * x[i_val] + f_val * x[i_val + 1]
    return out


def add_colored_noise(
    x: np.ndarray,
    snr_db: float = 25.0,
    sample_rate: int = SAMPLE_RATE,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Add bandlimited comfort / background noise with specified SNR."""
    if rng is None:
        rng = np.random.RandomState()

    signal_power = np.mean(x ** 2) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    raw_noise = rng.randn(len(x)).astype(np.float32)
    # Bandlimit noise to telephone band
    filtered_noise = apply_bandpass_filter(raw_noise, sample_rate=sample_rate)
    filtered_noise_power = np.mean(filtered_noise ** 2) + 1e-12
    scaled_noise = filtered_noise * np.sqrt(noise_power / filtered_noise_power)

    return (x + scaled_noise).astype(np.float32)


@dataclass
class TurnSegment:
    speaker: str  # "A", "B", "third"
    start_sec: float
    end_sec: float
    audio: np.ndarray


@dataclass
class SynthesisResult:
    audio: np.ndarray  # Shape (2, num_samples): [0] is Track A, [1] is Track B
    sample_rate: int
    duration_sec: float
    num_frames: int
    ground_truth_intervals: Dict[str, List[Dict[str, Any]]]
    clean_references: Dict[str, np.ndarray]  # Pure s_A, s_B without leak or distortion
    scenario: str
    metadata: Dict[str, Any]
    target_audio: Optional[Dict[str, np.ndarray]] = None


class TelephoneDialogueSynthesizer:
    """Full acoustic telephone dialogue synthesizer."""

    def __init__(
        self,
        config: Optional[SynthesisConfig] = None,
        seed: Optional[int] = None,
        rir_manager: Optional[RIRManager] = None,
    ):
        self.config = config or SynthesisConfig()
        self.rng = np.random.RandomState(seed)
        self.rir_manager = rir_manager or RIRManager(target_sr=self.config.sample_rate)

    def generate_synthetic_speech_utterance(
        self,
        duration_sec: float,
        pitch_hz: float = 150.0,
        formants: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Generate a synthetic harmonic vowel-like utterance with glottal pulse excitation."""
        sr = self.config.sample_rate
        num_samples = int(sr * duration_sec)
        t = np.arange(num_samples) / sr

        # Glottal pulse train with slight jitter
        period = int(sr / pitch_hz)
        excitation = np.zeros(num_samples, dtype=np.float32)
        excitation[::period] = 1.0

        # Formant resonators
        if formants is None:
            formants = [500.0, 1500.0, 2500.0]

        filtered = excitation.copy()
        for f in formants:
            b, a = signal.iirpeak(f, Q=5.0, fs=sr)
            filtered = signal.lfilter(b, a, filtered)

        # Smooth envelope (fade in / fade out)
        env = np.hanning(min(num_samples, int(sr * 0.05)))
        envelope = np.ones(num_samples, dtype=np.float32)
        n_fade = len(env) // 2
        envelope[:n_fade] = env[:n_fade]
        envelope[-n_fade:] = env[-n_fade:]

        out = filtered * envelope
        max_val = np.max(np.abs(out)) + 1e-6
        return (out / max_val * 0.6).astype(np.float32)

    def simulate_telephone_channel(
        self,
        signal_clean: np.ndarray,
        snr_db: float = 25.0,
    ) -> np.ndarray:
        """Apply telephone transmission chain T(x): Bandpass -> A-law -> Packet loss -> Noise."""
        # 1. Bandpass 300 - 3400 Hz
        x = apply_bandpass_filter(signal_clean, sample_rate=self.config.sample_rate)

        # 2. G.711 A-law companding
        if self.config.apply_alaw:
            x = apply_alaw_compression(x)

        # 3. 20 ms packet loss
        if self.config.packet_loss_prob > 0.0:
            x = apply_packet_loss(
                x,
                chunk_samples=int(self.config.sample_rate * self.config.packet_loss_chunk_sec),
                drop_prob=self.config.packet_loss_prob,
                rng=self.rng,
            )

        # 4. Realistic room/ambient noise
        x = self.rir_manager.add_realistic_noise(x, snr_db=snr_db, rng=self.rng)
        return x

    def synthesize_dialogue(
        self,
        utterance_pool: Optional[Dict[str, List[np.ndarray]]] = None,
        duration_sec: float = 12.0,
        scenario: str = "normal",
        leak_delay_sec: Optional[float] = None,
        leak_attenuation_db: Optional[float] = None,
        third_speaker_prob: float = 0.5,
    ) -> SynthesisResult:
        """Synthesize a complete two-channel dialogue with ground-truth labels.

        Scenarios:
            - "normal": turn-taking, moderate leak (15-25 dB), low overlap
            - "double_talk": simultaneous speech between A and B
            - "high_leak_lt_10db": severe acoustic/electric leak (6-9 dB attenuation)
            - "path_drift": leak delay tau drifts dynamically during call
            - "third_speaker": third person talks near microphone A or B
        """
        sr = self.config.sample_rate
        total_samples = int(sr * duration_sec)
        num_frames = int(round(duration_sec / self.config.sample_rate * sr / FRAME_DURATION_SEC))
        total_frames = int(math.ceil(duration_sec / FRAME_DURATION_SEC))

        # 1. Parameter resolution
        if leak_delay_sec is None:
            leak_delay_sec = float(self.rng.uniform(*self.config.leak_delay_range_sec))

        if leak_attenuation_db is None:
            if scenario == "high_leak_lt_10db":
                leak_attenuation_db = float(self.rng.uniform(6.0, 9.5))
            else:
                leak_attenuation_db = float(self.rng.uniform(15.0, 28.0))

        # 2. Turn schedule generation
        timeline_a = np.zeros(total_samples, dtype=np.float32)
        timeline_b = np.zeros(total_samples, dtype=np.float32)
        timeline_3 = np.zeros(total_samples, dtype=np.float32)

        gt_intervals: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}

        # Helper to get or synthesize an utterance
        def get_utterance(spk: str, dur: float) -> np.ndarray:
            if utterance_pool and spk in utterance_pool and len(utterance_pool[spk]) > 0:
                candidates = utterance_pool[spk]
                utt = candidates[self.rng.randint(len(candidates))].copy()

                # Speed / pitch perturbation to diversify Russian speaker timbres.
                if self.rng.rand() < 0.70:
                    factor = float(self.rng.uniform(0.88, 1.12))
                    num_resampled = int(round(len(utt) / factor))
                    if num_resampled > 160:
                        utt = signal.resample(utt, num_resampled).astype(np.float32)

                target_len = int(sr * dur)
                if len(utt) >= target_len:
                    return utt[:target_len].copy()
                else:
                    return np.pad(utt, (0, target_len - len(utt)))
            # Default synthetic vowel excitation
            pitch = 120.0 if spk == "A" else (210.0 if spk == "B" else 165.0)
            return self.generate_synthetic_speech_utterance(dur, pitch_hz=pitch)

        # Build turn-taking schedule
        curr_time = 0.5  # initial quiet pause
        while curr_time < (duration_sec - 1.5):
            dur_a = float(self.rng.uniform(1.2, 2.5))
            if curr_time + dur_a > duration_sec - 0.5:
                dur_a = max(0.5, duration_sec - 0.5 - curr_time)

            # Speaker A turn
            s_idx = int(curr_time * sr)
            e_idx = min(total_samples, s_idx + int(dur_a * sr))
            actual_dur = (e_idx - s_idx) / sr

            utt_a = get_utterance("A", actual_dur)
            seg_len_a = min(len(utt_a), e_idx - s_idx)
            timeline_a[s_idx : s_idx + seg_len_a] += utt_a[:seg_len_a]

            start_a_sec = round(curr_time, 4)
            end_a_sec = round(curr_time + actual_dur, 4)
            gt_intervals["A"].append({"start": start_a_sec, "end": end_a_sec, "label": LABEL_OWN, "score": 1.0})

            # Transition to B
            if scenario == "double_talk":
                # Significant overlap: B interrupts A before A finishes
                overlap = float(self.rng.uniform(0.6, 1.4))
                curr_time = max(0.2, curr_time + actual_dur - overlap)
            else:
                pause = float(self.rng.uniform(0.3, 0.8))
                curr_time = curr_time + actual_dur + pause

            if curr_time >= (duration_sec - 1.0):
                break

            # Speaker B turn
            dur_b = float(self.rng.uniform(1.2, 2.5))
            if curr_time + dur_b > duration_sec - 0.5:
                dur_b = max(0.5, duration_sec - 0.5 - curr_time)

            s_idx_b = int(curr_time * sr)
            e_idx_b = min(total_samples, s_idx_b + int(dur_b * sr))
            actual_dur_b = (e_idx_b - s_idx_b) / sr

            utt_b = get_utterance("B", actual_dur_b)
            seg_len_b = min(len(utt_b), e_idx_b - s_idx_b)
            timeline_b[s_idx_b : s_idx_b + seg_len_b] += utt_b[:seg_len_b]

            start_b_sec = round(curr_time, 4)
            end_b_sec = round(curr_time + actual_dur_b, 4)
            gt_intervals["B"].append({"start": start_b_sec, "end": end_b_sec, "label": LABEL_OWN, "score": 1.0})

            if scenario == "double_talk":
                overlap = float(self.rng.uniform(0.5, 1.2))
                curr_time = max(0.2, curr_time + actual_dur_b - overlap)
            else:
                pause = float(self.rng.uniform(0.3, 0.8))
                curr_time = curr_time + actual_dur_b + pause

        # Add Third Speaker (if requested or in third_speaker scenario)
        has_third = (scenario == "third_speaker") or (self.rng.rand() < third_speaker_prob)
        if has_third and duration_sec > 4.0:
            start_third_sec = float(self.rng.uniform(1.5, max(2.0, duration_sec - 3.0)))
            dur_third = float(self.rng.uniform(1.0, 2.2))
            s_idx_3 = int(start_third_sec * sr)
            e_idx_3 = min(total_samples, s_idx_3 + int(dur_third * sr))
            actual_dur_3 = (e_idx_3 - s_idx_3) / sr

            utt_3 = get_utterance("third", actual_dur_3)
            seg_len_3 = min(len(utt_3), e_idx_3 - s_idx_3)
            timeline_3[s_idx_3 : s_idx_3 + seg_len_3] += utt_3[:seg_len_3]

            # Third speaker talks near Track A
            gt_intervals["A"].append({
                "start": round(start_third_sec, 4),
                "end": round(start_third_sec + actual_dur_3, 4),
                "label": LABEL_THIRD,
                "score": 1.0,
            })

        # Keep pure references s_A, s_B
        clean_s_a = timeline_a.copy()
        clean_s_b = timeline_b.copy()

        # 3. Acoustic modeling: RIR convolution
        # Room response for owner A & B (high DRR, close-talk mic)
        rir_owner_a = self.rir_manager.get_owner_rir(rng=self.rng)
        rir_owner_b = self.rir_manager.get_owner_rir(rng=self.rng)
        rec_a = signal.fftconvolve(timeline_a, rir_owner_a, mode="same")
        rec_b = signal.fftconvolve(timeline_b, rir_owner_b, mode="same")

        # Third speaker RIR: more distant and reverberant (real office / conference room)
        rir_third = self.rir_manager.get_third_speaker_rir(rng=self.rng)
        rec_third = signal.fftconvolve(timeline_3, rir_third, mode="same")
        # Attenuate third speaker relative to owner
        third_gain = 10.0 ** (-self.config.third_speaker_attenuation_db / 20.0)
        rec_third = rec_third * third_gain

        # 4. Leakage modeling: Speaker B leaks into Track A (and vice-versa)
        rir_leak = self.rir_manager.get_leakage_rir(rng=self.rng)

        # Speaker saturation nonlinearity on counterpart signal
        sig_b_speaker = apply_speaker_nonlinearity(rec_b, saturation_factor=1.8)
        sig_b_coupled = signal.fftconvolve(sig_b_speaker, rir_leak, mode="same")

        # Delay & drift calculation for B -> A leak
        base_delay_samples = leak_delay_sec * sr
        if scenario == "path_drift" or self.config.apply_drift:
            # Linear drift: e.g. +2 ms per second
            drift_rate_samples_per_sample = (self.config.drift_speed_ms_per_sec / 1000.0)
            delay_profile = base_delay_samples + np.arange(total_samples) * drift_rate_samples_per_sample
        else:
            delay_profile = np.full(total_samples, base_delay_samples, dtype=np.float32)

        leak_b_to_a = apply_time_varying_delay(sig_b_coupled, delay_profile)

        # Attenuation of leak
        leak_scale = 10.0 ** (-leak_attenuation_db / 20.0)
        leak_b_to_a = leak_b_to_a * leak_scale

        # Build ground truth leak intervals on Track A
        # For each segment of B, shift by tau
        for seg_b in [s for s in gt_intervals["B"] if s["label"] == LABEL_OWN]:
            leak_start = round(seg_b["start"] + leak_delay_sec, 4)
            leak_end = round(seg_b["end"] + leak_delay_sec, 4)
            if leak_start < duration_sec:
                gt_intervals["A"].append({
                    "start": leak_start,
                    "end": min(round(duration_sec, 4), leak_end),
                    "label": LABEL_LEAK,
                    "score": 1.0,
                })

        # Mirror leak: A leaks into Track B (with moderate 20 dB attenuation)
        sig_a_speaker = apply_speaker_nonlinearity(rec_a, saturation_factor=1.5)
        sig_a_coupled = signal.fftconvolve(sig_a_speaker, rir_leak, mode="same")
        delay_a_to_b = np.full(total_samples, 0.040 * sr, dtype=np.float32)
        leak_a_to_b = apply_time_varying_delay(sig_a_coupled, delay_a_to_b) * (10.0 ** (-20.0 / 20.0))

        for seg_a in [s for s in gt_intervals["A"] if s["label"] == LABEL_OWN]:
            l_start = round(seg_a["start"] + 0.040, 4)
            l_end = round(seg_a["end"] + 0.040, 4)
            if l_start < duration_sec:
                gt_intervals["B"].append({
                    "start": l_start,
                    "end": min(round(duration_sec, 4), l_end),
                    "label": LABEL_LEAK,
                    "score": 1.0,
                })

        # 5. Acoustic summation at microphone:
        # Track A mic: own speech rec_a + leak_b_to_a + rec_third
        # Track B mic: own speech rec_b + leak_a_to_b
        mic_a = rec_a + leak_b_to_a + rec_third
        mic_b = rec_b + leak_a_to_b

        # 6. Telephone narrowband transmission channel T
        snr = float(self.rng.uniform(*self.config.snr_db_range))
        track_a = self.simulate_telephone_channel(mic_a, snr_db=snr)
        track_b = self.simulate_telephone_channel(mic_b, snr_db=snr)

        stereo = np.stack([track_a, track_b], axis=0).astype(np.float32)

        # Sort all ground truth intervals by start time
        gt_intervals["A"].sort(key=lambda x: x["start"])
        gt_intervals["B"].sort(key=lambda x: x["start"])

        metadata = {
            "scenario": scenario,
            "leak_delay_sec": leak_delay_sec,
            "leak_attenuation_db": leak_attenuation_db,
            "snr_db": snr,
            "has_third_speaker": has_third,
            "duration_sec": duration_sec,
        }

        leak_filt_a = apply_bandpass_filter(leak_b_to_a, sample_rate=sr)
        leak_filt_b = apply_bandpass_filter(leak_a_to_b, sample_rate=sr)
        target_a = (track_a - leak_filt_a).astype(np.float32)
        target_b = (track_b - leak_filt_b).astype(np.float32)

        return SynthesisResult(
            audio=stereo,
            sample_rate=sr,
            duration_sec=duration_sec,
            num_frames=total_frames,
            ground_truth_intervals=gt_intervals,
            clean_references={"A": clean_s_a, "B": clean_s_b},
            scenario=scenario,
            metadata=metadata,
            target_audio={"A": target_a, "B": target_b},
        )

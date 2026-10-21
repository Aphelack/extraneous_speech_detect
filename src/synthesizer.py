"""Telephone signal and dialogue synthesizer.

Implements the physical transmission and leakage model described in the
specification:

- Dialogue between two participants (A and B) with a consistent speaker per channel
- Episodic third speaker (more distant/reverberant RIR, lower amplitude),
  placed randomly on channel A, channel B, or both
- Symmetric leakage paths h_BA and h_AB:
  - Acoustic coupling / RIR
  - Delay tau in [20, 300] ms with independent drift tau(t) per direction
  - Attenuation in [6, 40] dB
  - Loudspeaker non-linearity phi (soft clipping / saturation)
- Narrowband telephone channel T:
  - Bandpass filtering 300 - 3400 Hz
  - G.711 A-law companding & 8-bit quantization
  - Random 20 ms packet loss (PLC simulation)
  - Additive ambient noise n(t) at target SNR
- Ground truth is derived from the *actual audible energy* of each generated
  component on the 10 ms grid, not from the scheduling rectangles. This makes
  the labels automatically account for attenuation, delay drift, pauses inside
  utterances, and the noise floor of each track.
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
    FRAME_DURATION_SEC,
    FRAME_SAMPLES,
    LABEL_LEAK,
    LABEL_OWN,
    LABEL_THIRD,
    LEAK_DELAY_MAX_SAMPLES,
    LEAK_DELAY_MIN_SAMPLES,
    PACKET_LOSS_CHUNK_SAMPLES,
    PACKET_LOSS_CHUNK_SEC,
    SAMPLE_RATE,
    SynthesisConfig,
)
from src.metrics import frame_mask_to_intervals

# A component is considered audible in a frame when its energy exceeds the
# local noise energy by this linear factor and is above the absolute floor.
AUDIBILITY_NOISE_RATIO: float = 1.5
AUDIBILITY_ABS_DBFS: float = -52.0


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

    decay = np.exp(-6.908 * t / max(t60_sec, 0.01))
    noise = rng.randn(num_samples).astype(np.float32)
    reverb = noise * decay

    direct = np.zeros(num_samples, dtype=np.float32)
    direct[0] = 1.0

    dr_linear = 10.0 ** (direct_to_reverberant_ratio_db / 20.0)
    reverb_norm = np.linalg.norm(reverb) + 1e-10
    rir = direct + (reverb / reverb_norm) * (1.0 / dr_linear)

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

    def generate_noise(
        self,
        n_samples: int,
        snr_db: float,
        reference: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Generate additive noise scaled to the requested SNR w.r.t. ``reference``.

        Returns the noise component alone (not the mixture), so that the
        synthesizer can compute the true per-frame noise floor used for
        audibility-based ground truth.
        """
        if rng is None:
            rng = np.random.RandomState()

        ref_power = float(np.mean(reference ** 2)) + 1e-12
        target_noise_power = ref_power / (10.0 ** (snr_db / 10.0))

        if self.available and self._noise_files:
            p = self._noise_files[rng.randint(len(self._noise_files))]
            try:
                import soundfile as sf
                info = sf.info(str(p))
                needed_samples = int(n_samples * (info.samplerate / self.target_sr))
                max_start = max(0, info.frames - needed_samples)
                start_frame = rng.randint(0, max_start + 1) if max_start > 0 else 0
                noise_slice, n_sr = sf.read(str(p), start=start_frame, frames=needed_samples)
                if noise_slice.ndim > 1:
                    noise_slice = noise_slice[:, 0]
                if n_sr != self.target_sr:
                    noise_slice = signal.resample_poly(noise_slice, self.target_sr // 1000, n_sr // 1000)
                if len(noise_slice) < n_samples:
                    noise_slice = np.pad(noise_slice, (0, n_samples - len(noise_slice)), mode="wrap")
                else:
                    noise_slice = noise_slice[:n_samples]

                noise_slice = apply_bandpass_filter(noise_slice, sample_rate=self.target_sr)
                noise_power = float(np.mean(noise_slice ** 2)) + 1e-12
                return (noise_slice * np.sqrt(target_noise_power / noise_power)).astype(np.float32)
            except Exception:
                pass

        raw_noise = rng.randn(n_samples).astype(np.float32)
        filtered = apply_bandpass_filter(raw_noise, sample_rate=self.target_sr)
        filtered_power = float(np.mean(filtered ** 2)) + 1e-12
        return (filtered * np.sqrt(target_noise_power / filtered_power)).astype(np.float32)

    def add_realistic_noise(
        self,
        x: np.ndarray,
        snr_db: float = 25.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        noise = self.generate_noise(len(x), snr_db=snr_db, reference=x, rng=rng)
        return (x + noise).astype(np.float32)


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

    low_mask = abs_x < (1.0 / a_param)
    compressed = np.zeros_like(flat_x)
    compressed[low_mask] = (a_param * abs_x[low_mask]) / denom

    high_mask = ~low_mask
    safe_high = np.maximum(a_param * abs_x[high_mask], 1e-12)
    compressed[high_mask] = (1.0 + np.log(safe_high)) / denom

    quantized = np.round(compressed * 127.0) / 127.0

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
    filtered_noise = apply_bandpass_filter(raw_noise, sample_rate=sample_rate)
    filtered_noise_power = np.mean(filtered_noise ** 2) + 1e-12
    scaled_noise = filtered_noise * np.sqrt(noise_power / filtered_noise_power)

    return (x + scaled_noise).astype(np.float32)


def frame_energy_db(x: np.ndarray, num_frames: int, hop: int = FRAME_SAMPLES) -> np.ndarray:
    """Per-frame mean-square energy in dBFS on the 10 ms grid."""
    needed = num_frames * hop
    if len(x) < needed:
        x = np.pad(x, (0, needed - len(x)), mode="constant")
    else:
        x = x[:needed]
    frames = x.reshape(num_frames, hop)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def audible_mask(
    component: np.ndarray,
    noise: np.ndarray,
    num_frames: int,
    hop: int = FRAME_SAMPLES,
) -> np.ndarray:
    """Frame mask where ``component`` is audible above ``noise``.

    A frame is audible when the component energy exceeds the noise energy by
    ``AUDIBILITY_NOISE_RATIO`` and is above ``AUDIBILITY_ABS_DBFS``.
    """
    comp_db = frame_energy_db(component, num_frames, hop)
    noise_db = frame_energy_db(noise, num_frames, hop)
    noise_energy = 10.0 ** (noise_db / 10.0)
    comp_energy = 10.0 ** (comp_db / 10.0)
    return (comp_energy > noise_energy * AUDIBILITY_NOISE_RATIO) & (comp_db > AUDIBILITY_ABS_DBFS)


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

    # ------------------------------------------------------------------
    # Speech material
    # ------------------------------------------------------------------
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

        period = int(sr / pitch_hz)
        excitation = np.zeros(num_samples, dtype=np.float32)
        excitation[::period] = 1.0

        if formants is None:
            formants = [500.0, 1500.0, 2500.0]

        filtered = excitation.copy()
        for f in formants:
            b, a = signal.iirpeak(f, Q=5.0, fs=sr)
            filtered = signal.lfilter(b, a, filtered)

        env = np.hanning(min(num_samples, int(sr * 0.05)))
        envelope = np.ones(num_samples, dtype=np.float32)
        n_fade = len(env) // 2
        envelope[:n_fade] = env[:n_fade]
        envelope[-n_fade:] = env[-n_fade:]

        out = filtered * envelope
        max_val = np.max(np.abs(out)) + 1e-6
        return (out / max_val * 0.6).astype(np.float32)

    def _get_utterance(
        self,
        utterance_pool: Optional[Dict[str, List[np.ndarray]]],
        speaker: str,
        duration_sec: float,
    ) -> np.ndarray:
        sr = self.config.sample_rate
        target_len = max(1, int(sr * duration_sec))

        if utterance_pool and speaker in utterance_pool and len(utterance_pool[speaker]) > 0:
            candidates = utterance_pool[speaker]
            utt = candidates[self.rng.randint(len(candidates))].copy()

            # Speed / pitch perturbation to diversify speaker timbres.
            if self.rng.rand() < 0.70:
                factor = float(self.rng.uniform(0.88, 1.12))
                num_resampled = int(round(len(utt) / factor))
                if num_resampled > 160:
                    utt = signal.resample(utt, num_resampled).astype(np.float32)

            if len(utt) >= target_len:
                return utt[:target_len].copy()
            return np.pad(utt, (0, target_len - len(utt)))

        pitch = 120.0 if speaker == "A" else (210.0 if speaker == "B" else 165.0)
        return self.generate_synthetic_speech_utterance(duration_sec, pitch_hz=pitch)

    # ------------------------------------------------------------------
    # Dialogue schedule
    # ------------------------------------------------------------------
    def _build_schedule(self, duration_sec: float) -> List[Tuple[str, float, float]]:
        """Random turn-taking schedule with random first speaker and interruptions.

        Utterance lengths are drawn from 0.4 .. 4.5 s, pauses are short, and
        the other participant frequently interrupts before the current turn
        ends (creating genuine double-talk).
        """
        events: List[Tuple[str, float, float]] = []
        speaker = "A" if self.rng.rand() < 0.5 else "B"
        t = float(self.rng.uniform(0.15, 0.8))
        limit = duration_sec - 0.25

        while t < limit:
            dur = float(self.rng.uniform(0.4, 4.5))
            dur = min(dur, limit - t)
            if dur < 0.3:
                break
            events.append((speaker, t, dur))
            end = t + dur

            other = "B" if speaker == "A" else "A"
            if self.rng.rand() < 0.40:
                # Interruption: the other speaker starts before this turn ends.
                max_overlap = min(1.4, dur * 0.7)
                overlap = float(self.rng.uniform(0.15, max_overlap)) if max_overlap > 0.15 else 0.15
                next_start = max(t + 0.2, end - overlap)
                speaker = other
                t = next_start
            else:
                speaker = other
                t = end + float(self.rng.uniform(0.05, 0.6))

        return events

    # ------------------------------------------------------------------
    # Channel model
    # ------------------------------------------------------------------
    @staticmethod
    def _convolve_causal(x: np.ndarray, rir: np.ndarray, total_samples: int) -> np.ndarray:
        """Causal convolution with an RIR, truncated/padded to ``total_samples``.

        ``mode="same"`` would centre the impulse response and shift the signal
        earlier in time, which corrupts the ground-truth timing. A causal
        convolution keeps the direct path at the original sample index.
        """
        out = signal.fftconvolve(x, rir, mode="full")[:total_samples]
        if len(out) < total_samples:
            out = np.pad(out, (0, total_samples - len(out)))
        return out.astype(np.float32)

    def _transmit(self, x: np.ndarray) -> np.ndarray:
        """Telephone transmission chain without additive noise."""
        out = apply_bandpass_filter(x, sample_rate=self.config.sample_rate)
        if self.config.apply_alaw:
            out = apply_alaw_compression(out)
        if self.config.packet_loss_prob > 0.0:
            out = apply_packet_loss(
                out,
                chunk_samples=int(self.config.sample_rate * self.config.packet_loss_chunk_sec),
                drop_prob=self.config.packet_loss_prob,
                rng=self.rng,
            )
        return out

    def simulate_telephone_channel(
        self,
        signal_clean: np.ndarray,
        snr_db: float = 25.0,
    ) -> np.ndarray:
        """Apply telephone transmission chain T(x): Bandpass -> A-law -> Packet loss -> Noise."""
        x = self._transmit(signal_clean)
        return self.rir_manager.add_realistic_noise(x, snr_db=snr_db, rng=self.rng)

    def _sample_attenuation_db(self, scenario: str) -> float:
        """Sample leak attenuation covering the full 6-40 dB range."""
        if scenario == "high_leak_lt_10db":
            return float(self.rng.uniform(6.0, 9.5))
        if scenario == "path_drift":
            return float(self.rng.uniform(8.0, 35.0))
        if scenario == "double_talk":
            return float(self.rng.uniform(10.0, 30.0))
        if scenario == "third_speaker":
            return float(self.rng.uniform(10.0, 32.0))
        # normal / no_leak: mix of moderate and extreme attenuation so that the
        # 9.5-15 dB and 28-40 dB bands are represented too.
        if self.rng.rand() < 0.35:
            return float(self.rng.uniform(6.0, 40.0))
        return float(self.rng.uniform(12.0, 30.0))

    def _build_delay_profile(
        self,
        base_delay_samples: float,
        total_samples: int,
        drift_enabled: bool,
    ) -> np.ndarray:
        """Delay track in samples, clamped to the physical [20, 300] ms range."""
        if drift_enabled:
            drift_ms_per_sec = float(self.rng.uniform(0.5, 4.0)) * (1.0 if self.rng.rand() < 0.5 else -1.0)
            rate = drift_ms_per_sec / 1000.0
            profile = base_delay_samples + np.arange(total_samples, dtype=np.float64) * rate
        else:
            profile = np.full(total_samples, base_delay_samples, dtype=np.float64)
        return np.clip(profile, LEAK_DELAY_MIN_SAMPLES, LEAK_DELAY_MAX_SAMPLES).astype(np.float32)

    # ------------------------------------------------------------------
    # Main synthesis
    # ------------------------------------------------------------------
    def synthesize_dialogue(
        self,
        utterance_pool: Optional[Dict[str, List[np.ndarray]]] = None,
        duration_sec: float = 12.0,
        scenario: str = "normal",
        leak_delay_sec: Optional[float] = None,
        leak_attenuation_db: Optional[float] = None,
        third_speaker_prob: float = 0.5,
    ) -> SynthesisResult:
        """Synthesize a complete two-channel dialogue with energy-based ground truth.

        Scenarios:
            - "normal": turn-taking, moderate leak, low overlap
            - "double_talk": frequent simultaneous speech between A and B
            - "high_leak_lt_10db": severe acoustic/electric leak (6-9.5 dB)
            - "path_drift": leak delay tau drifts dynamically during the call
            - "third_speaker": third person talks near microphone A and/or B
            - "no_leak": no leakage at all on either track
        """
        sr = self.config.sample_rate
        total_samples = int(sr * duration_sec)
        total_frames = int(round(duration_sec / FRAME_DURATION_SEC))
        hop = FRAME_SAMPLES

        # 1. Turn schedule and clean timelines.
        timeline_a = np.zeros(total_samples, dtype=np.float32)
        timeline_b = np.zeros(total_samples, dtype=np.float32)
        timeline_3 = np.zeros(total_samples, dtype=np.float32)

        for speaker, start_sec, dur_sec in self._build_schedule(duration_sec):
            s_idx = int(start_sec * sr)
            e_idx = min(total_samples, s_idx + int(dur_sec * sr))
            if e_idx <= s_idx:
                continue
            utt = self._get_utterance(utterance_pool, speaker, (e_idx - s_idx) / sr)
            seg_len = min(len(utt), e_idx - s_idx)
            target = timeline_a if speaker == "A" else timeline_b
            target[s_idx : s_idx + seg_len] += utt[:seg_len]

        # 2. Third speaker: random channel(s), random attenuation 3-15 dB.
        has_third = (scenario == "third_speaker") or (self.rng.rand() < third_speaker_prob)
        third_channels: List[str] = []
        third_segments: List[Tuple[str, float, float]] = []
        if has_third and duration_sec > 3.0:
            if scenario == "third_speaker":
                third_channels = [str(self.rng.choice(["A", "B"]))]
                if self.rng.rand() < 0.35:
                    third_channels.append("B" if third_channels[0] == "A" else "A")
            else:
                third_channels = [str(self.rng.choice(["A", "B"]))]

            for ch in third_channels:
                start_third = float(self.rng.uniform(0.8, max(1.0, duration_sec - 2.0)))
                dur_third = float(self.rng.uniform(0.6, 2.5))
                s_idx = int(start_third * sr)
                e_idx = min(total_samples, s_idx + int(dur_third * sr))
                if e_idx <= s_idx:
                    continue
                utt = self._get_utterance(utterance_pool, "third", (e_idx - s_idx) / sr)
                seg_len = min(len(utt), e_idx - s_idx)
                timeline_3[s_idx : s_idx + seg_len] += utt[:seg_len]
                third_segments.append((ch, start_third, (e_idx - s_idx) / sr))

        clean_s_a = timeline_a.copy()
        clean_s_b = timeline_b.copy()

        # 3. Acoustic modeling: RIR convolution.
        rir_owner_a = self.rir_manager.get_owner_rir(rng=self.rng)
        rir_owner_b = self.rir_manager.get_owner_rir(rng=self.rng)
        rec_a = self._convolve_causal(timeline_a, rir_owner_a, total_samples)
        rec_b = self._convolve_causal(timeline_b, rir_owner_b, total_samples)

        # Third speaker: independent distant RIR per channel, 3-15 dB below owner.
        rec_third_a = np.zeros(total_samples, dtype=np.float32)
        rec_third_b = np.zeros(total_samples, dtype=np.float32)
        third_gains: Dict[str, float] = {}
        for ch in third_channels:
            rir_third = self.rir_manager.get_third_speaker_rir(rng=self.rng)
            rec_third = self._convolve_causal(timeline_3, rir_third, total_samples)
            atten_db = float(self.rng.uniform(3.0, 15.0))
            third_gains[ch] = atten_db
            rec_third = rec_third * (10.0 ** (-atten_db / 20.0))
            if ch == "A":
                rec_third_a = rec_third
            else:
                rec_third_b = rec_third

        # 4. Symmetric leakage paths B -> A and A -> B.
        if scenario == "no_leak":
            leak_a_enabled = False
            leak_b_enabled = False
        else:
            leak_a_enabled = True
            leak_b_enabled = True
            # 20% of records have no leak on one or both tracks.
            if self.rng.rand() < 0.20:
                mode = str(self.rng.choice(["A", "B", "both"]))
                if mode in ("A", "both"):
                    leak_a_enabled = False
                if mode in ("B", "both"):
                    leak_b_enabled = False

        if leak_delay_sec is None:
            leak_delay_sec = float(self.rng.uniform(*self.config.leak_delay_range_sec))
        if leak_attenuation_db is None:
            leak_attenuation_db = self._sample_attenuation_db(scenario)

        drift_prob = 0.75 if scenario == "path_drift" else 0.25

        leak_b_to_a = np.zeros(total_samples, dtype=np.float32)
        leak_a_to_b = np.zeros(total_samples, dtype=np.float32)
        leak_delay_a = leak_delay_sec
        leak_delay_b = leak_delay_sec
        leak_atten_a = leak_attenuation_db
        leak_atten_b = leak_attenuation_db

        if leak_a_enabled:
            rir_leak_ba = self.rir_manager.get_leakage_rir(rng=self.rng)
            sig_b_speaker = apply_speaker_nonlinearity(rec_b, saturation_factor=1.8)
            sig_b_coupled = self._convolve_causal(sig_b_speaker, rir_leak_ba, total_samples)
            delay_profile = self._build_delay_profile(
                leak_delay_sec * sr, total_samples, drift_enabled=self.rng.rand() < drift_prob
            )
            leak_b_to_a = apply_time_varying_delay(sig_b_coupled, delay_profile)
            leak_b_to_a = leak_b_to_a * (10.0 ** (-leak_attenuation_db / 20.0))

        if leak_b_enabled:
            rir_leak_ab = self.rir_manager.get_leakage_rir(rng=self.rng)
            sig_a_speaker = apply_speaker_nonlinearity(rec_a, saturation_factor=1.5)
            sig_a_coupled = self._convolve_causal(sig_a_speaker, rir_leak_ab, total_samples)
            # Independent delay and attenuation for the opposite direction.
            leak_delay_b = float(self.rng.uniform(*self.config.leak_delay_range_sec))
            leak_atten_b = self._sample_attenuation_db(scenario)
            delay_profile_b = self._build_delay_profile(
                leak_delay_b * sr, total_samples, drift_enabled=self.rng.rand() < drift_prob
            )
            leak_a_to_b = apply_time_varying_delay(sig_a_coupled, delay_profile_b)
            leak_a_to_b = leak_a_to_b * (10.0 ** (-leak_atten_b / 20.0))

        # 5. Acoustic summation at each microphone.
        mic_a = rec_a + leak_b_to_a + rec_third_a
        mic_b = rec_b + leak_a_to_b + rec_third_b

        # 6. Telephone transmission + independent noise per track.
        snr_a = float(self.rng.uniform(*self.config.snr_db_range))
        snr_b = float(self.rng.uniform(*self.config.snr_db_range))

        tx_a = self._transmit(mic_a)
        tx_b = self._transmit(mic_b)
        noise_a = self.rir_manager.generate_noise(total_samples, snr_db=snr_a, reference=tx_a, rng=self.rng)
        noise_b = self.rir_manager.generate_noise(total_samples, snr_db=snr_b, reference=tx_b, rng=self.rng)

        track_a = (tx_a + noise_a).astype(np.float32)
        track_b = (tx_b + noise_b).astype(np.float32)
        stereo = np.stack([track_a, track_b], axis=0).astype(np.float32)

        # 7. Ground truth from the actual audible energy of each component.
        # Components are band-limited to the telephone band so their levels are
        # directly comparable with the additive noise floor of the track.
        comp_own_a = apply_bandpass_filter(rec_a, sample_rate=sr)
        comp_own_b = apply_bandpass_filter(rec_b, sample_rate=sr)
        comp_leak_ba = apply_bandpass_filter(leak_b_to_a, sample_rate=sr)
        comp_leak_ab = apply_bandpass_filter(leak_a_to_b, sample_rate=sr)
        comp_third_a = apply_bandpass_filter(rec_third_a, sample_rate=sr)
        comp_third_b = apply_bandpass_filter(rec_third_b, sample_rate=sr)

        masks_a = {
            LABEL_OWN: audible_mask(comp_own_a, noise_a, total_frames, hop),
            LABEL_LEAK: audible_mask(comp_leak_ba, noise_a, total_frames, hop),
            LABEL_THIRD: audible_mask(comp_third_a, noise_a, total_frames, hop),
        }
        masks_b = {
            LABEL_OWN: audible_mask(comp_own_b, noise_b, total_frames, hop),
            LABEL_LEAK: audible_mask(comp_leak_ab, noise_b, total_frames, hop),
            LABEL_THIRD: audible_mask(comp_third_b, noise_b, total_frames, hop),
        }

        gt_intervals: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
        for ch, masks in (("A", masks_a), ("B", masks_b)):
            for label, mask in masks.items():
                gt_intervals[ch].extend(
                    frame_mask_to_intervals(mask, label=label, frame_sec=FRAME_DURATION_SEC)
                )
            gt_intervals[ch].sort(key=lambda x: (x["start"], x["label"]))

        # 8. Cleaner targets: track minus the band-limited leak component.
        target_a = (track_a - comp_leak_ba).astype(np.float32)
        target_b = (track_b - comp_leak_ab).astype(np.float32)

        metadata = {
            "scenario": scenario,
            "leak_delay_sec": leak_delay_a,
            "leak_delay_b_sec": leak_delay_b,
            "leak_attenuation_db": leak_atten_a,
            "leak_attenuation_b_db": leak_atten_b,
            "leak_enabled_a": leak_a_enabled,
            "leak_enabled_b": leak_b_enabled,
            "snr_db": snr_a,
            "snr_b_db": snr_b,
            "has_third_speaker": has_third,
            "third_channels": third_channels,
            "third_attenuation_db": third_gains,
            "duration_sec": duration_sec,
        }

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

#!/usr/bin/env python3
"""
family_spec_generation — Audio preprocessing and song detection for the scoring pipeline.

This module provides the signal-processing primitives used by
``prepare_batch.py`` to read audio, detect song bouts, and compute
spectrograms suitable for human scoring.

It can also be run standalone to batch-process a collection of audio files,
filter them for song content, and save spectrogram PNGs.

Public API
----------
read_audio_file
    Read ``.cbin``, ``.wav``, or ``soundfile``-compatible audio.
bandpass
    Butterworth band-pass filter.
smooth_envelope
    Rectified, band-passed, smoothed amplitude envelope.
segment_notes
    Threshold-based syllable/note segmentation.
score_song_candidate
    Sliding-window song-detection score for a single file.
make_song_spectrogram
    Normalised log-magnitude STFT suitable for display and ML.

Notes
-----
``.cbin`` support requires the ``evfuncs`` package::

    pip install evfuncs

If ``evfuncs`` is not installed, calling :func:`read_audio_file` on a
``.cbin`` path raises ``ImportError``.  All other formats (WAV, FLAC, OGG,
…) work without it via ``soundfile`` or ``scipy.io.wavfile``.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, butter, filtfilt, get_window, spectrogram

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    sf = None
    _HAS_SF = False

# Optional .cbin support — try installed package first, then local copy.
try:
    from evfuncs import load_cbin
    _HAS_CBIN = True
except ImportError:
    try:
        from tools.evfuncs import load_cbin   # local project copy
        _HAS_CBIN = True
    except ImportError:
        load_cbin  = None
        _HAS_CBIN  = False

EPSILON = 1e-12

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
DEFAULT_SONG_RESULTS_JSON = "file_management/priority_bird_songpaths.json"
DEFAULT_OUTPUT_ROOT       = "file_management/xfoster_specs"
DEFAULT_MANIFEST_PATH     = "file_management/xfoster_specs/spectrogram_manifest.csv"
DEFAULT_RESULTS_JSON      = "file_management/xfoster_specs/spectrogram_results.json"

# Optional aliases for birds that appear under multiple IDs in source systems.
BIRD_ALIASES: Dict[str, List[str]] = {
    "ye1tut0": ["ye1tut0", "ye1", "y1"],
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_song_results(filepath: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a song-results dictionary from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to a JSON file whose top-level keys are bird IDs and whose
        values are lists of ``{"filepath": ..., ...}`` dicts.

    Returns
    -------
    dict
        Loaded data, or an empty dict on failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded song_results from: {filepath}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}


def resolve_audio_path(filepath: str) -> str:
    """Attempt to resolve a stale or relative audio path.

    Checks the path as-is, relative to the current working directory, and
    relative to the directory of this module.  Returns the first resolvable
    path, or the original string if none exist.

    Parameters
    ----------
    filepath : str
        Original (possibly stale) file path.

    Returns
    -------
    str
        Resolved absolute path, or ``filepath`` unchanged if not found.
    """
    p = Path(filepath)
    if p.exists():
        return str(p)

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / p
    if script_candidate.exists():
        return str(script_candidate.resolve())

    return str(p)


def read_audio_file(filepath: str) -> Tuple[np.ndarray, int]:
    """Read a ``.cbin``, ``.wav``, or ``soundfile``-compatible audio file.

    Tries, in order:

    1. ``evfuncs.load_cbin`` for ``.cbin`` files.
    2. ``soundfile.read`` for any format it supports (WAV, FLAC, OGG, …).
    3. ``scipy.io.wavfile.read`` as a final fallback for plain WAV.

    Parameters
    ----------
    filepath : str
        Path to the audio file.  Relative paths are resolved via
        :func:`resolve_audio_path`.

    Returns
    -------
    audio : numpy.ndarray, shape (n_samples,), dtype float64
        Mono audio signal.
    sr : int
        Sample rate in Hz.

    Raises
    ------
    ImportError
        If *filepath* has a ``.cbin`` extension and ``evfuncs`` is not
        installed.
    """
    filepath = resolve_audio_path(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".cbin":
        if not _HAS_CBIN:
            raise ImportError(
                ".cbin files require evfuncs: pip install evfuncs"
            )
        audio, sr = load_cbin(filepath)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float64), int(sr)

    if _HAS_SF:
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio.astype(np.float64), int(sr)
        except Exception:
            pass

    sr, audio = wavfile.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float64), int(sr)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass(
    audio: np.ndarray,
    sr: int,
    low: float = 500.0,
    high: float = 10000.0,
    order: int = 8,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    audio : numpy.ndarray
        Input signal.
    sr : int
        Sample rate in Hz.
    low : float, optional
        Lower cut-off frequency in Hz (default 500).
    high : float, optional
        Upper cut-off frequency in Hz, clamped to ``sr/2 - 1000``
        (default 10000).
    order : int, optional
        Filter order (default 8).

    Returns
    -------
    numpy.ndarray
        Filtered signal, same shape as *audio*.
    """
    nyq  = sr / 2
    high = min(high, nyq - 1000)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, audio)


def smooth_envelope(
    audio: np.ndarray,
    sr: int,
    smooth_ms: float = 2.0,
) -> np.ndarray:
    """Compute a smoothed amplitude envelope of a band-passed signal.

    Applies :func:`bandpass`, squares the result (power), then convolves
    with a rectangular window of length ``smooth_ms`` milliseconds.

    Parameters
    ----------
    audio : numpy.ndarray
        Raw audio signal.
    sr : int
        Sample rate in Hz.
    smooth_ms : float, optional
        Smoothing window duration in milliseconds (default 2.0).

    Returns
    -------
    numpy.ndarray
        Smoothed power envelope, same length as *audio*.
    """
    x   = bandpass(audio, sr)
    x   = x ** 2
    win = max(1, int(round(sr * smooth_ms / 1000.0)))
    return np.convolve(x, np.ones(win) / win, mode="same")


def segment_notes(
    env: np.ndarray,
    sr: int,
    threshold: float,
    min_int_ms: float = 2.0,
    min_dur_ms: float = 30.0,
    max_dur_ms: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect note/syllable boundaries from a smoothed amplitude envelope.

    Segments are defined as contiguous regions where *env* exceeds
    *threshold*.  Short inter-segment gaps are merged; segments outside
    the duration bounds are discarded.

    Parameters
    ----------
    env : numpy.ndarray
        Smoothed amplitude envelope (output of :func:`smooth_envelope`).
    sr : int
        Sample rate of the original audio (used for time conversions).
    threshold : float
        Amplitude threshold.  Samples above this are considered "active".
    min_int_ms : float, optional
        Minimum inter-segment gap in ms; shorter gaps are bridged
        (default 2.0).
    min_dur_ms : float, optional
        Minimum segment duration in ms; shorter segments are discarded
        (removes transient noise, default 30.0).
    max_dur_ms : float, optional
        Maximum segment duration in ms; longer segments are discarded
        (removes sustained calls / tones that are not song syllables,
        default 400.0).

    Returns
    -------
    onsets : numpy.ndarray
        Onset times in seconds.
    offsets : numpy.ndarray
        Offset times in seconds.  Parallel to *onsets*.
    """
    mask  = env > threshold
    trans = np.diff(mask.astype(np.int8))
    on    = np.where(trans == 1)[0] + 1
    off   = np.where(trans == -1)[0] + 1

    if len(on) == 0 or len(off) == 0:
        return np.array([]), np.array([])

    if off[0] < on[0]:
        off = off[1:]
    if len(on) > len(off):
        on = on[:len(off)]
    elif len(off) > len(on):
        off = off[:len(on)]

    keep = [0]
    for i in range(1, len(on)):
        if (on[i] - off[i - 1]) * 1000.0 / sr > min_int_ms:
            keep.append(i)
    on  = on[keep]
    off = off[keep]

    dur_ms = (off - on) * 1000.0 / sr
    mask   = (dur_ms >= min_dur_ms) & (dur_ms <= max_dur_ms)
    return on[mask] / sr, off[mask] / sr


def score_song_candidate(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 2.0,
    min_segments: int = 8,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
    min_windows: int = 3,
) -> Dict[str, Any]:
    """Score an audio recording for song-likeness.

    Uses a sliding window over the detected note segments to find the
    densest bout of activity, then applies two criteria:

    1. The best window must contain at least *min_segments* notes.
    2. At least *min_windows* windows must each contain
       ``ceil(min_segments * 0.6)`` or more notes (sustained activity).

    This two-layer check distinguishes genuine song bouts from isolated
    calls or cage noise that might pass a single-window count threshold.

    Parameters
    ----------
    audio : numpy.ndarray
        Raw mono audio signal.
    sr : int
        Sample rate in Hz.
    window_sec : float, optional
        Analysis window duration in seconds (default 2.0).
    min_segments : int, optional
        Minimum notes required in the best window to pass (default 8).
    step_sec : float, optional
        Sliding window step size in seconds (default 0.25).
    threshold_mode : str, optional
        ``"percentile"`` (default) sets the segmentation threshold at
        30 % of the 90th-percentile envelope amplitude.  Any other
        string uses a fixed threshold of 0.05.
    min_windows : int, optional
        Minimum windows with sustained activity required to pass
        (default 3).

    Returns
    -------
    dict with keys:

    passed : bool
        Whether the file meets the song criterion.
    max_segments : int
        Note count in the best window.
    n_windows_passing : int
        Number of windows that exceeded the soft threshold.
    best_window : tuple[float, float] or None
        ``(start_s, end_s)`` of the densest window, or ``None`` if failed.
    onsets : numpy.ndarray
        Detected note onset times (seconds).
    offsets : numpy.ndarray
        Detected note offset times (seconds).
    threshold : float
        Envelope threshold used for segmentation.
    """
    env = smooth_envelope(audio, sr, smooth_ms=2.0)

    if np.max(env) <= 0:
        return {"passed": False, "max_segments": 0, "best_window": None,
                "onsets": np.array([]), "offsets": np.array([]), "threshold": 0.0}

    env = env / np.max(env)
    threshold = np.percentile(env, 90) * 0.3 if threshold_mode == "percentile" else 0.05
    onsets, offsets = segment_notes(env, sr, threshold)

    if len(onsets) == 0:
        return {"passed": False, "max_segments": 0, "best_window": None,
                "onsets": onsets, "offsets": offsets, "threshold": threshold}

    duration = len(audio) / sr
    if duration <= window_sec:
        count  = len(onsets)
        passed = count >= min_segments
        return {"passed": passed, "max_segments": count,
                "best_window": (0.0, min(window_sec, duration)) if passed else None,
                "onsets": onsets, "offsets": offsets, "threshold": threshold}

    best_count = 0
    best_start = 0.0
    soft_thresh        = int(np.ceil(min_segments * 0.6))
    n_windows_passing  = 0

    for start_t in np.arange(0, duration - window_sec + 1e-9, step_sec):
        end_t   = start_t + window_sec
        count   = int(np.sum((onsets < end_t) & (offsets > start_t)))
        if count > best_count:
            best_count = count
            best_start = start_t
        if count >= soft_thresh:
            n_windows_passing += 1

    passed = (best_count >= min_segments) and (n_windows_passing >= min_windows)
    return {
        "passed":             passed,
        "max_segments":       best_count,
        "n_windows_passing":  n_windows_passing,
        "best_window":        (best_start, best_start + window_sec) if passed else None,
        "onsets":             onsets,
        "offsets":            offsets,
        "threshold":          threshold,
    }


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------

def make_song_spectrogram(
    audio: np.ndarray,
    fs: int,
    nfft: int = 1024,
    hop: int = 1,
    min_freq: float = 400.0,
    max_freq: float = 10000.0,
    p_low: float = 2.0,
    p_high: float = 98.0,
    max_duration_s: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a normalised log-magnitude STFT for birdsong display.

    Audio longer than *max_duration_s* is truncated before the STFT to
    prevent memory errors on session-length recordings.  The output is
    percentile-normalised to ``[0, 1]`` for consistent display across
    files with different recording levels.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio signal (float64 recommended).
    fs : int
        Sample rate in Hz.
    nfft : int, optional
        FFT size (default 1024).
    hop : int, optional
        STFT hop size in samples (default 1; use 256 for batch efficiency).
    min_freq : float, optional
        Lower frequency bound in Hz for the output (default 400).
    max_freq : float, optional
        Upper frequency bound in Hz for the output (default 10000).
    p_low : float, optional
        Lower percentile for contrast normalisation (default 2).
    p_high : float, optional
        Upper percentile for contrast normalisation (default 98).
    max_duration_s : float, optional
        Maximum audio duration before truncation (default 60).

    Returns
    -------
    spec : numpy.ndarray, shape (n_freqs, n_times), float32
        Normalised log-magnitude spectrogram, values in ``[0, 1]``.
        Returns an empty array if *audio* is empty.
    f : numpy.ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    t : numpy.ndarray, shape (n_times,)
        Time axis in seconds.
    """
    if audio is None or len(audio) == 0:
        return np.array([]), np.array([]), np.array([])

    max_samples = int(max_duration_s * fs)
    if len(audio) > max_samples:
        print(f"    truncating audio from {len(audio)/fs:.1f}s "
              f"to {max_duration_s:.0f}s for spectrogram")
        audio = audio[:max_samples]

    w    = get_window("hann", Nx=nfft)
    stft = ShortTimeFFT(w, hop=hop, fs=fs)
    Sx   = stft.stft(audio)
    t    = stft.t(len(audio))
    f    = stft.f

    keep_t = t >= 0
    if len(keep_t) > 0 and nfft // 2 < len(keep_t):
        keep_t[-(nfft // 2):] = False

    t  = t[keep_t]
    Sx = Sx[:, keep_t]

    spec   = np.log(np.abs(Sx) + EPSILON)
    keep_f = (f >= min_freq) & (f <= max_freq)
    f_sel  = f[keep_f]
    spec   = spec[keep_f, :]

    finite = spec[np.isfinite(spec)]
    if finite.size == 0:
        return np.zeros_like(spec), f_sel, t

    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        hi = lo + EPSILON

    return np.clip((spec - lo) / (hi - lo), 0, 1), f_sel, t


def spectrogram_for_plot(
    audio: np.ndarray, sr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a quick diagnostic spectrogram via ``scipy.signal.spectrogram``.

    Parameters
    ----------
    audio : numpy.ndarray
        Mono audio signal.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    f : numpy.ndarray
        Frequency axis in Hz.
    t : numpy.ndarray
        Time axis in seconds.
    Sxx : numpy.ndarray
        Power spectral density in dB.
    """
    f, tt, Sxx = spectrogram(audio, fs=sr, nperseg=1024, noverlap=768)
    return f, tt, 10 * np.log10(Sxx + EPSILON)


# ---------------------------------------------------------------------------
# Optional debugging plot
# ---------------------------------------------------------------------------

def plot_segmentation_summary(
    audio: np.ndarray,
    fs: int,
    onsets: np.ndarray,
    offsets: np.ndarray,
    threshold: Optional[float] = None,
    out_path: Optional[str] = None,
    title: str = "",
) -> None:
    """Plot waveform, envelope, and spectrogram with detected segments overlaid.

    Intended for interactive debugging.  Production spectrograms written by
    :func:`make_song_spectrogram` do **not** show segment overlays.

    Parameters
    ----------
    audio : numpy.ndarray
        Raw audio signal.
    fs : int
        Sample rate in Hz.
    onsets : numpy.ndarray
        Segment onset times in seconds.
    offsets : numpy.ndarray
        Segment offset times in seconds.
    threshold : float, optional
        Envelope threshold to draw as a dashed line.
    out_path : str, optional
        If given, save figure to this path instead of displaying.
    title : str, optional
        Figure title.
    """
    fig = None
    try:
        env = smooth_envelope(audio, fs, smooth_ms=2.0)
        if np.max(env) > 0:
            env = env / np.max(env)

        t  = np.arange(len(audio)) / fs
        te = np.arange(len(env)) / fs

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False)

        axes[0].plot(t, audio)
        for o, off in zip(onsets, offsets):
            axes[0].axvspan(o, off, alpha=0.2)
        axes[0].set_title(title or "Waveform with detected segments")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(te, env)
        if threshold is not None:
            axes[1].axhline(threshold, linestyle="--")
        for o, off in zip(onsets, offsets):
            axes[1].axvspan(o, off, alpha=0.2)
        axes[1].set_title("Smoothed envelope + threshold")
        axes[1].set_ylabel("Normalised envelope")

        f, tt, Sxx = spectrogram_for_plot(audio, fs)
        axes[2].pcolormesh(tt, f, Sxx, shading="auto", cmap="magma")
        for o, off in zip(onsets, offsets):
            axes[2].axvspan(o, off, alpha=0.2)
        axes[2].set_ylim(400, min(10000, fs / 2))
        axes[2].set_title("Spectrogram with detected segments")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=400, bbox_inches="tight")
        else:
            plt.show()
    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-file processing
# ---------------------------------------------------------------------------

def safe_basename(filename: str) -> str:
    """Return a filesystem-safe stem for *filename*."""
    base = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def process_one_file(
    bird: str,
    filepath: str,
    output_root: str,
    window_sec: float = 2.0,
    min_segments: int = 8,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Process a single audio file and return a manifest row.

    Reads the file, scores it with :func:`score_song_candidate`, and — if
    it passes — saves a clean spectrogram PNG (no onset/offset overlays).

    Parameters
    ----------
    bird : str
        Bird identifier string (used as sub-directory name under
        *output_root*).
    filepath : str
        Path to the audio file.
    output_root : str
        Root directory for spectrogram PNG output.
    window_sec : float, optional
        Analysis window in seconds passed to :func:`score_song_candidate`.
    min_segments : int, optional
        Minimum notes required to pass (default 8).
    step_sec : float, optional
        Sliding window step in seconds (default 0.25).
    threshold_mode : str, optional
        Threshold mode passed to :func:`score_song_candidate`
        (default ``"percentile"``).
    overwrite : bool, optional
        Re-generate PNGs even if they already exist (default ``False``).

    Returns
    -------
    dict
        Manifest row with keys ``bird``, ``filepath``, ``output_path``,
        ``status``, ``n_segments``, ``threshold``, ``best_window_start``,
        ``best_window_end``, ``n_onsets``, ``n_offsets``.
        ``status`` is one of ``"song"``, ``"non_song"``,
        ``"skipped_exists"``, or ``"error: <repr>"``.
    """
    fig = None
    try:
        filepath = resolve_audio_path(filepath)
        audio, fs = read_audio_file(filepath)

        score    = score_song_candidate(audio, fs, window_sec=window_sec,
                                        min_segments=min_segments,
                                        step_sec=step_sec,
                                        threshold_mode=threshold_mode)
        passed      = bool(score["passed"])
        n_segments  = int(score["max_segments"])
        threshold   = float(score["threshold"])
        best_window = score["best_window"]
        onsets      = score["onsets"]
        offsets     = score["offsets"]

        bird_dir = os.path.join(output_root, bird)
        os.makedirs(bird_dir, exist_ok=True)

        out_path = os.path.join(bird_dir, safe_basename(filepath) + ".png")

        _base_row = {
            "bird": bird, "filepath": filepath,
            "n_segments": n_segments, "threshold": threshold,
            "best_window_start": best_window[0] if best_window else None,
            "best_window_end":   best_window[1] if best_window else None,
            "n_onsets": int(len(onsets)), "n_offsets": int(len(offsets)),
        }

        if not passed:
            return {**_base_row, "output_path": "", "status": "non_song"}

        if os.path.exists(out_path) and not overwrite:
            return {**_base_row, "output_path": out_path, "status": "skipped_exists"}

        start_t, end_t = best_window
        audio_seg = audio[int(round(start_t * fs)): int(round(end_t * fs))]
        spec, f_sel, _ = make_song_spectrogram(
            audio_seg, fs=fs, nfft=1024, hop=1,
            min_freq=400, max_freq=10000,
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            if spec.size > 0:
                ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                          vmin=0, vmax=1,
                          extent=[0, end_t - start_t, f_sel[0], f_sel[-1]])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim(400, min(10000, fs / 2))
            plt.tight_layout()
            fig.savefig(out_path, dpi=400, bbox_inches="tight")
        finally:
            plt.close(fig)
            fig = None

        return {**_base_row, "output_path": out_path, "status": "song"}

    except Exception as e:
        if fig is not None:
            plt.close(fig)
        return {
            "bird": bird, "filepath": filepath,
            "output_path": "", "status": f"error: {repr(e)}",
            "n_segments": None, "threshold": None,
            "best_window_start": None, "best_window_end": None,
            "n_onsets": None, "n_offsets": None,
        }


def build_spectrogram_pipeline(
    song_results: Dict[str, List[Dict[str, Any]]],
    output_root: Optional[str] = None,
    window_sec: float = 2.0,
    min_segments: int = 8,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
    overwrite: bool = False,
    max_files_per_bird: int = 5,
    use_parallel: bool = False,
    max_workers: Optional[int] = None,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run :func:`process_one_file` over a collection of birds and files.

    Parameters
    ----------
    song_results : dict
        ``{bird_id: [{"filepath": ..., ...}, ...]}`` as loaded by
        :func:`load_song_results`.
    output_root : str, optional
        Root directory for PNG output (default ``./xfoster_specs``).
    window_sec : float, optional
        Analysis window in seconds (default 2.0).
    min_segments : int, optional
        Song criterion threshold (default 8).
    step_sec : float, optional
        Sliding window step in seconds (default 0.25).
    threshold_mode : str, optional
        ``"percentile"`` or ``"fixed"`` (default ``"percentile"``).
    overwrite : bool, optional
        Regenerate existing PNGs (default ``False``).
    max_files_per_bird : int, optional
        Maximum number of files to process per bird (default 5).
    use_parallel : bool, optional
        Use :class:`~concurrent.futures.ProcessPoolExecutor` (default
        ``False``).
    max_workers : int, optional
        Worker count when *use_parallel* is ``True``.
    manifest_path : str, optional
        Override path for the output CSV manifest.

    Returns
    -------
    list[dict]
        One manifest row per processed file (see :func:`process_one_file`).
    """
    if output_root is None:
        output_root = os.path.join(os.getcwd(), "xfoster_specs")
    os.makedirs(output_root, exist_ok=True)

    tasks = [(bird, fi["filepath"])
             for bird, files in song_results.items()
             for fi in files[:max_files_per_bird]]

    results: List[Dict[str, Any]] = []

    if use_parallel and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(process_one_file, bird, fp, output_root,
                          window_sec, min_segments, step_sec,
                          threshold_mode, overwrite)
                for bird, fp in tasks
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for bird, fp in tasks:
            row = process_one_file(bird, fp, output_root, window_sec,
                                   min_segments, step_sec, threshold_mode,
                                   overwrite)
            results.append(row)
            print(f"{bird}: {row['status']} | n_segments={row['n_segments']}")

    if manifest_path is None:
        manifest_path = os.path.join(output_root, "spectrogram_manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bird", "filepath", "output_path", "status",
            "n_segments", "threshold",
            "best_window_start", "best_window_end",
            "n_onsets", "n_offsets",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Total files processed: {len(results)}")
    print(f"Song:           {sum(r['status'] == 'song'          for r in results)}")
    print(f"Non-song:       {sum(r['status'] == 'non_song'      for r in results)}")
    print(f"Skipped:        {sum(r['status'] == 'skipped_exists' for r in results)}")
    print(f"Errors:         {sum(str(r['status']).startswith('error') for r in results)}")
    return results


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_birds_from_txt(txt_path: str) -> List[str]:
    """Read a plain-text bird list (one ID per line)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def expand_bird_aliases(bird: str) -> List[str]:
    """Return all known aliases for *bird*, including itself."""
    return BIRD_ALIASES.get(bird, [bird])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    song_results_path = DEFAULT_SONG_RESULTS_JSON
    output_root       = DEFAULT_OUTPUT_ROOT

    if not os.path.exists(song_results_path):
        raise FileNotFoundError(
            f"Could not find song results JSON: {song_results_path}"
        )

    song_results = load_song_results(song_results_path)
    if not song_results:
        raise RuntimeError("song_results is empty; nothing to process")

    results = build_spectrogram_pipeline(
        song_results=song_results,
        output_root=output_root,
        window_sec=6.0,
        min_segments=8,
        step_sec=0.25,
        threshold_mode="percentile",
        overwrite=True,
        max_files_per_bird=5,
        use_parallel=False,
        max_workers=2,
        manifest_path=DEFAULT_MANIFEST_PATH,
    )

    with open(DEFAULT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to: {DEFAULT_RESULTS_JSON}")


if __name__ == "__main__":
    main()

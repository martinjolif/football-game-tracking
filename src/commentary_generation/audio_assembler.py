"""Assemble TTS waveform segments into a single audio track aligned to video timeline."""

import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def assemble_audio_track(
    segments: list[dict],
    sample_rate: int,
    total_duration_sec: float,
    output_path: str,
) -> bool:
    """Build a WAV file from timestamped waveform segments.

    Args:
        segments: list of {"timestamp_sec": float, "waveform": np.ndarray}
        sample_rate: audio sample rate (Hz)
        total_duration_sec: total video duration in seconds
        output_path: path to write the output WAV

    Returns:
        True on success, False on failure.
    """
    try:
        total_samples = int(total_duration_sec * sample_rate)
        audio = np.zeros(total_samples, dtype=np.float32)

        # Sort segments by timestamp so overlaps are handled predictably
        segments_sorted = sorted(segments, key=lambda s: s["timestamp_sec"])

        for seg in segments_sorted:
            start_sample = int(seg["timestamp_sec"] * sample_rate)
            waveform = seg["waveform"].astype(np.float32)
            end_sample = start_sample + len(waveform)

            if start_sample >= total_samples:
                continue

            # Clip to fit within total duration
            if end_sample > total_samples:
                waveform = waveform[: total_samples - start_sample]
                end_sample = total_samples

            audio[start_sample:end_sample] += waveform

        # Clamp to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio /= peak

        sf.write(output_path, audio, sample_rate)
        logger.info(
            "Assembled audio track: %.1fs, %d segments -> %s",
            total_duration_sec,
            len(segments),
            output_path,
        )
        return True

    except Exception:
        logger.exception("Failed to assemble audio track")
        return False

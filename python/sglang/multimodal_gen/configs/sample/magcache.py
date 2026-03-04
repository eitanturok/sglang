# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
import json
import os
from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


def nearest_interp(data: list[float], target_len: int) -> list[float]:
    """Simple nearest neighbor interpolation for 1D arrays."""
    n = len(data)
    indices = [round(i * (n - 1) / (target_len - 1)) for i in range(target_len)]
    return [data[i] for i in indices]

def get_interpolated_mag_ratios(sample_steps: int, raw_ratios: list[float]) -> list[float]:
    """
    Interpolates magnitude ratios to match the number of sampling steps.
    Returns a flattened list of [cond, uncond] pairs.
    """
    # The original logic assumes ratios are stored as [cond, uncond, cond, uncond...]
    # If the current total length doesn't match steps * 2, interpolate
    if len(raw_ratios) != sample_steps * 2:
        # Separate conditional and unconditional streams
        mag_ratio_con = nearest_interp(raw_ratios[0::2], sample_steps)
        mag_ratio_ucon = nearest_interp(raw_ratios[1::2], sample_steps)

        # Zip them back together and flatten
        return [v for pair in zip(mag_ratio_con, mag_ratio_ucon) for v in pair]
    return raw_ratios


def load_mag_ratios(jsonl_path: str) -> list[float]:
    """Read calibration JSONL and return a list of mag ratios indexed by cnt."""
    records = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    records.sort(key=lambda r: r["cnt"])
    return [r["mag_ratio"] for r in records]


@dataclass
class MagCacheParams(CacheParams):
    """
    MagCache configuration for magnitude-ratio-based caching.

    MagCache accelerates diffusion inference by skipping forward passes when
    magnitude ratios of consecutive residuals are predictably similar.

    Attributes:
        threshold: Accumulated error threshold (default 0.06 from paper).
                   Lower = higher quality but slower. Higher = faster but lower quality.
        max_skip_steps: Maximum consecutive skips allowed (default 3).
                        Prevents infinite skipping even if error is low.
        skip_start_step: Number of denoising steps at the start where skipping is disabled.
        skip_end_step: Number of denoising steps at the end where skipping is disabled (0 = active until last step).
    """

    cache_type: str = "magcache"
    threshold: float = 0.12
    max_skip_steps: int = 4
    skip_start_step: int = 10
    skip_end_step: int = 0
    mag_ratios: list[float] | None = None

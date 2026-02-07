# SPDX-License-Identifier: Apache-2.0
"""Configuration parameters for MagCache optimization."""

import json
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class MagCacheParams(CacheParams):
    """Configuration for MagCache optimization.

    MagCache uses pre-computed magnitude ratios of residual changes to decide
    when to skip computation and reuse cached outputs. This requires a one-time
    calibration phase to compute the magnitude ratios for your model/steps configuration.

    Attributes:
        cache_type: Always "magcache" for this class.
        magnitude_ratios: Pre-computed magnitude ratios per timestep from calibration.
            Length must match num_inference_steps. Each ratio represents
            ||residual_t|| / ||residual_{t-1}||. Lower ratios indicate smaller
            changes, making the step a good candidate for caching.
        skip_threshold: Threshold for skip decision. When magnitude_ratios[t] < threshold,
            skip computation and use cache. Typical values: 0.05-0.15.
            Lower = more aggressive caching (faster, potentially lower quality)
            Higher = more conservative (slower, better quality)
        calibration_file: Path to pre-computed calibration results JSON file.
            If provided, loads magnitude_ratios automatically on initialization.
            Alternative to passing magnitude_ratios directly.
        is_calibration: Whether this is a calibration run. When True, the model
            will collect residuals at each step for magnitude ratio computation
            instead of using them for caching.

    Example:
        # Load from calibration file
        params = MagCacheParams(
            calibration_file="~/.cache/sgl_diffusion/magcache_calibration/wan2.1/50steps.json",
            skip_threshold=0.1
        )

        # Or provide ratios directly
        params = MagCacheParams(
            magnitude_ratios=[0.95, 0.92, 0.88, ...],
            skip_threshold=0.1
        )

        # Calibration mode
        params = MagCacheParams(
            is_calibration=True
        )
    """

    cache_type: str = "magcache"

    # Pre-computed magnitude ratios per timestep
    # Length must match num_inference_steps
    magnitude_ratios: list[float] = field(default_factory=list)

    # Skip threshold: if magnitude ratio < threshold, use cache
    # Typical values: 0.05-0.15
    # Lower = more aggressive caching (faster, potentially lower quality)
    # Higher = more conservative (slower, better quality)
    skip_threshold: float = 0.1

    # Path to calibration file (alternative to passing ratios directly)
    calibration_file: str | None = None

    # Whether this is a calibration run
    is_calibration: bool = False

    def __post_init__(self):
        """Load calibration file if provided and ratios not already set."""
        if self.calibration_file is not None and not self.magnitude_ratios:
            self._load_from_calibration_file()

    def _load_from_calibration_file(self) -> None:
        """Load magnitude ratios from calibration file.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
            KeyError: If calibration file doesn't contain 'magnitude_ratios'.
        """
        if self.calibration_file is None:
            return

        path = os.path.expanduser(self.calibration_file)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibration file not found: {path}\n"
                f"Please run calibration first to generate this file."
            )

        with open(path, "r") as f:
            data = json.load(f)

        if "magnitude_ratios" not in data:
            raise KeyError(
                f"Calibration file missing 'magnitude_ratios' key: {path}\n"
                f"Expected format: {{'magnitude_ratios': [0.95, 0.92, ...], ...}}"
            )

        self.magnitude_ratios = data["magnitude_ratios"]

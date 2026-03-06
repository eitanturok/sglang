# SPDX-License-Identifier: Apache-2.0
"""
MagCache: Magnitude-based caching for diffusion models.

MagCache accelerates diffusion inference by selectively skipping redundant
computation based on the predictable decay pattern of residual magnitudes.
Unlike other methods, it uses pre-computed magnitude ratios from a calibration
phase to make intelligent caching decisions.

Key concepts:
- Magnitude ratio: ||residual_t|| / ||residual_{t-1}|| - exhibits predictable decay
- Calibration: One-time setup to compute magnitude ratios for a model/steps config
- Skip threshold: When magnitude ratio < threshold, reuse cached output
- CFG support: Separate caches for positive and negative branches

References:
- MagCache: Magnitude Caching for Diffusion Models
  https://zehong-ma.github.io/MagCache/
- HuggingFace PR: https://github.com/huggingface/diffusers/pull/12744
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.configs.models import DiTConfig

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams


@dataclass
class MagCacheContext:
    """Common context extracted for MagCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        current_timestep: Current denoising timestep index (0-indexed).
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        skip_threshold: Threshold for magnitude ratio skip decision.
        magnitude_ratios: Pre-computed ratios from calibration.
        is_calibration: True during calibration phase.
        magcache_params: Full MagCacheParams for model-specific access.
    """

    current_timestep: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool  # For CFG branch selection
    skip_threshold: float
    magnitude_ratios: list[float]  # Pre-computed during calibration
    is_calibration: bool  # True during calibration runs
    magcache_params: "MagCacheParams"  # Full params for model-specific access


class MagCacheMixin:
    """
    Mixin class providing MagCache optimization functionality.

    MagCache accelerates diffusion inference by selectively skipping redundant
    computation when residual magnitudes follow predictable decay patterns.

    This mixin should be inherited by DiT model classes that want to support
    MagCache optimization. It provides:
    - State management for tracking residual magnitudes
    - CFG-aware caching (separate caches for positive/negative branches)
    - Decision logic for when to compute vs. use cache
    - Calibration mode for computing magnitude ratios

    Example usage in a DiT model:
        class MyDiT(MagCacheMixin, BaseDiT):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                self._init_magcache_state()

            def forward(self, hidden_states, timestep, ...):
                ctx = self._get_magcache_context()

                # CALIBRATION MODE: Collect residuals
                if ctx is not None and ctx.is_calibration:
                    original_hidden_states = hidden_states.clone()
                    hidden_states = self._transformer_forward(...)
                    self.maybe_cache_states(hidden_states, original_hidden_states)
                    return hidden_states

                # INFERENCE MODE: Check if we can skip
                if ctx is not None and not ctx.is_calibration:
                    should_skip = self._should_skip_using_magnitude_ratio(
                        current_timestep=ctx.current_timestep,
                        magnitude_ratios=ctx.magnitude_ratios,
                        skip_threshold=ctx.skip_threshold
                    )

                    if should_skip:
                        return self.retrieve_cached_states(hidden_states)

                # NORMAL COMPUTATION
                original_hidden_states = hidden_states.clone()
                hidden_states = self._transformer_forward(...)

                if ctx is not None:
                    self.maybe_cache_states(hidden_states, original_hidden_states)

                return hidden_states

    Subclass implementation notes:
        - `retrieve_cached_states()`: Must be overridden to return cached output
        - `maybe_cache_states()`: Override to store states for cache retrieval

    Attributes:
        enable_magcache: Whether MagCache is enabled.
        previous_residual: Cached residual for positive branch.
        calibration_residuals: List of residuals collected during calibration.
        _supports_cfg_cache: Whether this model supports CFG cache separation.

    CFG-specific attributes (only when _supports_cfg_cache is True):
        previous_residual_negative: Cached residual for negative branch.
    """

    # Models that support CFG cache separation (wan/hunyuan/zimage)
    # Models not in this set (flux/qwen) use single cache regardless of CFG
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def _init_magcache_state(self) -> None:
        """Initialize MagCache state. Call this in subclass __init__."""
        # Common MagCache state
        self.enable_magcache = True
        # Flag indicating if this model supports CFG cache separation
        self._supports_cfg_cache = (
            self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        )

        # Always initialize positive cache fields (used in all modes)
        self.previous_residual: torch.Tensor | None = None
        self.calibration_residuals: list[torch.Tensor] = []

        # CFG-specific fields initialized to None (created when CFG is used)
        # These are only used when _supports_cfg_cache is True AND do_cfg is True
        if self._supports_cfg_cache:
            self.previous_residual_negative: torch.Tensor | None = None

    def reset_magcache_state(self) -> None:
        """Reset all MagCache state at the start of each generation task."""
        # Primary cache fields (always present)
        self.previous_residual = None
        self.calibration_residuals = []
        self.enable_magcache = True

        # CFG negative cache fields (always reset, may be unused)
        if self._supports_cfg_cache:
            self.previous_residual_negative = None

    def _get_magcache_context(self) -> MagCacheContext | None:
        """
        Check MagCache preconditions and extract common context.

        Returns:
            MagCacheContext if MagCache is enabled and properly configured,
            None if should skip MagCache logic entirely.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        # Early return checks
        if (
            forward_batch is None
            or not forward_batch.enable_magcache
            or forward_batch.magcache_params is None
        ):
            return None

        magcache_params = forward_batch.magcache_params

        # Extract common values
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps
        do_cfg = forward_batch.do_classifier_free_guidance
        is_cfg_negative = forward_batch.is_cfg_negative

        # Reset at first timestep (only for positive branch to avoid double reset)
        if current_timestep == 0 and not is_cfg_negative:
            self.reset_magcache_state()

        return MagCacheContext(
            current_timestep=current_timestep,
            num_inference_steps=num_inference_steps,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            skip_threshold=magcache_params.skip_threshold,
            magnitude_ratios=magcache_params.magnitude_ratios,
            is_calibration=magcache_params.is_calibration,
            magcache_params=magcache_params,
        )

    def _should_skip_using_magnitude_ratio(
        self,
        current_timestep: int,
        magnitude_ratios: list[float],
        skip_threshold: float,
    ) -> bool:
        """
        Decide whether to skip based on pre-computed magnitude ratio.

        If the magnitude ratio at current timestep is below threshold,
        it indicates the residual change is small enough to reuse the
        cached output.

        Args:
            current_timestep: Current denoising step (0-indexed)
            magnitude_ratios: Pre-computed ratios from calibration
            skip_threshold: Threshold below which to skip computation

        Returns:
            True if should skip computation and use cache
        """
        if not self.enable_magcache:
            return False

        # No previous residual cached yet - must compute
        if self.previous_residual is None:
            return False

        # Out of bounds - must compute
        if current_timestep >= len(magnitude_ratios):
            return False

        # Check magnitude ratio against threshold
        ratio = magnitude_ratios[current_timestep]
        return ratio < skip_threshold

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """
        Cache residual for next step with CFG separation.

        During calibration: Store residuals for magnitude ratio computation
        During inference: Cache most recent residual for reuse

        Args:
            hidden_states: Output hidden states from transformer
            original_hidden_states: Input hidden states to transformer
        """
        residual = hidden_states - original_hidden_states

        ctx = self._get_magcache_context()
        if ctx is not None and ctx.is_calibration:
            # Store for calibration analysis
            self.calibration_residuals.append(residual.detach().cpu())

        # Cache for next step (both calibration and inference modes)
        if ctx is not None and ctx.is_cfg_negative and self._supports_cfg_cache:
            self.previous_residual_negative = residual
        else:
            self.previous_residual = residual

    def should_skip_forward_for_cached_states(self, **kwargs: dict[str, Any]) -> bool:
        """Check if forward can be skipped using cached states."""
        ctx = self._get_magcache_context()
        if ctx is None or ctx.is_calibration:
            return False

        return self._should_skip_using_magnitude_ratio(
            current_timestep=ctx.current_timestep,
            magnitude_ratios=ctx.magnitude_ratios,
            skip_threshold=ctx.skip_threshold,
        )

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieve cached residual with CFG separation.

        Args:
            hidden_states: Current input hidden states

        Returns:
            hidden_states + cached_residual
        """
        ctx = self._get_magcache_context()

        if ctx is not None and ctx.is_cfg_negative and self._supports_cfg_cache:
            if self.previous_residual_negative is not None:
                return hidden_states + self.previous_residual_negative

        if self.previous_residual is not None:
            return hidden_states + self.previous_residual

        # Fallback: no cache available (shouldn't happen if skip decision was correct)
        return hidden_states

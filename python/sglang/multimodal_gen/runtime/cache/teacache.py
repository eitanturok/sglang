# SPDX-License-Identifier: Apache-2.0
"""
TeaCache: Temporal similarity-based caching for diffusion models.

TeaCache accelerates diffusion inference by selectively skipping redundant
computation when consecutive diffusion steps are similar enough. This is
achieved by tracking the L1 distance between modulated inputs across timesteps.

Key concepts:
- Modulated input: The input to transformer blocks after timestep conditioning
- L1 distance: Measures how different consecutive timesteps are
- Threshold: When accumulated L1 distance exceeds threshold, force computation
- CFG support: Separate caches for positive and negative branches

References:
- TeaCache: Accelerating Diffusion Models with Temporal Similarity
  https://arxiv.org/abs/2411.14324
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class TeaCacheState:
    """TeaCache state for a single CFG branch (positive or negative)."""

    previous_modulated_input: "torch.Tensor | None" = field(default=None, repr=False)
    previous_residual: "torch.Tensor | None" = field(default=None, repr=False)
    accumulated_rel_l1_distance: float = 0.0

    def reset(self) -> None:
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0


@dataclass
class TeaCacheContext:
    """Common context extracted for TeaCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        teacache_params: Full TeaCacheParams for model-specific access.
    """

    cnt: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool  # For CFG branch selection
    teacache_params: "TeaCacheParams"  # Full params for model-specific access


class TeaCacheMixin:
    """
    Mixin class providing TeaCache optimization functionality.

    TeaCache accelerates diffusion inference by selectively skipping redundant
    computation when consecutive diffusion steps are similar enough.

    This mixin should be inherited by DiT model classes that want to support
    TeaCache optimization. It provides:
    - State management for tracking L1 distances
    - CFG-aware caching (separate caches for positive/negative branches)
    - Decision logic for when to compute vs. use cache

    Example usage in a DiT model:
        class MyDiT(TeaCacheMixin, BaseDiT):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                self._init_teacache_state()

            def forward(self, hidden_states, timestep, ...):
                ctx = self._get_teacache_context()
                if ctx is not None:
                    # Compute modulated input (model-specific, e.g., after timestep embedding)
                    modulated_input = self._compute_modulated_input(hidden_states, timestep)
                    is_boundary = (ctx.current_timestep == 0 or
                                   ctx.current_timestep >= ctx.num_inference_steps - 1)

                    should_skip = self.should_skip_forward(
                        modulated_inp=modulated_input,
                        is_boundary_step=is_boundary,
                        coefficients=ctx.coefficients,
                        teacache_thresh=ctx.teacache_thresh,
                    )

                    if should_skip:
                        # Use cached residual (must implement retrieve_cached_states)
                        return self.retrieve_cached_states(hidden_states)

                # Normal forward pass...
                output = self._transformer_forward(hidden_states, timestep, ...)

                # Cache states for next step
                if ctx is not None:
                    self.maybe_cache_states(output, hidden_states)

                return output

    Subclass implementation notes:
        - `_compute_modulated_input()`: Model-specific method to compute the input
          after timestep conditioning (used for L1 distance calculation)
        - `retrieve_cached_states()`: Must be overridden to return cached output
        - `maybe_cache_states()`: Override to store states for cache retrieval

    Attributes:
        cnt: Counter for tracking steps.
        enable_teacache: Whether TeaCache is enabled.
        previous_modulated_input: Cached modulated input for positive branch.
        previous_residual: Cached residual for positive branch.
        accumulated_rel_l1_distance: Accumulated L1 distance for positive branch.
        is_cfg_negative: Whether currently processing negative CFG branch.
        _supports_cfg_cache: Whether this model supports CFG cache separation.
    """

    # Models that support CFG cache separation (wan/hunyuan/zimage)
    # Models not in this set (flux/qwen) auto-disable TeaCache when CFG is enabled
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def _init_teacache(self) -> None:
        """Initialize TeaCache state. Call this in subclass __init__."""
        self.cnt = 0
        self.cache_state = TeaCacheState()
        self.cache_state_neg = TeaCacheState() if self._supports_cfg_cache else None

    def reset_teacache_state(self) -> None:
        """Reset all TeaCache state at the start of each generation task."""
        # self.cnt = 0 #todo: remove?
        self.cache_state.reset()
        if self._supports_cfg_cache:
            self.cache_state_neg.reset()

    def should_skip_forward_teacache(self, ctx: TeaCacheContext, **kwargs) -> bool:
        """
        Decide whether to skip forward pass using TeaCache.

        Args:
            modulated_inp: Current timestep's modulated input tensor.
            cnt: Current step counter (increments each forward call, incl. CFG branches).
            ret_steps: Number of initial steps that always compute (boundary).
            cutoff_steps: Step index at which trailing boundary begins.
            coefficients: Polynomial coefficients for L1 rescaling.
            teacache_thresh: Threshold for accumulated L1 distance.
            is_cfg_negative: Whether currently processing the negative CFG branch.

        Returns:
            True if forward pass should be skipped (use cache), False to compute.
        """

        if not self.enable_teacache:
            return False

        # unpack parameters
        cnt, num_inference_steps, do_cfg, is_cfg_negative, teacache_params = (
            ctx.cnt,
            ctx.num_inference_steps,
            ctx.do_cfg,
            ctx.is_cfg_negative,
            ctx.teacache_params,
        )
        modulated_inp = kwargs["timestep_proj"] if teacache_params.use_ret_steps else kwargs["temb"]
        state = self.cache_state_neg if is_cfg_negative else self.cache_state

        # do not skip if boundary step
        ret_steps = teacache_params.ret_steps
        cutoff_steps = teacache_params.get_cutoff_steps(num_inference_steps)
        if not do_cfg:
            ret_steps //= 2
            cutoff_steps //= 2
        is_boundary_step = cnt < ret_steps or cnt >= cutoff_steps
        if is_boundary_step:
            state.accumulated_rel_l1_distance = 0.0
            return False

        # do not skip if previous input is not set
        if state.previous_modulated_input is None:
            state.accumulated_rel_l1_distance = 0.0
            return False

        # update the previous modulated input
        state.previous_modulated_input = modulated_inp.clone()

        # compute relative L1 distance and accumulated distance
        diff = modulated_inp - state.previous_modulated_input
        rel_l1 = (diff.abs().mean() / state.previous_modulated_input.abs().mean()).cpu().item()
        accumulated_rel_l1_distance = state.accumulated_rel_l1_distance + np.poly1d(teacache_params.coefficients)(rel_l1)

        # skip if accumulated distance is below threshold, otherwise reset and do not skip
        if accumulated_rel_l1_distance < teacache_params.teacache_thresh:
            state.accumulated_rel_l1_distance = accumulated_rel_l1_distance
            return True  # skip forward, use cache
        state.accumulated_rel_l1_distance = 0.0
        return False

    def _get_teacache_context(self) -> TeaCacheContext | None:
        """
        Check TeaCache preconditions and extract common context.

        Returns:
            TeaCacheContext if TeaCache is enabled and properly configured,
            None if should skip TeaCache logic entirely.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return None

        return TeaCacheContext(
            cnt=self.cnt,
            num_inference_steps=forward_batch.num_inference_steps,
            do_cfg=forward_batch.do_classifier_free_guidance,
            is_cfg_negative=forward_batch.is_cfg_negative,
            teacache_params=getattr(forward_batch.sampling_params, "teacache_params", None),
        )

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache states for later retrieval. Override in subclass if needed."""
        pass

    def should_skip_forward_for_cached_states(self, **kwargs: dict[str, Any]) -> bool:
        """Check if forward can be skipped using cached states."""
        return False

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached states. Must be implemented by subclass."""
        raise NotImplementedError("retrieve_cached_states is not implemented")

    def calibrate_teacache(self, ctx: TeaCacheContext, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor) -> None:
        """Calibrate TeaCache by logging L1 distance metrics. Override if needed."""
        pass

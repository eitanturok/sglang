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
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.multimodal_gen.runtime.cache.base import DiffusionCache

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import (
        TeaCacheParams,
        WanTeaCacheParams,
    )


@dataclass
class TeaCacheState:
    """Per-CFG-branch state for TeaCache."""

    previous_modulated_input: torch.Tensor | None = field(default=None, repr=False)
    previous_residual: torch.Tensor | None = field(default=None, repr=False)
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
        cnt: The number of forward passes (0-indexed).
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        teacache_thresh: Threshold for accumulated L1 distance.
        coefficients: Polynomial coefficients for L1 rescaling.
        teacache_params: Full TeaCacheParams for model-specific access.
    """

    cnt: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool
    params: "TeaCacheParams|WanTeaCacheParams"


class TeaCacheStrategy(DiffusionCache):
    """TeaCache speedups diffusion inference by skipping entire forward
    passes when the L1 distance between modulated inputs are similar enough.

    `TeaCacheStrategy` can be used by all models that inherit from
    `CachableDiT` via `CachableDiT.cache`. This is lazily initialized in the
    first forward pass of the model via `CachableDiT.init_cache()` because
    it depends on sampling parameters that are only known at inference time.

    `TeaCacheStrategy` maintains a state `TeaCacheState`, one for each
    positive and (optionally) negative CFG branches. It also uses `TeaCacheContext`
    which contains all the necessary information for deciding to cache or not.

    Example usage in a `CachableDiT` model:

        def forward(self, hidden_states, encoder_hidden_states, timestep, ...):
            # Lazy-init cache on first call
            if self.cache is None:
                self.init_cache()

            # Reset at the start of each generation (timestep 0, positive branch)
            if self.cache and current_timestep == 0 and not is_cfg_negative:
                self.cache.reset()
                self.cnt = 0

            # Check whether this forward pass can be skipped
            ctx = self.cache.get_context(self.cnt) if self.cache is not None else None
            should_skip = (
                ctx is not None and not self.calibrate_cache
                and self.cache.should_skip(ctx, timestep_proj=timestep_proj, temb=temb)
            )

            if should_skip:
                # Reuse cached residual from the previous non-skipped step
                hidden_states = self.cache.retrieve(hidden_states, ctx)
            else:
                if self.cache is not None:
                    original_hidden_states = hidden_states.clone()

                for block in self.blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states, ...)

                if self.cache is not None:
                    if self.calibrate_cache:
                        self.cache.calibrate(hidden_states, original_hidden_states, ctx)
                    else:
                        self.cache.maybe_cache(hidden_states, original_hidden_states, ctx)

            self.cnt += 1
            ...
    """

    def __init__(self, supports_cfg_cache: bool) -> None:
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg_cache else None

    def reset(self) -> None:
        assert isinstance(self.state, TeaCacheState)
        self.state.reset()
        if self.state_neg is not None:
            assert isinstance(self.state_neg, TeaCacheState)
            self.state_neg.reset()

    def get_context(self, cnt: int) -> TeaCacheContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        fb = forward_context.forward_batch
        if fb is None:
            return None

        steps = fb.num_inference_steps
        do_cfg = fb.do_classifier_free_guidance
        is_neg = fb.is_cfg_negative
        params = getattr(fb.sampling_params, "teacache_params", None)
        assert (
            params is not None
        ), "TeaCacheStrategy requires teacache_params in sampling_params"

        return TeaCacheContext(cnt, steps, do_cfg, is_neg, params)

    def should_skip(self, ctx: TeaCacheContext, **kwargs) -> bool:
        state = (
            self.state_neg
            if (ctx.is_cfg_negative and self.state_neg is not None)
            else self.state
        )
        assert isinstance(state, TeaCacheState) and isinstance(ctx, TeaCacheContext)

        # Cannot skip on boundary steps
        min_cnt = (
            ctx.params.skip_start_step * 2 if ctx.do_cfg else ctx.params.skip_start_step
        )
        max_cnt = (
            (ctx.num_inference_steps - ctx.params.skip_end_step) * 2
            if ctx.do_cfg
            else (ctx.num_inference_steps - ctx.params.skip_end_step)
        )
        if ctx.cnt < min_cnt or ctx.cnt >= max_cnt:
            state.reset()
            return False

        modulated_inp = (
            kwargs["timestep_proj"]
            if getattr(ctx.params, "use_ret_steps", None)
            else kwargs["temb"]
        )

        # Cannot skip when have no previous input
        if state.previous_modulated_input is None:
            state.previous_modulated_input = modulated_inp.clone()
            return False

        # Accumulate relative L1 distance
        diff = modulated_inp - state.previous_modulated_input
        rel_l1 = (
            (diff.abs().mean() / state.previous_modulated_input.abs().mean())
            .cpu()
            .item()
        )
        accumulated = state.accumulated_rel_l1_distance + np.poly1d(
            ctx.params.coefficients
        )(rel_l1)

        state.accumulated_rel_l1_distance = accumulated
        state.previous_modulated_input = modulated_inp.clone()

        if accumulated < ctx.params.teacache_thresh:
            return True
        state.reset()
        return False

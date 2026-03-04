# SPDX-License-Identifier: Apache-2.0
"""
TeaCache: Temporal similarity-based caching for diffusion models.

Skips redundant transformer forward passes by tracking the accumulated L1
distance between modulated inputs across consecutive diffusion steps.

References:
- TeaCache: https://arxiv.org/abs/2411.14324
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.multimodal_gen.runtime.cache.base import DiffusionCache

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams, WanTeaCacheParams


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
    """Per-step snapshot for TeaCache decisions.

    cnt is the forward-call index: timestep * 2 + cfg_offset when CFG is on,
    so boundary thresholds (ret_steps / cutoff_steps) are in the same units.
    """
    cnt: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool
    params: "TeaCacheParams|WanTeaCacheParams"


class TeaCacheStrategy(DiffusionCache):
    """TeaCache caching strategy.

    Constructed by CachableDiT.init_cache() once per generation when
    teacache is selected. Owns both CFG-branch states.
    """

    def __init__(self, supports_cfg_cache: bool) -> None:
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg_cache else None

    def reset(self) -> None:
        self.state.reset()
        if self.state_neg is not None:
            self.state_neg.reset()

    def get_context(self, cnt: int) -> TeaCacheContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
        forward_context = get_forward_context()
        fb = forward_context.forward_batch
        if fb is None:
            return None

        do_cfg = fb.do_classifier_free_guidance
        is_neg = fb.is_cfg_negative
        steps = fb.num_inference_steps
        params = getattr(fb.sampling_params, "teacache_params", None)
        assert params is not None, "TeaCacheStrategy requires teacache_params in sampling_params"

        return TeaCacheContext(cnt=cnt, num_inference_steps=steps, do_cfg=do_cfg, is_cfg_negative=is_neg, params=params)

    def should_skip(self, ctx: TeaCacheContext, **kwargs) -> bool:
        state = self.state_neg if (ctx.is_cfg_negative and self.state_neg is not None) else self.state
        modulated_inp = kwargs["timestep_proj"] if ctx.params.use_ret_steps else kwargs["temb"]

        ret_steps = ctx.params.ret_steps if ctx.do_cfg else ctx.params.ret_steps // 2
        cutoff_steps = ctx.params.get_cutoff_steps(ctx.num_inference_steps) if ctx.do_cfg else ctx.params.get_cutoff_steps(ctx.num_inference_steps) // 2

        # Always compute boundary steps
        if ctx.cnt < ret_steps or ctx.cnt >= cutoff_steps:
            state.reset()
            return False

        # No previous input yet — compute and store
        if state.previous_modulated_input is None:
            state.previous_modulated_input = modulated_inp.clone()
            state.accumulated_rel_l1_distance = 0.0
            return False

        # Accumulate relative L1 distance
        diff = modulated_inp - state.previous_modulated_input
        rel_l1 = (diff.abs().mean() / state.previous_modulated_input.abs().mean()).cpu().item()
        accumulated = state.accumulated_rel_l1_distance + np.poly1d(ctx.params.coefficients)(rel_l1)

        state.previous_modulated_input = modulated_inp.clone()

        if accumulated < ctx.params.teacache_thresh:
            state.accumulated_rel_l1_distance = accumulated
            return True
        state.accumulated_rel_l1_distance = 0.0
        return False

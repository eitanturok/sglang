# SPDX-License-Identifier: Apache-2.0
"""
MagCache: Magnitude-ratio-based caching for diffusion models.

Skips redundant transformer forward passes when magnitude ratios of
consecutive residuals are predictably similar.

References:
- MagCache: https://openreview.net/forum?id=KZn7TDOL4J
"""

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.cache.base import DiffusionCache

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams


@dataclass
class MagCacheState:
    """Per-CFG-branch state for MagCache."""
    norm_ratio: float = 1.0
    accumulated_error: float = 0.0
    consecutive_skips: int = 0
    previous_residual: torch.Tensor | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.norm_ratio = 1.0
        self.accumulated_error = 0.0
        self.consecutive_skips = 0
        self.previous_residual = None


@dataclass
class MagCacheContext:
    """Per-step snapshot for MagCache decisions.

    cnt is the forward-call index: timestep * 2 + cfg_offset when CFG is on,
    used to index into mag_ratios and for boundary checks.
    """
    cnt: int
    do_cfg: bool
    is_cfg_negative: bool
    params: "MagCacheParams"


class MagCacheStrategy(DiffusionCache):
    """MagCache caching strategy.

    Constructed by CachableDiT.init_cache() once per generation when
    magcache is selected. Owns both CFG-branch states.
    """

    def __init__(self, params: "MagCacheParams", supports_cfg_cache: bool) -> None:
        # Precompute boundary step indices from params (only needed at init)
        self.min_steps = int(params.num_steps * params.retention_ratio) * 2 if params.use_ret_steps else 2
        self.max_steps = params.num_steps * 2 if params.use_ret_steps else params.num_steps * 2 - 2
        self.calibration_path = None
        self.state = MagCacheState()
        self.state_neg = MagCacheState() if supports_cfg_cache else None

    def reset(self) -> None:
        self.state.reset()
        if self.state_neg is not None:
            self.state_neg.reset()

    def get_context(self, cnt: int) -> MagCacheContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
        forward_context = get_forward_context()
        fb = forward_context.forward_batch
        if fb is None:
            return None

        do_cfg = fb.do_classifier_free_guidance
        is_neg = fb.is_cfg_negative
        params = getattr(fb.sampling_params, "magcache_params", None)
        assert params is not None, "MagCacheStrategy requires magcache_params in sampling_params"

        return MagCacheContext(cnt=cnt, do_cfg=do_cfg, is_cfg_negative=is_neg, params=params)

    def should_skip(self, ctx: MagCacheContext, **kwargs) -> bool:
        state = self.state_neg if (ctx.is_cfg_negative and self.state_neg is not None) else self.state

        # Always compute boundary steps
        if ctx.cnt < self.min_steps or ctx.cnt >= self.max_steps:
            state.reset()
            return False

        if ctx.params.mag_ratios is None:
            return False

        state.norm_ratio *= ctx.params.mag_ratios[ctx.cnt]
        state.consecutive_skips += 1
        state.accumulated_error += abs(1 - state.norm_ratio)

        if state.accumulated_error < ctx.params.threshold and state.consecutive_skips <= ctx.params.max_skip_steps:
            return True
        state.reset()
        return False

    def calibrate(self, hidden_states, original_hidden_states, ctx: MagCacheContext) -> None:
        state = self.state_neg if (ctx.is_cfg_negative and self.state_neg is not None) else self.state
        prev = state.previous_residual

        curr = hidden_states.squeeze(0) - original_hidden_states

        if prev is None:
            mag_ratio, mag_std, cos_dis = 1.0, 0.0, 0.0
        else:
            norms = curr.norm(dim=-1) / prev.norm(dim=-1)
            mag_ratio = norms.mean().item()
            mag_std = norms.std().item()
            cos_dis = (1 - F.cosine_similarity(curr, prev, dim=-1, eps=1e-8)).mean().item()

        state.previous_residual = curr

        if self.calibration_path is None:
            from sglang.multimodal_gen.envs import SGLANG_DIFFUSION_CACHE_ROOT
            from sglang.multimodal_gen.runtime.server_args import get_global_server_args
            cache_dir = os.path.join(SGLANG_DIFFUSION_CACHE_ROOT, "magcache_calibration")
            os.makedirs(cache_dir, exist_ok=True)
            model_name = get_global_server_args().model_path.replace("/", "--")
            self.calibration_path = os.path.join(cache_dir, f"{model_name}.jsonl")

        with open(self.calibration_path, "a") as f:
            f.write(json.dumps({
                "cnt": ctx.cnt,
                "mag_ratio": mag_ratio,
                "mag_std": mag_std,
                "cos_dis": cos_dis,
                "negative": ctx.is_cfg_negative,
            }) + "\n")

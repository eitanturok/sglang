# SPDX-License-Identifier: Apache-2.0
"""
MagCache: Magnitude-ratio-based caching for diffusion models.

MagCache accelerates diffusion inference by skipping forward passes when
magnitude ratios of consecutive residuals are predictably similar.

Key differences from TeaCache:
- Uses magnitude ratios of residuals instead of L1 distance of inputs
- Tracks consecutive_skips counter to prevent infinite skipping
- Simpler accumulation (no polynomial rescaling)

References:
- MagCache: Fast Video Generation with Magnitude-Aware Cache
  https://openreview.net/forum?id=KZn7TDOL4J
"""

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models import DiTConfig

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams


@dataclass
class MagCacheState:
    """Per-branch state for MagCache (one instance for positive, one for negative CFG)."""

    norm_ratio: float = 1.0
    accumulated_error: float = 0.0
    consecutive_skips: int = 0
    previous_residual: "torch.Tensor | None" = field(default=None, repr=False)
    previous_residual_norm: float = 0.0

    def reset(self) -> None:
        self.norm_ratio = 1.0
        self.accumulated_error = 0.0
        self.consecutive_skips = 0
        self.previous_residual = None
        self.previous_residual_norm = 0.0


@dataclass
class MagCacheContext:
    """
    Context for MagCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        current_timestep: Current denoising timestep index (0-indexed).
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
    """

    current_timestep: int
    cnt: int
    do_cfg: bool
    is_cfg_negative: bool
    magcache_params: "MagCacheParams | None" = None


class MagCacheMixin:
    # Models that support CFG cache separation (same as TeaCache)
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def _init_magcache(self, is_cfg_negative:bool, magcache_params:"MagCacheParams") -> None:

        self.num_steps = magcache_params.num_steps # todo: don't hardcode
        self.retention_ratio = magcache_params.retention_ratio
        self.magcache_thresh = magcache_params.threshold
        self.max_skip_steps = magcache_params.max_skip_steps
        self.mag_ratios = magcache_params.mag_ratios
        self.use_ret_steps = magcache_params.use_ret_steps

        self.min_steps = int(self.num_steps * self.retention_ratio) * 2 if self.use_ret_steps else 2
        self.max_steps = self.num_steps * 2 if self.use_ret_steps else self.num_steps * 2 - 2

        self.calibration_path = None

        # Per-branch state: index 0 = positive CFG. Index 1 (negative) added when CFG is used.
        self._cache_states: list[MagCacheState] = [MagCacheState()]
        if self._supports_cfg_cache and is_cfg_negative:
            self._cache_states.append(MagCacheState())

    def reset(self, is_cfg_negative):
        self._cache_states[int(is_cfg_negative)].reset()

    def should_skip_forward_magcache(self, current_timestep, cnt, do_cfg, is_cfg_negative=False):
        if not self.enable_magcache:
            return False

        state = self._cache_states[int(is_cfg_negative)]

        # always compute first few and last few steps
        is_boundary_step = cnt < self.min_steps or cnt >= self.max_steps
        if is_boundary_step:
            state.reset()
            return False

        state.norm_ratio *= self.mag_ratios[cnt]  # magnitude ratio between current step and the cached step
        state.consecutive_skips += 1
        state.accumulated_error += abs(1 - state.norm_ratio)

        if state.accumulated_error < self.magcache_thresh and state.consecutive_skips <= self.max_skip_steps:
            return True
        else:
            state.reset()
            return False

    def calibrate_magcache(self, ctx, hidden_states, original_hidden_states):

        # create directory for magcache calibration results
        if self.calibration_path is None:
            from sglang.multimodal_gen.envs import SGLANG_DIFFUSION_CACHE_ROOT
            from sglang.multimodal_gen.runtime.server_args import get_global_server_args
            cache_dir = os.path.join(SGLANG_DIFFUSION_CACHE_ROOT, "magcache_calibration")
            os.makedirs(cache_dir, exist_ok=True)
            model_name = get_global_server_args().model_path.replace("/", "--")
            self.calibration_path = os.path.join(cache_dir, f"{model_name}.jsonl")

        prev_residual = self._cache_states[int(ctx.is_cfg_negative)].previous_residual
        if prev_residual is None:
            mag_ratio = 1.0
            mag_std = 0.0
            cos_dis = 0.0
        else:
            curr_residual = hidden_states.squeeze(0) - original_hidden_states
            mag_ratio = ((curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).mean()).item()
            mag_std = (curr_residual.norm(dim=-1)/prev_residual.norm(dim=-1)).std().item()
            cos_dis = (1-F.cosine_similarity(curr_residual, prev_residual, dim=-1, eps=1e-8)).mean().item()

        with open(self.calibration_path, "a") as f:
            f.write(json.dumps({"cnt": ctx.cnt, "mag_ratio": mag_ratio, "mag_std": mag_std, "cos_dis": cos_dis, "negative": ctx.is_cfg_negative}) + "\n")

    def _get_magcache_context(self) -> MagCacheContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None:
            return None

        # unpack parameters
        current_timestep=forward_context.current_timestep
        do_cfg=forward_batch.do_classifier_free_guidance
        is_cfg_negative=forward_batch.is_cfg_negative
        magcache_params=getattr(forward_batch.sampling_params, "magcache_params", None)
        assert magcache_params is not None, "MagCache parameters not found in sampling_params." # todo: what about calibration

        # init cache at the start of each generation
        if current_timestep == 0:
            self._init_magcache(is_cfg_negative, magcache_params)

        # compute cnt index differently for cond and uncond branches in CFG
        cnt = current_timestep
        if do_cfg:
            cnt = current_timestep * 2 + (1 if is_cfg_negative else 0)

        return MagCacheContext(
            current_timestep=current_timestep,
            cnt=cnt,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            magcache_params=magcache_params,
        )

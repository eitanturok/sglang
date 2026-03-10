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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.runtime.cache import DiffusionCache

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


def _rescale_distance_tensor(
    coefficients: list[float], x: torch.Tensor
) -> torch.Tensor:
    """Polynomial rescaling using tensor operations (torch.compile friendly)."""
    c = coefficients
    return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]


def _compute_rel_l1_distance_tensor(
    current: torch.Tensor, previous: torch.Tensor
) -> torch.Tensor:
    """Compute relative L1 distance as a tensor (torch.compile friendly)."""
    prev_mean = previous.abs().mean()
    curr_diff_mean = (current - previous).abs().mean()
    rel_distance = torch.where(
        prev_mean > 1e-9,
        curr_diff_mean / prev_mean,
        torch.where(
            current.abs().mean() < 1e-9,
            torch.zeros(1, device=current.device, dtype=current.dtype),
            torch.full((1,), float("inf"), device=current.device, dtype=current.dtype),
        ),
    )
    return rel_distance.squeeze()


class TeaCacheState:
    """Mutable per-step state for one CFG branch."""

    def __init__(self) -> None:
        self.step: int = -1
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: torch.Tensor | None = None

    def reset(self) -> None:
        self.step = -1
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = None

    def update(
        self, modulated_inp: torch.Tensor | None, previous_residual: torch.Tensor | None
    ) -> None:
        self.previous_modulated_input = modulated_inp
        self.previous_residual = previous_residual

    def __repr__(self):
        return f"TeaCacheState(step={self.step}, accumulated_rel_l1_distance={self.accumulated_rel_l1_distance})"


class TeaCacheStrategy(DiffusionCache):
    """TeaCache skips diffusion forward passes when consecutive steps are similar enough.

    Owns two TeaCacheState objects (positive + optional negative CFG branch).
    params, num_steps, and coefficients are read from forward_batch once at the
    start of each generation via maybe_reset() and reused for all subsequent steps.

    Typical usage in a CachableDiT forward():

        cache.maybe_reset()
        if cache.should_skip(temb, timestep_proj):
            hidden_states = cache.read(hidden_states)
        else:
            original = hidden_states.clone()
            # ... run transformer blocks ...
            cache.write(hidden_states, original)
    """

    def __init__(self, supports_cfg: bool) -> None:
        # params updated every forward pass
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg else None
        # params updated at the start of each new generation
        # set in maybe_reset()
        self.cache_params: TeaCacheParams | None = None
        self.coefficients: list[float] = []
        self.num_steps: int = 0
        self.start_skipping: int = 0
        self.end_skipping: int = -1

    def _get_state(self) -> TeaCacheState:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        fb = get_forward_context().forward_batch
        is_cfg_negative = fb.is_cfg_negative if fb is not None else False
        if is_cfg_negative and self.state_neg is not None:
            return self.state_neg
        return self.state

    def maybe_reset(self, **kwargs) -> None:
        """Reset state when the previous generation is complete and initialize TeaCacheParams,
        num_steps, and coefficients at the start of a new generation. Called on every forward
        pass before should_skip().
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        state = self._get_state()

        # Reset state if we completed a generation
        if state.step == self.num_steps and state.step > 0:
            state.reset()

        # increment the number of steps at the beginning of each forward pass
        state.step += 1

        # Initialize values at the start of each new generation
        if state.step == 0:

            # set the teacache parameters
            fb = get_forward_context().forward_batch
            assert (
                fb is not None
            ), "TeaCacheStrategy required the forward_batch not be None"
            self.cache_params = getattr(fb.sampling_params, "teacache_params", None)

            # set the number of inference steps
            assert (
                self.cache_params is not None
            ), "TeaCacheStrategy requires teacache_params in sampling_params"
            self.num_steps = int(fb.num_inference_steps)

            # set the teacache coefficients
            if self.cache_params.coefficients_callback:
                self.coefficients = self.cache_params.coefficients_callback(
                    self.cache_params
                )
            else:
                self.coefficients = self.cache_params.coefficients

            # set the start and end skippable steps
            if isinstance(self.cache_params.start_skipping, float):
                self.start_skipping = int(
                    self.num_steps * self.cache_params.start_skipping
                )
            elif self.cache_params.start_skipping < 0:
                self.start_skipping = self.num_steps + self.cache_params.start_skipping
            else:
                self.start_skipping = self.cache_params.start_skipping

            if isinstance(self.cache_params.end_skipping, float):
                self.end_skipping = int(self.num_steps * self.cache_params.end_skipping)
            elif self.cache_params.end_skipping < 0:
                self.end_skipping = self.num_steps + self.cache_params.end_skipping
            else:
                self.end_skipping = self.cache_params.end_skipping

            assert (
                self.start_skipping <= self.end_skipping
            ), f"expected start_skipping <= end_skipping but got start_skipping={self.start_skipping} end_skipping={self.end_skipping}"

    def should_skip(
        self, modulated_input: torch.Tensor | None = None, **kwargs
    ) -> bool:
        """Decide whether this forward pass can be skipped."""
        state = self._get_state()
        assert self.cache_params is not None

        # Boundary steps always compute
        if state.step < self.start_skipping or state.step >= self.end_skipping:
            return False

        # First time computing, no previous input to compare against
        if state.accumulated_rel_l1_distance is None:
            state.accumulated_rel_l1_distance = torch.zeros(
                1, device=modulated_input.device, dtype=modulated_input.dtype
            )
            return False

        assert state.previous_modulated_input is not None
        assert modulated_input is not None
        rel_l1 = _compute_rel_l1_distance_tensor(
            modulated_input, state.previous_modulated_input
        )
        rescaled = _rescale_distance_tensor(self.coefficients, rel_l1)
        state.accumulated_rel_l1_distance += rescaled

        # If below threshold, skip the forward pass
        if state.accumulated_rel_l1_distance < self.cache_params.rel_l1_thresh:
            return True

        # If threshold exceeded, reset accumulated so next window starts fresh
        state.accumulated_rel_l1_distance = torch.zeros(
            1, device=modulated_input.device, dtype=modulated_input.dtype
        )
        return False

    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        modulated_input: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Store residual after a full forward pass."""
        assert self.cache_params is not None
        residual = hidden_states.squeeze(0) - original_hidden_states
        state = self._get_state()
        state.update(modulated_input, residual)

    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct output from cached residual."""
        return hidden_states + self._get_state().previous_residual

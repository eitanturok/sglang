# SPDX-License-Identifier: Apache-2.0
"""
TeaCache accelerates diffusion inference by skipping redundant forward
passes when consecutive denoising steps are sufficiently similar, as measured
by the accumulated relative L1 distance of modulated inputs.

References:
- TeaCache: Accelerating Diffusion Models with Temporal Similarity
  https://arxiv.org/abs/2411.14324
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import torch


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
    """Tracks step progress, cached tensors, and L1 distances for a single CFG path. Updated every timestep."""

    def __init__(self) -> None:
        self.step: int = 0
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: torch.Tensor | None = None

    def update(
        self, modulated_inp: torch.Tensor | None, previous_residual: torch.Tensor | None
    ) -> None:
        """Store the current modulated input and its computed residual for possible future reuse."""
        self.previous_modulated_input = modulated_inp
        self.previous_residual = previous_residual

    def __repr__(self):
        return f"TeaCacheState(step={self.step}, accumulated_rel_l1_distance={self.accumulated_rel_l1_distance})"


class TeaCacheStrategy:
    """Implements TeaCache to skip redundant diffusion forward passes.

    TeaCacheStrategy manages two TeaCacheState objects (positive + optional
    negative CFG branch) and stores parameters needed to make the skip decision.
    """

    def __init__(
        self,
        supports_cfg: bool,
        coefficients: list[float],
        rel_l1_thresh: float,
        start_skipping: int | None,
        end_skipping: int | None,
    ) -> None:
        """Initialize cache states and all generation parameters."""
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg else None
        self.coefficients = coefficients
        self.rel_l1_thresh = rel_l1_thresh
        self.start_skipping = start_skipping
        self.end_skipping = end_skipping

    def _get_state(self) -> TeaCacheState:
        """Select the appropriate cache state (positive/negative cfg) based on the forward context."""
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_batch = get_forward_context().forward_batch
        is_cfg_negative = (
            forward_batch.is_cfg_negative if forward_batch is not None else False
        )
        if is_cfg_negative and self.state_neg is not None:
            return self.state_neg
        return self.state

    def maybe_reset(self) -> None:
        """Increment the step counter."""
        self._get_state().step += 1

    def should_skip(
        self, modulated_input: torch.Tensor | None = None, **kwargs
    ) -> bool:
        """Decide whether this forward pass can be skipped based on the accumulated L1 distance of the modulated input."""
        state = self._get_state()

        # No valid skip window for this generation
        if self.start_skipping is None or self.end_skipping is None:
            return False

        # Boundary steps always compute
        if state.step < self.start_skipping or state.step >= self.end_skipping:
            return False

        # First time computing, no previous input to compare against
        if state.accumulated_rel_l1_distance is None:
            state.accumulated_rel_l1_distance = torch.zeros(
                1, device=modulated_input.device, dtype=modulated_input.dtype
            )
            return False

        # compute the accumulated relative l1 distance
        assert state.previous_modulated_input is not None
        assert modulated_input is not None
        rel_l1 = _compute_rel_l1_distance_tensor(
            modulated_input, state.previous_modulated_input
        )
        rescaled = _rescale_distance_tensor(self.coefficients, rel_l1)
        state.accumulated_rel_l1_distance += rescaled

        # If below threshold, skip the forward pass
        if state.accumulated_rel_l1_distance < self.rel_l1_thresh:
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
        """After the forward pass, cache the residual and the current modulated input."""
        residual = hidden_states.squeeze(0) - original_hidden_states
        state = self._get_state()
        state.update(modulated_input, residual)

    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Before the forward pass, read from the cache and apply it to the current hidden states."""
        return hidden_states + self._get_state().previous_residual

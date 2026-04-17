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
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import torch


@dataclass
class TeaCacheState:
    """Tracks step progress, cached tensors, and L1 distances for a single CFG path."""

    step: int = 0
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    accumulated_rel_l1_distance: torch.Tensor | None = None


def _rescale_distance_tensor(
    coefficients: list[float], x: torch.Tensor
) -> torch.Tensor:
    """Polynomial rescaling using tensor operations (torch.compile friendly)."""
    result = torch.zeros_like(x)
    for i, c in enumerate(coefficients):
        result = result + c * x ** (len(coefficients) - 1 - i)
    return result


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
        start_skipping: int,
        end_skipping: int,
    ) -> None:
        """Initialize cache states and all generation parameters."""
        self.supports_cfg = supports_cfg
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if supports_cfg else None
        self.coefficients = coefficients
        self.rel_l1_thresh = rel_l1_thresh
        self.start_skipping = start_skipping
        self.end_skipping = end_skipping
        if start_skipping >= end_skipping:
            logger.warning(
                f"TeaCache skip window is invalid (start_skipping={start_skipping} >= "
                f"end_skipping={end_skipping}). This can happen during warmup runs with "
                "very few steps. Skipping disabled."
            )

    def reset_states(self) -> None:
        """Reset cache states, discarding any stale tensors from a previous generation."""
        self.state = TeaCacheState()
        self.state_neg = TeaCacheState() if self.supports_cfg else None

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

    def advance(self, modulated_input: torch.Tensor) -> None:
        """Advance state by one step: update step counter, modulated input reference,
        and accumulated L1 distance. Always call once per forward pass before should_skip.

        accumulated_rel_l1_distance is None iff this step must be computed (out of window,
        no previous reference, or threshold just exceeded). should_skip reads this directly.
        """
        state = self._get_state()
        prev = state.previous_modulated_input
        step = state.step
        state.step += 1
        state.previous_modulated_input = modulated_input

        in_window = self.start_skipping <= step < self.end_skipping
        if not in_window or prev is None:
            state.accumulated_rel_l1_distance = None
            return

        if state.accumulated_rel_l1_distance is None:
            state.accumulated_rel_l1_distance = torch.zeros(
                1, device=modulated_input.device, dtype=modulated_input.dtype
            )

        rel_l1 = _compute_rel_l1_distance_tensor(modulated_input, prev)
        state.accumulated_rel_l1_distance += _rescale_distance_tensor(
            self.coefficients, rel_l1
        )

        if state.accumulated_rel_l1_distance >= self.rel_l1_thresh:
            state.accumulated_rel_l1_distance = None  # reset: force compute next step

    def should_skip(self) -> bool:
        """Return whether this forward pass can be skipped. Call after advance()."""
        return self._get_state().accumulated_rel_l1_distance is not None

    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        **kwargs,
    ) -> None:
        """After the forward pass, cache the residual."""
        state = self._get_state()
        state.previous_residual = hidden_states.squeeze(0) - original_hidden_states

    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Before the forward pass, read from the cache and apply it to the current hidden states."""
        return hidden_states + self._get_state().previous_residual

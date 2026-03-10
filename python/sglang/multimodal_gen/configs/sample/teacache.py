# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class TeaCacheParams(CacheParams):
    """
    Parameters for [TeaCache](https://arxiv.org/abs/2411.14324).

    TeaCache accelerates diffusion inference by skipping redundant forward passes
    when consecutive denoising steps are sufficiently similar, as measured by the
    accumulated relative L1 distance of modulated inputs.

    Attributes:
        rel_l1_thresh (`float`, defaults to `0.0`):
            Sensitivity for skipping steps. A higher threshold increases speed by
            skipping more aggressively, but may reduce image fidelity.
            (e.g., 0.25 $\\approx$ 1.5x speedup; 0.6 $\\approx$ 2.0x).
        start_skipping (`int`, defaults to `5`):
            Initial steps to always compute. These early steps define the global
            structure/composition and are too critical to skip.
        end_skipping (`int`, defaults to `0`):
            Final steps to always compute. Use this to ensure the last refinement
            passes preserve fine textures and details.
        coefficients (`List[float]`, defaults to `[]`):
            Polynomial coefficients for rescaling the raw relative L1 distance,
            evaluated as ``c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + c[3]*x + c[4]``.
            If empty and no ``coefficients_callback`` is set, defaults are
            selected based on the model type.
        coefficients_callback (`Callable[[TeaCacheParams], List[float]]`, *optional*):
            A function that receives this ``TeaCacheParams`` instance and returns
            the polynomial coefficients to use. When set, it takes precedence over
            the ``coefficients`` field, allowing dynamic coefficient selection based
            on any property of the params (e.g., ``use_ret_steps`` for Wan models).
    """

    rel_l1_thresh: float = 0.0
    start_skipping: int = 5
    end_skipping: int = 0
    coefficients: list[float] = field(default_factory=list)
    coefficients_callback: Callable[[TeaCacheParams], list[float]] | None = field(
        default=None, repr=False
    )
    use_ret_steps: bool | None = None

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class TeaCacheParams(CacheParams):
    cache_type: str = "teacache"
    teacache_thresh: float = 0.0
    skip_start_step: int = 5
    skip_end_step: int = 0
    coefficients: list[float] = field(default_factory=list)


@dataclass
class WanTeaCacheParams(CacheParams):
    # Unfortunately, TeaCache is very different for Wan than other models
    cache_type: str = "teacache"
    teacache_thresh: float = 0.08
    skip_start_step: int = 5
    skip_end_step: int = 0
    use_ret_steps: bool = True
    # from original teacache paper https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L883
    ret_steps_coeffs: list[float] = field(
        default_factory=lambda: [
            -5.21862437e04,
            9.23041404e03,
            -5.28275948e02,
            1.36987616e01,
            -4.99875664e-02,
        ]
    )
    # from original teacache paper https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4Wan2.1/teacache_generate.py#L890
    non_ret_steps_coeffs: list[float] = field(
        default_factory=lambda: [
            2.39676752e03,
            -1.31110545e03,
            2.01331979e02,
            -8.29855975e00,
            1.37887774e-01,
        ]
    )

    @property
    def coefficients(self) -> list[float]:
        if self.use_ret_steps:
            return self.ret_steps_coeffs
        else:
            return self.non_ret_steps_coeffs

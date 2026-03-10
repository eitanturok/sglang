# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch


class DiffusionCache(ABC):
    """Base class for diffusion timestep-caching strategies.

    Each timestep caching technique must implement `maybe_reset`,
    `should_skip`, `write`, `read`, and optionally `calibrate`.
    """

    @abstractmethod
    def maybe_reset(self, curr_step: int) -> None:
        """Clear the cached state for a new generation.

        Args:
            curr_step: Current diffusion timestep index.
        """

    @abstractmethod
    def should_skip(self, curr_step: int, **kwargs) -> bool:
        """Decide whether to skip this timestep pass.

        Args:
            curr_step: Current diffusion timestep index.
            **kwargs: Keyword args.
        """

    @abstractmethod
    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        **kwargs
    ) -> None:
        """Cache the result of a full forward pass to the cache state.

        Args:
            hidden_states: Output of transformer blocks.
            original_hidden_states: Input before blocks.
            **kwargs: Keyword args.
        """

    @abstractmethod
    def read(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct output from cache.

        Args:
            hidden_states: Output of transformer blocks.
            original_hidden_states: Input before blocks.
            **kwargs: Keyword args.
        """

    def calibrate(self, **kwargs):
        """Calibrate the values for the cache.

        Args:
            **kwargs
        """
        pass

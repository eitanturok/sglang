# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch


class DiffusionCache(ABC):
    """Base class for diffusion timestep-caching strategies.

    The contract between a CachableDiT and a DiffusionCache:
    1. CachableDiT calls reset() at step 0
    2. CachableDiT calls should_skip() with model-specific
       context to decide whether to run transformer blocks
    3. If skipping:  output = cache.read(hidden_states)
    4. If computing: cache.write(hidden_states, original, context)

    Subclasses define what "context" means for their technique.
    """

    @abstractmethod
    def maybe_reset(self, curr_step: int) -> None:
        """Clear all cached state for a new generation.

        Args:
            curr_step: Current diffusion timestep index.
        """

    @abstractmethod
    def should_skip(self, curr_step: int, **context) -> bool:
        """Decide whether to skip this timestep pass.

        Args:
            curr_step: Current diffusion timestep index.
            **context: Model-specific parameters the caching strategy
                       needs to decide whether to skip.
        """

    @abstractmethod
    def write(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        **context
    ) -> None:
        """Cache the result of a full forward pass to the cache state.

        Args:
            hidden_states: Output of transformer blocks.
            original_hidden_states: Input before blocks.
            **context: Same context passed to should_skip.
        """

    @abstractmethod
    def read(self, hidden_states: torch.Tensor, **context) -> torch.Tensor:
        """Reconstruct output from cache.

        Args:
            hidden_states: Output of transformer blocks.
            original_hidden_states: Input before blocks.
            **context: Same context passed to should_skip.
        """

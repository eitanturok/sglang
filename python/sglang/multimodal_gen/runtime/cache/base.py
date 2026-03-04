# SPDX-License-Identifier: Apache-2.0
"""
Base class for diffusion model cache strategies (TeaCache, MagCache, etc.).
"""

import torch


class DiffusionCache:
    """
    Base class for diffusion model caching strategies.

    Each subclass owns its own state (positive + negative CFG branch) and
    context extraction logic. CachableDiT holds a single
    `self.cache: DiffusionCache | None` and delegates all decisions here.

    Subclasses must implement: reset, get_context, should_skip, maybe_cache,
    retrieve. calibrate is optional (no-op by default).

    Typical forward pass usage in CachableDiT:

        ctx = self.cache.get_context()
        if ctx and self.cache.should_skip(ctx, timestep_proj=..., temb=...):
            hidden_states = self.cache.retrieve(hidden_states, ctx)
        else:
            original_hidden_states = hidden_states.clone()
            # ... run transformer blocks ...
            if calibrate_cache:
                self.cache.calibrate(hidden_states, original_hidden_states, ctx)
            else:
                self.cache.maybe_cache(hidden_states, original_hidden_states, ctx)
    """

    def reset(self) -> None:
        """Reset all state at the start of a new generation."""
        raise NotImplementedError

    def get_context(self):
        """
        Read the global forward_context / forward_batch and return a
        strategy-specific context dataclass, or None to bypass caching.
        """
        raise NotImplementedError

    def should_skip(self, ctx, **kwargs) -> bool:
        """
        Decide whether to skip the transformer forward pass and reuse the
        cached residual. kwargs carries model-specific tensors (e.g.
        timestep_proj, temb) needed by some strategies.
        """
        raise NotImplementedError

    def maybe_cache(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        ctx,
    ) -> None:
        """Store residual after a full forward pass for future reuse."""
        raise NotImplementedError

    def retrieve(self, hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        """Reconstruct output from cached residual."""
        raise NotImplementedError

    def calibrate(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        ctx,
    ) -> None:
        """Log calibration metrics. No-op by default."""
        pass

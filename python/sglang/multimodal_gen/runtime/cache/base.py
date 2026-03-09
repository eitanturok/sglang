# SPDX-License-Identifier: Apache-2.0
"""
Base class for diffusion model cache strategies (TeaCache, MagCache, etc.).
"""

from __future__ import annotations


class DiffusionCache:
    """
    Marker base class for diffusion model timestep caching strategies.

    Note: this is *timestep-level* caching — skipping the full transformer forward
    pass for certain denoising timesteps. It is unrelated to Cache-DiT's block-step
    caching, which operates at the transformer-block level.

    Concrete subclasses (TeaCacheStrategy, etc.) own their own state and implement:
        maybe_reset()                          — call every forward pass; detects new generations
        should_skip(temb, timestep_proj) -> bool  — skip decision
        write(hidden_states, original)         — store residual
        read(hidden_states) -> Tensor          — reconstruct from residual

    is_cfg_negative is read from forward_batch internally via _get_state().

    Typical forward pass usage in CachableDiT:

        # Each forward pass:
        cache.maybe_reset()
        should_skip = cache.should_skip(temb, timestep_proj)
        if should_skip:
            hidden_states = cache.read(is_cfg_negative, hidden_states)
        else:
            original_hidden_states = hidden_states.clone()
            # ... run transformer blocks ...
            cache.write(is_cfg_negative, hidden_states, original_hidden_states)
    """

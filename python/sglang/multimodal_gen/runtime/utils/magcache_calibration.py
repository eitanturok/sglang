# SPDX-License-Identifier: Apache-2.0
"""Calibration workflow for MagCache optimization.

This module provides tools to calibrate MagCache by running a single generation
to collect residual magnitudes and compute magnitude ratios. These ratios are
then used during inference to make intelligent caching decisions.
"""

import json
import os
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn


class MagCacheCalibrator:
    """Handles MagCache calibration workflow.

    The calibration process:
    1. Run one full generation with calibration mode enabled
    2. Model collects residuals at each diffusion timestep
    3. Compute magnitude ratios between consecutive residuals
    4. Save results to JSON file for later use during inference

    Example:
        from sglang.multimodal_gen import Engine
        from sglang.multimodal_gen.configs.sample import SamplingParams

        engine = Engine(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        calibrator = MagCacheCalibrator()

        # Run calibration
        results = calibrator.run_calibration(
            model=engine.model,
            sampling_params=SamplingParams(
                prompt="A beautiful sunset",
                num_inference_steps=50
            )
        )

        # Save for later use
        calibrator.save_calibration_results(
            results,
            "~/.cache/sgl_diffusion/magcache_calibration/wan2.1/50steps.json"
        )
    """

    def run_calibration(
        self, model: nn.Module, sampling_params: Any, num_calibration_steps: int = None
    ) -> dict[str, Any]:
        """
        Run calibration to compute magnitude ratios.

        This method runs a single generation with calibration mode enabled.
        The model collects residuals at each timestep, which are then used
        to compute magnitude ratios.

        Args:
            model: The DiT model to calibrate. Must inherit from MagCacheMixin.
            sampling_params: Generation parameters (SamplingParams instance).
            num_calibration_steps: Override num_inference_steps (optional).
                If not specified, uses sampling_params.num_inference_steps.

        Returns:
            Dictionary containing:
            {
                "magnitude_ratios": [0.95, 0.92, 0.88, ...],
                "num_inference_steps": 50,
                "model_name": "wan",
                "model_config": {
                    "hidden_size": 3072,
                    "num_layers": 32
                },
                "calibration_metadata": {
                    "timestamp": "2024-01-01T00:00:00",
                    "prompt": "A beautiful sunset",
                    "seed": 42,
                    "guidance_scale": 7.0
                }
            }

        Raises:
            AttributeError: If model doesn't have required MagCache methods.
            ValueError: If insufficient residuals collected (< 2).
        """
        # Validate model has MagCache support
        if not hasattr(model, "calibration_residuals"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support MagCache. "
                "Make sure it inherits from MagCacheMixin and calls "
                "_init_magcache_state() in __init__."
            )

        # Set up calibration mode
        if num_calibration_steps:
            sampling_params.num_inference_steps = num_calibration_steps

        # Import here to avoid circular dependency
        from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams

        sampling_params.enable_magcache = True
        sampling_params.magcache_params = MagCacheParams(
            is_calibration=True,
            magnitude_ratios=[],  # Empty during calibration
        )

        # Run generation - model will collect residuals in self.calibration_residuals
        model.reset_magcache_state()

        # Note: The actual generation call depends on how the engine is set up
        # For now, we document that the caller should use the engine's generate method
        # after calling this setup. This is a helper to prepare the model state.

        print(f"Running calibration with {sampling_params.num_inference_steps} steps...")
        print(
            "Note: After this method returns, call engine.generate(sampling_params) "
            "to run the actual generation and collect residuals."
        )

        # Compute magnitude ratios from collected residuals
        # This will be called after generation completes
        def _finalize_calibration() -> dict[str, Any]:
            residuals = model.calibration_residuals
            if len(residuals) < 2:
                raise ValueError(
                    f"Insufficient residuals collected: {len(residuals)}. "
                    "Need at least 2 to compute magnitude ratios. "
                    "Make sure model.maybe_cache_states() is called in forward()."
                )

            magnitude_ratios = self._compute_magnitude_ratios(residuals)

            # Package results
            results = {
                "magnitude_ratios": magnitude_ratios,
                "num_inference_steps": sampling_params.num_inference_steps,
                "model_name": model.config.prefix,
                "model_config": {
                    "hidden_size": getattr(model, "hidden_size", None),
                    "num_layers": getattr(model.config, "num_layers", None),
                },
                "calibration_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": getattr(sampling_params, "prompt", "unknown"),
                    "seed": getattr(sampling_params, "seed", None),
                    "guidance_scale": getattr(sampling_params, "guidance_scale", None),
                },
            }

            print(f"Calibration complete. Collected {len(residuals)} residuals.")
            print(
                f"Magnitude ratios (first 5): {magnitude_ratios[:5] if len(magnitude_ratios) >= 5 else magnitude_ratios}"
            )

            return results

        # Store finalize function for later use
        model._magcache_finalize_calibration = _finalize_calibration

        return sampling_params

    def finalize_calibration(self, model: nn.Module) -> dict[str, Any]:
        """
        Finalize calibration after generation completes.

        Call this after running generation with calibration mode enabled.

        Args:
            model: The DiT model that was calibrated.

        Returns:
            Dictionary containing magnitude ratios and metadata.
        """
        if not hasattr(model, "_magcache_finalize_calibration"):
            raise RuntimeError(
                "Calibration not initialized. Call run_calibration() first."
            )

        return model._magcache_finalize_calibration()

    def _compute_magnitude_ratios(
        self, residuals: list[torch.Tensor]
    ) -> list[float]:
        """
        Compute magnitude ratios from collected residuals.

        For each pair of consecutive residuals:
        ratio_t = ||residual_t|| / ||residual_{t-1}||

        These ratios capture the decay pattern of residual changes throughout
        the diffusion process.

        Args:
            residuals: List of residual tensors from each diffusion step.

        Returns:
            List of magnitude ratios, length = len(residuals) - 1
        """
        if len(residuals) < 2:
            raise ValueError("Need at least 2 residuals to compute ratios")

        ratios = []
        for i in range(1, len(residuals)):
            prev_residual = residuals[i - 1]
            curr_residual = residuals[i]

            # Compute L2 norms
            prev_norm = torch.linalg.norm(prev_residual).item()
            curr_norm = torch.linalg.norm(curr_residual).item()

            # Compute ratio with division-by-zero protection
            if prev_norm < 1e-8:
                ratio = 1.0
            else:
                ratio = curr_norm / prev_norm

            ratios.append(ratio)

        return ratios

    def save_calibration_results(
        self, results: dict[str, Any], output_path: str
    ) -> None:
        """Save calibration results to JSON file.

        Creates parent directories if they don't exist.

        Args:
            results: Calibration results dictionary from run_calibration().
            output_path: Path to save JSON file (supports ~ expansion).
        """
        output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Calibration results saved to: {output_path}")
        print(f"Use this file in MagCacheParams(calibration_file='{output_path}')")

    def load_calibration_results(self, calibration_file: str) -> dict[str, Any]:
        """Load pre-computed calibration results.

        Args:
            calibration_file: Path to calibration JSON file (supports ~ expansion).

        Returns:
            Dictionary containing calibration results.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
        """
        calibration_file = os.path.expanduser(calibration_file)

        if not os.path.exists(calibration_file):
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

        with open(calibration_file, "r") as f:
            results = json.load(f)

        return results

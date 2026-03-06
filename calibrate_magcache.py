#!/usr/bin/env python3
"""
MagCache calibration script for Wan2.1 model.

This script directly builds the pipeline to access the model for calibration,
bypassing the client-server architecture of DiffGenerator.
"""
import sys

sys.path.insert(0, '/home/ubuntu/sglang/python')

import torch

from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams
from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.magcache_calibration import MagCacheCalibrator

# Clear CUDA cache before starting
torch.cuda.empty_cache()

print("Setting up server args with aggressive CPU offloading...")
server_args = ServerArgs(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    # Enable aggressive CPU offloading to avoid OOM
    dit_cpu_offload=False,  # Keep DiT on GPU (needed for calibration)
    text_encoder_cpu_offload=True,  # Offload text encoder
    vae_cpu_offload=True,  # Offload VAE
)

print("Building pipeline...")
try:
    pipeline = build_pipeline(server_args)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"\n❌ CUDA Out of Memory Error: {e}")
        print("\nTry one of these solutions:")
        print("1. Free GPU memory: nvidia-smi to check usage")
        print("2. Enable more offloading (vae, text_encoder)")
        print("3. Use a machine with more VRAM")
        sys.exit(1)
    raise

print("Accessing DiT model from pipeline...")
# The DiT model should be accessible from the pipeline
# Common names: transformer, video_dit, dit
model = None
for attr_name in ['transformer', 'video_dit', 'dit', 'transformer_2']:
    if hasattr(pipeline, attr_name):
        candidate = getattr(pipeline, attr_name)
        if candidate is not None:
            model = candidate
            print(f"Found model at pipeline.{attr_name}")
            break

if model is None:
    print("ERROR: Could not find DiT model in pipeline")
    print(f"Pipeline has attributes: {[a for a in dir(pipeline) if not a.startswith('_')]}")
    sys.exit(1)

print(f"Model type: {type(model).__name__}")
print(f"Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'unknown'}")

# Check MagCache support
print("\nChecking MagCache support...")
if not hasattr(model, 'reset_magcache_state'):
    print("❌ ERROR: Model does not have reset_magcache_state method")
    print("   Model may not inherit from MagCacheMixin")
    sys.exit(1)

if not hasattr(model, 'calibration_residuals'):
    print("❌ ERROR: Model does not have calibration_residuals attribute")
    print("   Model may not inherit from MagCacheMixin")
    sys.exit(1)

print("✅ Model has MagCache support!")

print("\nPreparing calibration...")
calibrator = MagCacheCalibrator()

# Set up calibration sampling params
sampling_params = SamplingParams(
    prompt="A beautiful sunset",
    num_inference_steps=5,  # Just 5 steps for quick calibration
    guidance_scale=3.0,
)

print("Setting up calibration mode...")
calibrator_result = calibrator.run_calibration(
    model=model,
    sampling_params=sampling_params,
    num_calibration_steps=5
)

print("\n" + "="*60)
print("⚠️  NEXT STEP: Run the actual generation")
print("="*60)
print("\nThe calibration is now set up, but we need to run the actual")
print("generation through the pipeline to collect residuals.")
print("\nTo complete calibration, you need to:")
print("1. Call pipeline's generation method with calibrator_result")
print("2. Then call calibrator.finalize_calibration(model)")
print("\nThis requires understanding how to invoke the WanPipeline")
print("generation method directly, which may need:")
print("  - prepare_request()")
print("  - pipeline.__call__() or pipeline.generate()")
print("\n" + "="*60)

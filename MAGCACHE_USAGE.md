# MagCache Implementation Guide

This document explains how to use MagCache with SGLang for accelerated diffusion model inference.

## What is MagCache?

MagCache (Magnitude Caching) is an acceleration technique for diffusion models that achieves 2-3× speedup by intelligently skipping redundant computation. It uses pre-computed magnitude ratios of residual changes to decide when to reuse cached outputs instead of recomputing.

**Key Advantages:**
- **Lower calibration cost**: Only requires 1 random prompt (vs dozens for other methods)
- **Better quality preservation**: Maintains output quality while accelerating
- **Model-agnostic**: Works across different diffusion models

**Reference:**
- Paper: https://zehong-ma.github.io/MagCache/
- HuggingFace PR: https://github.com/huggingface/diffusers/pull/12744

## Implementation Status

### ✅ Completed Components

1. **Core Implementation**
   - [magcache.py](python/sglang/multimodal_gen/runtime/cache/magcache.py) - MagCacheMixin with magnitude ratio logic
   - [magcache.py](python/sglang/multimodal_gen/configs/sample/magcache.py) - MagCacheParams configuration
   - [magcache_calibration.py](python/sglang/multimodal_gen/runtime/utils/magcache_calibration.py) - Calibration workflow

2. **Base Classes**
   - MagCacheDiT in [base.py](python/sglang/multimodal_gen/runtime/models/dits/base.py)
   - MagCacheMixin integrated into WanTransformer3DModel

3. **Configuration**
   - Environment variables in [envs.py](python/sglang/multimodal_gen/envs.py)
   - Sampling parameters in [sampling_params.py](python/sglang/multimodal_gen/configs/sample/sampling_params.py)
   - Req dataclass fields in [schedule_batch.py](python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py)

4. **Model Integration**
   - Wan2.1 model ([wanvideo.py](python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py)) with dual TeaCache/MagCache support

### 🔄 Pending

- End-to-end testing with Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Calibration file generation and validation
- Performance benchmarking (speedup measurement)
- Quality validation (PSNR, SSIM, visual comparison)
- Additional model integrations (Flux, HunyuanVideo, etc.)

## Usage Workflow

### Phase 1: Calibration (One-Time Setup)

Run calibration once for each model/steps configuration:

```python
from sglang.multimodal_gen import Engine
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.runtime.utils.magcache_calibration import MagCacheCalibrator

# Initialize engine
engine = Engine(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Create calibrator
calibrator = MagCacheCalibrator()

# Prepare calibration sampling params
calibration_params = calibrator.run_calibration(
    model=engine.model,
    sampling_params=SamplingParams(
        prompt="A beautiful sunset over the ocean",  # Any random prompt works
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=42  # Optional: for reproducibility
    )
)

# Run generation (model will collect residuals)
output = engine.generate(calibration_params)

# Finalize calibration and compute magnitude ratios
results = calibrator.finalize_calibration(engine.model)

# Save calibration results
calibrator.save_calibration_results(
    results,
    "~/.cache/sgl_diffusion/magcache_calibration/wan2.1/50steps.json"
)

print(f"Calibration complete!")
print(f"Magnitude ratios: {results['magnitude_ratios'][:5]}...")
```

**Calibration Output Format:**
```json
{
  "magnitude_ratios": [0.95, 0.92, 0.88, 0.85, 0.82, ...],
  "num_inference_steps": 50,
  "model_name": "wan",
  "model_config": {
    "hidden_size": 3072,
    "num_layers": 32
  },
  "calibration_metadata": {
    "timestamp": "2024-01-01T00:00:00",
    "prompt": "A beautiful sunset over the ocean",
    "seed": 42,
    "guidance_scale": 3.0
  }
}
```

### Phase 2: Accelerated Inference

Use pre-computed magnitude ratios for fast generation:

```python
from sglang.multimodal_gen import Engine
from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.configs.sample.magcache import MagCacheParams

# Initialize engine
engine = Engine(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

# Configure MagCache with pre-computed ratios
sampling_params = SamplingParams(
    prompt="A cat playing piano in a jazz club",
    num_inference_steps=50,
    guidance_scale=3.0,
    enable_magcache=True,
    magcache_params=MagCacheParams(
        calibration_file="~/.cache/sgl_diffusion/magcache_calibration/wan2.1/50steps.json",
        skip_threshold=0.1  # Adjust for speed/quality tradeoff
    )
)

# Generate with MagCache acceleration
output = engine.generate(sampling_params)
print(f"Generated: {output.file_paths}")
```

## Configuration Options

### Environment Variables

```bash
# Enable MagCache globally
export SGLANG_MAGCACHE_ENABLED=true

# Calibration mode (for calibration runs)
export SGLANG_MAGCACHE_CALIBRATION_MODE=true

# Path to pre-computed calibration file
export SGLANG_MAGCACHE_RATIO_FILE=~/.cache/sgl_diffusion/magcache_calibration/wan2.1/50steps.json

# Skip threshold (lower = faster, potentially lower quality)
export SGLANG_MAGCACHE_SKIP_THRESHOLD=0.1
```

### MagCacheParams Fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `magnitude_ratios` | `list[float]` | `[]` | Pre-computed magnitude ratios from calibration |
| `skip_threshold` | `float` | `0.1` | Threshold for skip decision (typical: 0.05-0.15) |
| `calibration_file` | `str | None` | `None` | Path to calibration JSON file |
| `is_calibration` | `bool` | `False` | Whether this is a calibration run |

### Skip Threshold Tuning

The `skip_threshold` controls the speed/quality tradeoff:

- **Lower values (0.05-0.08)**: More aggressive caching
  - Faster inference
  - Potentially slight quality degradation

- **Medium values (0.1-0.12)**: Balanced
  - Good speedup (2-3×)
  - Minimal quality loss

- **Higher values (0.15-0.20)**: Conservative
  - Moderate speedup
  - Maximum quality preservation

## CLI Usage

```bash
# Enable MagCache via command line
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A cat playing piano" \
  --num-inference-steps 50 \
  --enable-magcache \
  --output-file output.mp4
```

## Calibration Directory Structure

Recommended organization for calibration files:

```
~/.cache/sgl_diffusion/magcache_calibration/
├── wan2.1/
│   ├── 28steps.json
│   ├── 50steps.json
│   └── 100steps.json
├── flux/
│   ├── 28steps.json
│   └── 50steps.json
└── hunyuan/
    └── 40steps.json
```

## Model Integration Guide

### For New Models

To add MagCache support to a new model:

1. **Inherit from MagCacheMixin:**
   ```python
   from sglang.multimodal_gen.runtime.cache.magcache import MagCacheMixin

   class MyDiTModel(MagCacheMixin, BaseDiT):
       def __init__(self, config, **kwargs):
           super().__init__(config, **kwargs)
           self._init_magcache_state()
   ```

2. **Add caching logic to forward():**
   ```python
   def forward(self, hidden_states, timestep, ...):
       ctx = self._get_magcache_context()

       # Calibration mode
       if ctx is not None and ctx.is_calibration:
           original_hidden_states = hidden_states.clone()
           hidden_states = self._transformer_forward(...)
           self.maybe_cache_states(hidden_states, original_hidden_states)
           return hidden_states

       # Inference mode with skip check
       if ctx is not None and not ctx.is_calibration:
           should_skip = self._should_skip_using_magnitude_ratio(
               current_timestep=ctx.current_timestep,
               magnitude_ratios=ctx.magnitude_ratios,
               skip_threshold=ctx.skip_threshold
           )
           if should_skip:
               return self.retrieve_cached_states(hidden_states)

       # Normal forward
       original_hidden_states = hidden_states.clone()
       hidden_states = self._transformer_forward(...)
       if ctx is not None:
           self.maybe_cache_states(hidden_states, original_hidden_states)

       return hidden_states
   ```

3. **CFG Support (Optional):**
   - Add model prefix to `_CFG_SUPPORTED_PREFIXES` in [magcache.py](python/sglang/multimodal_gen/runtime/cache/magcache.py)
   - Separate caches will be maintained for positive/negative CFG branches

## Mutual Exclusivity with Other Caching

MagCache, TeaCache, and Cache-DIT are **mutually exclusive**. Priority order:

1. **Cache-DIT** (highest priority)
2. **MagCache**
3. **TeaCache**

If multiple caching methods are enabled, only the highest priority one will be active, and a warning will be logged.

## Performance Tips

1. **Calibration Best Practices:**
   - Use any random prompt (doesn't need to match your actual use case)
   - Calibrate once per (model, num_inference_steps) configuration
   - Reuse calibration files across different prompts

2. **Threshold Tuning:**
   - Start with default (0.1)
   - If quality is acceptable, try lowering (0.05-0.08) for more speed
   - If quality degrades, increase (0.12-0.15)

3. **Storage:**
   - Calibration files are small (< 10KB)
   - Can be shared across machines/users
   - Commit to version control for reproducibility

## Troubleshooting

### Issue: "Calibration file not found"
- Ensure you ran calibration first
- Check path uses correct expansion (`~` vs absolute path)
- Verify file exists: `ls -la ~/.cache/sgl_diffusion/magcache_calibration/`

### Issue: "Insufficient residuals collected"
- Make sure model inherits from MagCacheMixin
- Verify `_init_magcache_state()` is called in `__init__`
- Check `maybe_cache_states()` is called in forward pass

### Issue: "Quality degradation"
- Increase skip_threshold (e.g., from 0.1 to 0.15)
- Recalibrate with more steps
- Verify calibration was done with same model/config

## Next Steps

1. **Test Calibration:**
   - Run calibration script with Wan2.1
   - Verify magnitude ratios are generated
   - Check calibration file format

2. **Test Inference:**
   - Use pre-computed ratios for generation
   - Compare output with/without MagCache
   - Measure speedup

3. **Benchmark:**
   - Test different skip_threshold values
   - Measure inference time
   - Evaluate output quality (visual + metrics)

4. **Expand:**
   - Add MagCache support to Flux, HunyuanVideo, etc.
   - Create pre-computed calibration files for common configs
   - Document best practices per model

## References

- **MagCache Paper**: https://zehong-ma.github.io/MagCache/
- **HuggingFace Implementation**: https://github.com/huggingface/diffusers/pull/12744
- **SGLang Diffusion Docs**: https://docs.sglang.io/supported_models/image_generation/diffusion_models.html
- **Implementation Plan**: /home/ubuntu/.claude/plans/adaptive-nibbling-swan.md

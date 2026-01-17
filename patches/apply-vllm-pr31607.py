#!/usr/bin/env python3
"""
Apply vLLM PR #31607 patches for SM 12.1 (Blackwell GB10) support.

This script patches vLLM to add exception handling for CUTLASS operations
that may not be available on SM 12.1 (GB10) GPUs. Without these patches,
FP8 models hang during kernel initialization on DGX Spark.

Reference: https://github.com/vllm-project/vllm/pull/31607
"""

import os
import re
import sys

VLLM_DIR = os.environ.get('VLLM_DIR', '/build/vllm')


def patch_custom_ops():
    """Patch _custom_ops.py to add exception handling for CUTLASS operations."""
    filepath = os.path.join(VLLM_DIR, 'vllm/_custom_ops.py')
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'cutlass_scaled_mm_supports_fp8 unavailable' in content:
        print(f"SKIP: {filepath} already patched")
        return True

    # Patch cutlass_scaled_mm_supports_fp8
    old = '''def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)'''

    new = '''def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    try:
        return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)
    except Exception as e:
        import warnings
        warnings.warn(f"cutlass_scaled_mm_supports_fp8 unavailable: {e}")
        return False'''

    if old in content:
        content = content.replace(old, new)
        print(f"PATCHED: cutlass_scaled_mm_supports_fp8 in {filepath}")
    else:
        print(f"WARNING: Could not find cutlass_scaled_mm_supports_fp8 pattern in {filepath}")

    # Patch cutlass_scaled_mm_supports_block_fp8
    old2 = '''def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(
        cuda_device_capability)'''

    new2 = '''def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    try:
        return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(
            cuda_device_capability)
    except Exception as e:
        import warnings
        warnings.warn(f"cutlass_scaled_mm_supports_block_fp8 unavailable: {e}")
        return False'''

    if old2 in content:
        content = content.replace(old2, new2)
        print(f"PATCHED: cutlass_scaled_mm_supports_block_fp8 in {filepath}")
    else:
        print(f"WARNING: Could not find cutlass_scaled_mm_supports_block_fp8 pattern in {filepath}")

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def patch_mxfp4():
    """Patch mxfp4.py to extend device capability range to include SM 12.1."""
    filepath = os.path.join(VLLM_DIR, 'vllm/model_executor/layers/quantization/mxfp4.py')
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if '<= (12, 1)' in content:
        print(f"SKIP: {filepath} already patched")
        return True

    # Replace capability check: < (11, 0) -> <= (12, 1)
    # This extends support from H100/H200 (sm_90) to Blackwell (sm_120, sm_121)
    old_pattern = r'< \(11, 0\)'
    new_pattern = '<= (12, 1)'

    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        print(f"PATCHED: device capability range in {filepath}")
    else:
        print(f"WARNING: Could not find capability pattern in {filepath}")

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def patch_w8a8_utils():
    """Patch w8a8_utils.py to wrap CUTLASS flags with exception handling."""
    filepath = os.path.join(VLLM_DIR, 'vllm/model_executor/layers/quantization/utils/w8a8_utils.py')
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'CUTLASS_FP8_SUPPORTED = False' in content and 'except Exception' in content:
        print(f"SKIP: {filepath} already patched")
        return True

    # Patch CUTLASS_FP8_SUPPORTED
    old = 'CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()'
    new = '''try:
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
except Exception:
    CUTLASS_FP8_SUPPORTED = False'''

    if old in content:
        content = content.replace(old, new)
        print(f"PATCHED: CUTLASS_FP8_SUPPORTED in {filepath}")
    else:
        print(f"WARNING: Could not find CUTLASS_FP8_SUPPORTED pattern in {filepath}")

    # Patch CUTLASS_BLOCK_FP8_SUPPORTED
    old2 = 'CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()'
    new2 = '''try:
    CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()
except Exception:
    CUTLASS_BLOCK_FP8_SUPPORTED = False'''

    if old2 in content:
        content = content.replace(old2, new2)
        print(f"PATCHED: CUTLASS_BLOCK_FP8_SUPPORTED in {filepath}")
    else:
        print(f"WARNING: Could not find CUTLASS_BLOCK_FP8_SUPPORTED pattern in {filepath}")

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def patch_matcher_utils():
    """Patch matcher_utils.py to add hasattr checks for torch custom ops."""
    filepath = os.path.join(VLLM_DIR, 'vllm/compilation/matcher_utils.py')
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'hasattr(torch.ops._C, "silu_and_mul")' in content:
        print(f"SKIP: {filepath} already patched")
        return True

    patched = False

    # Patch SILU_MUL_OP
    old = 'SILU_MUL_OP = torch.ops._C.silu_and_mul.default'
    new = 'SILU_MUL_OP = torch.ops._C.silu_and_mul.default if hasattr(torch.ops._C, "silu_and_mul") else None'

    if old in content:
        content = content.replace(old, new)
        print(f"PATCHED: SILU_MUL_OP in {filepath}")
        patched = True

    # Patch FP8 quant ops
    fp8_patterns = [
        ('torch.ops._C.static_scaled_fp8_quant.default',
         'torch.ops._C.static_scaled_fp8_quant.default if hasattr(torch.ops._C, "static_scaled_fp8_quant") else None'),
        ('torch.ops._C.dynamic_scaled_fp8_quant.default',
         'torch.ops._C.dynamic_scaled_fp8_quant.default if hasattr(torch.ops._C, "dynamic_scaled_fp8_quant") else None'),
        ('torch.ops._C.dynamic_per_token_scaled_fp8_quant.default',
         'torch.ops._C.dynamic_per_token_scaled_fp8_quant.default if hasattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant") else None'),
    ]

    for old_pat, new_pat in fp8_patterns:
        # Only replace if not already wrapped with hasattr
        if old_pat in content and 'hasattr' not in content.split(old_pat)[0].split('\n')[-1]:
            content = content.replace(old_pat, new_pat)
            print(f"PATCHED: {old_pat.split('.')[-2]} in {filepath}")
            patched = True

    if not patched:
        print(f"WARNING: No patterns found to patch in {filepath}")

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def main():
    print("=" * 60)
    print("Applying vLLM PR #31607 patches for SM 12.1 (Blackwell GB10)")
    print("=" * 60)
    print(f"VLLM_DIR: {VLLM_DIR}")
    print()

    results = []
    results.append(("_custom_ops.py", patch_custom_ops()))
    results.append(("mxfp4.py", patch_mxfp4()))
    results.append(("w8a8_utils.py", patch_w8a8_utils()))
    results.append(("matcher_utils.py", patch_matcher_utils()))

    print()
    print("=" * 60)
    print("Summary:")
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    failed = [name for name, success in results if not success]
    if failed:
        print(f"\nWARNING: Some patches failed: {', '.join(failed)}")
        return 1

    print("\nAll patches applied successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

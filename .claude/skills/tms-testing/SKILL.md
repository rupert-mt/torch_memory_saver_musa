---
name: tms-testing
description: Use when running ordinary torch_memory_saver regression tests or the full CUDA wheel runtime matrix.
---

# 1 Ordinary tests

- Use an isolated checkout on a Linux GPU host, such as `tom-workstation`, in an official PyTorch CUDA devel container with an idle GPU exposed. Install `pytest`, `nvidia-ml-py`, and `setuptools` in the same Python environment as PyTorch.
- From the repository root, rebuild the extensions for the container's CUDA major, then run all ordinary tests. This example uses CUDA 12; set `TMS_CUDA_MAJOR=13` for CUDA 13:

```bash
TMS_CUDA_MAJOR=12 uv run --no-project python setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0 timeout 3600 uv run --no-project python -m pytest test -vv -ra
```

- The suite parametrizes supported scenarios over `preload` and `torch` hook modes. Single-GPU runs skip multi-device cases; report skips with the passing summary.
- If preload children cannot find `libcudart.so.<major>`, prepend the matching runtime directory to `LD_LIBRARY_PATH` before pytest; pip CUDA runtimes may be outside the loader's default search path.
- For build/release tooling changes, also run:

```bash
uv run --no-project --with pytest --with packaging --with typer --with setuptools python -m pytest \
  test/test_merge_cuda_wheels.py \
  .claude/skills/tms-publish-release/scripts/test_*.py -vv -ra
```

# 2 MUSA tests

- Use a MUSA host/container with `pytorch_musa` and a MUSA toolkit (`mcc` on `PATH` or `MUSA_HOME`), then build in place and run the same suite against a visible device:

```bash
python setup.py build_ext --inplace
MUSA_VISIBLE_DEVICES=0 python -m pytest test -vv -ra
```

- Both hook modes are exercised. `test/test_musa_backend.py` and the `test_musa_*` cases cover MUSA-only behavior (resume zeroing, per-device disable pools, `torch.cuda` aliasing); the CUDA loader/RPATH tests skip because MUSA binaries are unsuffixed. `sum()` on `uint8` tensors fails in muDNN, so examples assert with float tensors or `.tolist()`.

# 3 Full matrix tests

- Run the ordinary and tooling tests above, then use [tms-publish-release](../tms-publish-release/SKILL.md) Sections 2–5 for isolated paths, Docker/ARM64 binfmt setup, both wheel builds, and fresh GPU containers on `tom-workstation`.
- For development testing, use the target commit and its `setup.py` version; release-only requirements to use merged `origin/master`, check PyPI, or publish do not apply.
- After building both wheels into `dist/`, run on the GPU host with the paths prepared above:

```bash
uv run --script .claude/skills/tms-publish-release/scripts/gpu_validation.py \
  --release-root "$TMS_REMOTE_ROOT" \
  --artifact-root "$TMS_REMOTE_ARTIFACTS" \
  --expected-version "$TMS_RELEASE_VERSION"
```

- Require all four cells to pass: x86_64/ARM64 × CUDA 12/13. The harness tests installed wheels outside the checkout and enforces exact skips; do not deselect failing cases.
- When adding `test/test_*.py`, update the harness runtime/build-tool inventory and run every classified build-tool module; unclassified modules fail preflight.
- Multi-device, XPU, and Lupine exclusions remain coverage limits. Validate affected backends on suitable hardware separately.
- For either chapter, save complete build/test output in unique artifact logs, run long commands in the background, and inspect progress at least every ten minutes.

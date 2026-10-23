"""MUSA-only backend paths: torch.cuda aliasing and unsuffixed preload binary."""

import pytest

from torch_memory_saver import utils


pytestmark = pytest.mark.skipif(not utils.is_musa(), reason="MUSA-only backend path")


def test_torch_cuda_entry_points_alias_musa():
    """The library keeps CUDA-shaped call sites; on MUSA they must resolve to torch.musa."""
    import torch

    import torch_memory_saver  # noqa: F401  (patches torch.cuda at import time)

    for name in ("MemPool", "use_mem_pool", "graph", "empty_cache"):
        assert getattr(torch.cuda, name) is getattr(torch.musa, name), name


def test_preload_binary_is_unsuffixed():
    try:
        path = utils.get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
    except RuntimeError:
        pytest.skip("preload hook binary is not built in this tree")
    assert path.name == "torch_memory_saver_hook_mode_preload.abi3.so"

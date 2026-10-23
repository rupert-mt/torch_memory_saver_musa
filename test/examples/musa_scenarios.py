"""MUSA-only lifecycle checks that CUDA/ROCm/XPU do not share.

The MUSA backend makes three behaviors observable:
1. resume() zeroes the remapped region, while CUDA leaves new pages undefined;
2. disable() reuses one MemPool per device, because a dropped MUSA MemPool does
   not return its cached blocks to the device;
3. torch_musa aborts in empty_cache() while a MemPool is active, so disable()
   neutralizes empty_cache for the duration of the scope.
"""

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device, get_device_module


def run_resume_zeroes_unbacked_region(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device_module = get_device_module()
    device_module.set_device(0)
    device = get_device()

    with torch_memory_saver.region(tag="scratch", enable_cpu_backup=False):
        tensor = torch.full((1024 * 1024,), 7.0, dtype=torch.float32, device=device)
    device_module.synchronize()
    assert tensor.cpu().numpy().any(), "fresh region should keep the written pattern"

    torch_memory_saver.pause("scratch")
    device_module.synchronize()
    torch_memory_saver.resume("scratch")
    device_module.synchronize()

    assert not tensor.cpu().numpy().any(), "MUSA resume must zero a region without CPU backup"
    print("resume zeroed the unbacked region")


def run_disable_reuses_device_pool(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device_module = get_device_module()
    device_module.set_device(0)
    device = get_device()

    empty_cache = device_module.empty_cache
    pools = []
    with torch_memory_saver.region(tag="kv_cache", enable_cpu_backup=True):
        resident = torch.full((1024 * 1024,), 3.0, dtype=torch.float32, device=device)
        device_module.synchronize()
        torch_memory_saver.pause("kv_cache")
        impl = torch_memory_saver._impl

        for _ in range(2):
            with torch_memory_saver.disable():
                assert device_module.empty_cache is not empty_cache, (
                    "disable() must neutralize empty_cache while the pool is active"
                )
                device_module.empty_cache()
                scratch = torch.full((64 * 1024 * 1024,), 1.0, dtype=torch.float32, device=device)
                del scratch
            assert device_module.empty_cache is empty_cache, "disable() must restore empty_cache"
            pools.append(impl._disable_mem_pools[impl._current_device()])

        assert pools[0] is pools[1], "disable() must reuse the per-device pool on MUSA"

        torch_memory_saver.resume("kv_cache")
        device_module.synchronize()

    assert resident.cpu().tolist() == [3.0] * 1024 * 1024
    print("disable() reused the per-device pool and kept the resident region intact")


if __name__ == "__main__":
    import sys

    globals()[sys.argv[1]](sys.argv[2])

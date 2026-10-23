"""TMS_TRACE_REGIONS must emit one line per region operation (alloc/pause/resume)."""

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device, get_device_module


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    device_module = get_device_module()
    device_module.set_device(0)

    with torch_memory_saver.region(tag="traced", enable_cpu_backup=True):
        tensor = torch.full((4096,), 1.0, dtype=torch.float32, device=get_device())
    device_module.synchronize()

    torch_memory_saver.pause("traced")
    torch_memory_saver.resume("traced")
    device_module.synchronize()

    assert torch.equal(tensor.cpu(), torch.full((4096,), 1.0))


if __name__ == "__main__":
    import sys

    run(sys.argv[1])

#pragma once

namespace MUSAImplementation {
    // MCCL and muDNN own internal buffers that cannot live in MUSA VMM memory:
    // with TMS-backed (muMemCreate/muMemMap) buffers MCCL either aborts with an
    // unhandled musa error or makes later muDNN kernel launches fail with
    // "invalid argument". Keep those library-internal allocations on the normal
    // runtime allocator, so only PyTorch caching-allocator segments are tracked.
    bool caller_requires_plain_malloc(void* caller);
}

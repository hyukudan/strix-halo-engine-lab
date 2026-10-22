#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>

namespace strix::dist {
// Device buffer -> mapped pinned host alias, byte exact, arbitrary alignment/length.
// Caller records/observes a HIP event on this stream before any CPU/DMA consumer.
// Each producer thread releases its writes to system scope before kernel completion.
void enqueue_host_d2h_copy(void* mapped_host, const void* device, std::size_t bytes, hipStream_t stream);
}

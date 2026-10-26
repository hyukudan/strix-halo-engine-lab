#pragma once
#include "strix/dist/staged_ready.hpp"
#include <cstddef>

namespace strix::dist {
// Probe/qualification primitives. Production producers/consumers publish from
// their own epilogues instead of adding these standalone kernels.
void enqueue_staged_pattern(ProducerStage view, std::size_t bytes, std::uint32_t pattern, hipStream_t stream);
void enqueue_staged_consume(ConsumerStage view, void* output, std::size_t bytes, hipStream_t stream);
// Qualification of read-only RX release, GPU-owned output only. Optional grid
// override exercises padding/empty blocks; 0 uses the ordinary proof grid.
void enqueue_staged_consume_reads(ConsumerStage view, void* output, std::size_t bytes, hipStream_t stream,
                                 unsigned grid_blocks = 0);
}

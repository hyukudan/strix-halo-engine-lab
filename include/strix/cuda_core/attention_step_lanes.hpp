#pragma once

// Prepared BF16 attention over the lanes of a replayed multi-request decode step (strix/state/lane_view.hpp).
//
// Same contract as append_attention_prepared_bf16_lanes (strix/cuda_core/attention.hpp): lane l is packed row
// l of the images, and each lane's cache bytes and output bits are those of append_attention_prepared_bf16 on
// that row alone. The difference is where a lane comes from: its position and its cache addresses are read on
// the device from the lane columns and the slot directory, so one captured launch set serves every request set
// of the program width. Padded rows (valid bit clear) store nothing; their flash reads the slot they repeat.
// The host admits the geometry (dims, shape, images, plane alignment, width); positions are the step builder's
// responsibility (exec::make_cohort_step refuses positions at or past the context).

#include "strix/cuda_core/attention.hpp"
#include "strix/state/lane_view.hpp"

#include <cstdint>

namespace strix::cuda_core {

struct PreparedBf16StepLanes {
    state::LaneView lanes;
    state::LanePlane keys;    // this visit's BF16 key cache inside a slot
    state::LanePlane values;  // this visit's BF16 value cache inside a slot
    std::uint32_t cache_slots;   // rows of each cache (the visit layout, equal for every slot)
    std::uint32_t wrapped;       // nonzero: row = absolute position % cache_slots
};

[[nodiscard]] bool can_launch_prepared_bf16_step_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                       const PreparedBf16LaneImages& images,
                                                       const PreparedBf16StepLanes& lanes, FlashPipe pipe);
// One cache store (padded rows skipped), one flash grid (heads x width) and, with `failure`, one finite check
// over the packed rows. EXACT: every valid lane's bits equal the one-row call at its position on its caches.
cudaError_t append_attention_prepared_bf16_step_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                      const PreparedBf16LaneImages& images,
                                                      const PreparedBf16StepLanes& lanes, cudaStream_t stream,
                                                      FlashPipe pipe = FlashPipe::pad2, unsigned* failure = nullptr);

}  // namespace strix::cuda_core

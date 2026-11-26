#pragma once

#include "../../core/select_segments.hpp"

#include <cstdint>

struct CUstream_st;
struct float2;

namespace strix::cuda::amdmath {
struct Tables;
}

namespace strix::cuda::qwen_qsa {
struct LayerWeights;
struct MultiShape;
struct RowBuffers;
struct SegmentMap;
struct SegmentState;
struct Workspace;

using ScoreMap = core::select::SegmentScoreMap<8>;

template <class StateMap>
struct ReplayDescriptor {
    StateMap state;
    ScoreMap score;
};

// Keep the packed score map and the Qwen layer metadata in the same ordered
// host-to-device copy. The score and picker nodes use the same device member.
using ReplaySegmentMap = ReplayDescriptor<SegmentMap>;

void layer_core_device(const RowBuffers& rows, const ReplaySegmentMap* device_map, const MultiShape& shape,
    const LayerWeights& weights, const float2* rope, const amdmath::Tables& tables, Workspace& workspace, CUstream_st* stream);
void prepare_replay(const SegmentState* segments, std::uint32_t count, const Workspace& workspace, ReplaySegmentMap& map);

}  // namespace strix::cuda::qwen_qsa

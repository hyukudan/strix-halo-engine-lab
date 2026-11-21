#pragma once

namespace strix::inkling2 {

inline constexpr unsigned kMaxContext = 1U << 20U;
inline constexpr unsigned kSlidingWindow = 1024;
inline constexpr unsigned kHiddenWidth = 2304;
inline constexpr unsigned kVocabulary = 201024;
inline constexpr unsigned kOutputVocabulary = 200058;
inline constexpr unsigned kMiniEosToken = 200006;
inline constexpr float kLogitSoftcap = 30.0F;
inline constexpr unsigned kHeadWidth = 128;
inline constexpr unsigned kQueryHeads = 24;
inline constexpr unsigned kKvHeads = 8;
inline constexpr unsigned kQueryWidth = kQueryHeads * kHeadWidth;
inline constexpr unsigned kKvWidth = kKvHeads * kHeadWidth;
inline constexpr unsigned kRelativeWidth = 16;
inline constexpr unsigned kRelativeExtent = 1024;
inline constexpr unsigned kConvTaps = 4;
inline constexpr unsigned kExperts = 128;
inline constexpr unsigned kExpertsPerToken = 8;
inline constexpr unsigned kExpertWidth = 1152;
inline constexpr unsigned kMxFp8First = 20;
inline constexpr unsigned kControllerRank = 64;
inline constexpr float kRouteScale = 8.0F;
inline constexpr float kMlpGateSoftcap = 4.0F;
inline constexpr float kMlpUpSoftcap = 25.0F;

enum class MiniExpertProjection { gate_up, down };

struct MiniExpertGeometry {
    unsigned inputs;
    unsigned outputs;
};

constexpr MiniExpertGeometry mini_expert_geometry(MiniExpertProjection projection) {
    if (projection == MiniExpertProjection::gate_up) return {kHiddenWidth, 2U * kExpertWidth};
    if (projection == MiniExpertProjection::down) return {kExpertWidth, kHiddenWidth};
    return {};
}

}  // namespace strix::inkling2

#pragma once
// Model dimensions shared by device arithmetic and host-only state layout.
namespace strix::glm {
inline constexpr unsigned kHidden = 4096, kHc = 4, kHcMix = (2 + kHc) * kHc, kHcIters = 20;
inline constexpr unsigned kLayers = 45, kMtpLayer = 45, kFirstMoe = 3, kVocab = 154880;
inline constexpr unsigned kKdaHeads = 64, kKdaDim = 128, kKdaQkv = kKdaHeads * kKdaDim, kConvK = 4;
inline constexpr unsigned kQLora = 1536, kKvLora = 512, kMlaHeads = 64, kMlaDim = 256;
inline constexpr unsigned kIdxHeads = 32, kIdxDim = 128, kKpool = 4, kIdxTopk = 2048, kIdxPools = kIdxTopk / kKpool;
inline constexpr unsigned kExperts = 288, kTopE = 8, kExpertI = 2048, kDenseI = 12288;
}  // namespace strix::glm

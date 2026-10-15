#pragma once
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/execution_plan.hpp"
#include <cstdint>
#include <vector>
namespace strix {
inline constexpr const char* lane_chain_hyper_output_suffix=".lane-chain-v1";
[[nodiscard]] std::uint64_t lane_chain_hyper_output_lds(std::uint64_t streams,std::uint64_t rank) noexcept;
[[nodiscard]] bool lane_chain_hyper_output_eligible(const ManifestTensor&,std::uint64_t hidden,std::uint64_t streams,std::uint64_t rank,bool registered) noexcept;
[[nodiscard]] std::vector<std::uint32_t> lane_chain_hyper_output_steps(const ModelContract&,const ModelManifest&,bool registered);
}

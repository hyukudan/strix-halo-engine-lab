#pragma once

#include "strix/model_manifest.hpp"

#include <stdexcept>
#include <memory>

namespace strix {

inline void validate_wave_router_selection_contract(
    std::uint64_t experts, std::uint64_t top_k, bool parallel, bool ranked) {
    if (!parallel || ranked || experts == 0 || experts > 512 ||
        top_k == 0 || top_k > 32 || top_k > experts)
        throw std::invalid_argument("wave router requires parallel E<=512, K<=32 and excludes ranked mode");
}

// Raw GGUF metadata is available at intake, unlike the shape-only ModelContract.
inline void validate_wave_router_model_inputs(const gguf::FileInfo& info) {
    for (const auto& entry : info.metadata)
        if (entry.key.ends_with(".expert_group_count") || entry.key.ends_with(".expert_group_used_count"))
            throw std::invalid_argument("wave router does not support expert groups");
    for (const auto& tensor : info.tensors)
        if (tensor.name.find(".exp_probs_b") != std::string::npos)
            throw std::invalid_argument("wave router does not support expert selection bias");
}

inline void validate_wave_router_manifest(const ModelManifest& manifest) {
    for (const auto& tensor : manifest.tensors)
        if (tensor.name.find(".exp_probs_b") != std::string::npos)
            throw std::invalid_argument("wave router does not support expert selection bias");
}

// A private deep copy gives dispatch a real immutable owner, rather than a
// cache keyed by a caller-controlled layout ID on a mutable manifest. Preflight
// validates the complete names/layout once; dispatch only compares pointers.
class WaveRouterManifestAdmission final {
public:
    [[nodiscard]] static std::shared_ptr<const WaveRouterManifestAdmission> create(const ModelManifest& manifest) {
        validate_wave_router_manifest(manifest);
        if (manifest.layout_identity == 0 || model_manifest_layout_identity(manifest) != manifest.layout_identity)
            throw std::invalid_argument("wave router manifest layout hash does not match its actual contents");
        return std::shared_ptr<const WaveRouterManifestAdmission>(new WaveRouterManifestAdmission(manifest));
    }
    [[nodiscard]] const ModelManifest& manifest() const noexcept { return manifest_; }
private:
    explicit WaveRouterManifestAdmission(const ModelManifest& manifest) : manifest_(manifest) {}
    const ModelManifest manifest_;
};

} // namespace strix

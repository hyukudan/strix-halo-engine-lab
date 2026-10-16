#pragma once
#include "strix/compiled_execution.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/offline_tuning_cache.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace strix {
struct LoadedOfflineTuningCache {
    std::vector<kernels::OfflineTuningRecord> records;
    std::string payload_sha256;
    std::uint64_t payload_bytes = 0;
};
// One bounded read: digest and parser consume the SAME immutable byte buffer.
[[nodiscard]] LoadedOfflineTuningCache load_offline_tuning_cache_file(
    const std::filesystem::path&, kernels::OfflineTuningCacheLimits limits = {});
struct OfflineTuningLiveContext {
    std::string gfx_target, device_sku, hip_identity;
    std::string binary_sha256, model_content_sha256;
    // Empty means unknown. Never fill this from a cache record or by hashing
    // a partial/ambiguous command line. Declared context is a separate receipt.
    std::string compiler_flags_sha256;
    std::uint32_t runtime_reported_processor_count = 0;
    std::uint64_t context_tokens = 0, state_layout_identity = 0;
    std::string memory_regime, resolved_configuration;
};
// Immutable owned preparation data. Registry selection remains the authority;
// this object cannot qualify a kernel or assign numerical equivalence classes.
class PreparedOfflineTuning final {
public:
    PreparedOfflineTuning(LoadedOfflineTuningCache cache, const OfflineTuningLiveContext&,
        const ModelContract&, const CompiledSerialExecution& baseline,
        const ModelManifest&, const ExecutionTensorAccessPlan&,
        const kernels::KernelRegistry&, const ExecutionBindingRequest&);
    PreparedOfflineTuning(const PreparedOfflineTuning&) = delete;
    PreparedOfflineTuning& operator=(const PreparedOfflineTuning&) = delete;
    PreparedOfflineTuning(PreparedOfflineTuning&&) = delete;
    PreparedOfflineTuning& operator=(PreparedOfflineTuning&&) = delete;
    // Borrowed spans: retain this nonmovable owner unchanged until every plan
    // consumer returns. This does no I/O and is never a token-loop lookup.
    [[nodiscard]] kernels::OfflineTuningPlan plan() const noexcept { return {keys_, records_}; }
    [[nodiscard]] const std::string& payload_sha256() const noexcept { return payload_sha256_; }
    [[nodiscard]] std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    [[nodiscard]] std::size_t record_count() const noexcept { return records_.size(); }
private:
    std::vector<kernels::OfflineTuningKey> keys_;
    std::vector<kernels::OfflineTuningRecord> records_;
    std::string payload_sha256_;
    std::uint64_t payload_bytes_ = 0;
};
} // namespace strix

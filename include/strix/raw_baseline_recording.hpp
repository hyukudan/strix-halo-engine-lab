#pragma once

#include "strix/serial_session.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strix {
enum class RawBaselineMode { teacher_forced, greedy };
enum class RawBaselineStatePolicy { final, each };
enum class RawBaselineEvidence { native_eager, injected_fixture };
enum class RawBaselineActivationStorage { bf16, fp32 };
struct RawBaselineMetadata {
    std::string model_sha256, model_content_identity, executable_sha256;
    std::string gpu_identity, gpu_architecture;
    std::uint64_t execution_identity = 0, kernel_plan_identity = 0, runtime_compatibility_identity = 0;
    std::uint32_t hip_runtime_version = 0, hip_driver_version = 0;
    std::uint64_t vocabulary_size = 0;
    std::vector<std::uint32_t> supplied_tokens;
    std::uint64_t maximum_generated_tokens = 0;
    std::optional<std::uint32_t> eos_token;
    RawBaselineMode mode = RawBaselineMode::teacher_forced;
    RawBaselineEvidence evidence = RawBaselineEvidence::injected_fixture;
    RawBaselineActivationStorage activation_storage = RawBaselineActivationStorage::bf16;
};
struct RawBaselinePolicy {
    std::uint64_t maximum_bytes = 8ULL * 1024 * 1024 * 1024;
    std::uint64_t maximum_steps = 0;
    std::uint32_t chunk_bytes = 65536;
    RawBaselineStatePolicy state_policy = RawBaselineStatePolicy::final;
};
// A reader must fill exactly the requested allocation-relative byte range or
// throw. On throw the bounded destination and captured reader owners are held
// to process lifetime. Readers must capture their source owners; reference-only
// native readers must fail-stop if synchronous quiescence cannot be proved.
// HIP uses checked D2H copies; fixture readers never become native proof.
using RawBaselineStateReader = std::function<void(const SessionStateAllocation&, std::uint64_t, std::span<std::byte>)>;
void validate_raw_baseline_output_directory(const std::filesystem::path&);
class RawBaselineRecording final {
public:
    RawBaselineRecording(std::filesystem::path, RawBaselineMetadata, RawBaselinePolicy, SerialSessionStateLayout);
    ~RawBaselineRecording();
    RawBaselineRecording(const RawBaselineRecording&) = delete;
    RawBaselineRecording& operator=(const RawBaselineRecording&) = delete;
    void capture_initial(const SerialSessionContext&, const RawBaselineStateReader&);
    void record_token(std::uint32_t input, std::uint64_t position, std::span<const float> logits,
                      const SerialSessionContext&, const RawBaselineStateReader&,
                      std::optional<std::uint32_t> teacher_forced_target = {});
    void finalize(const SerialSessionContext&, const RawBaselineStateReader&,
                  std::span<const std::uint32_t> generated_output_tokens = {}, std::string stop_reason = "raw_input_exhausted",
                  std::function<void()> verify_binding_before_publication = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix

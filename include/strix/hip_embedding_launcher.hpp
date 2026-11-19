#pragma once
#include <span>

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/graph_inputs.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipEmbeddingLauncherConfig {
    std::uint32_t hidden_size = 0;
    HipActivationStorage output =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
};

// Experimental C1 device-input gather. All allocations must remain owned and
// unchanged through stream completion. status is overwritten on every launch:
// 0=valid, 1=token out of range, 2=position out of range (flags may combine).
// This separate API does not promote the existing eager whole-model launcher.
void launch_hip_graph_embedding(
    const HipEmbeddingLauncherConfig& config, std::uint32_t ggml_type,
    const RuntimeTensorBinding& weights, std::uint32_t vocabulary_size,
    std::uint64_t context_tokens, const GraphDynamicArguments& arguments,
    const ExecutionBufferAddress& output, const ExecutionMemoryBinding& status,
    void* stream);

void register_hip_embedding_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipEmbeddingLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_embedding_launcher(
    std::string kernel_id,
    HipEmbeddingLauncherConfig* config);

class HipEmbeddingLauncherBundle {
public:
    explicit HipEmbeddingLauncherBundle(
        HipEmbeddingLauncherConfig config)
        : config_(config) {}

    HipEmbeddingLauncherBundle(
        const HipEmbeddingLauncherBundle&) = delete;
    HipEmbeddingLauncherBundle& operator=(
        const HipEmbeddingLauncherBundle&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipEmbeddingLauncherConfig config_;
};

struct MtpNativeContext;
void preflight_hip_mtp_embedding(const MtpNativeContext&,std::uint32_t token);
void launch_hip_mtp_embedding(const MtpNativeContext&,std::uint32_t token,void* stream);
// Same row load/store as above with the token read on device (sync-free draft
// chain); an out-of-vocabulary word reads row 0 and is rejected by its producer.
void launch_hip_mtp_embedding_device(const MtpNativeContext&,const std::uint32_t* token,void* stream);
// Multi-session draft batch: row t loads tokens[t] (a device token word, as launch_hip_mtp_embedding_device) or,
// when tokens[t] is null, the host token values[t] (as launch_hip_mtp_embedding); at most 8 contexts.
void launch_hip_mtp_embedding_rows(std::span<const MtpNativeContext> contexts,std::span<const std::uint32_t* const> tokens,
    std::span<const std::uint32_t> values,void* stream);

// Framework-free row gather over the same kernels (for an engine adapter that owns its pointers): one GGML table on
// the device (F32 / F16 / BF16 / Q8_0 / IQ4_XS / K / GSQ kinds, row_width values per row, vocabulary_size rows).
struct HipEmbeddingTable {
    const void* weights = nullptr;
    std::uint32_t ggml_type = 0, row_width = 0, vocabulary_size = 0;
};
// Output row t (pitch elements after row t-1) = the decoded table row of host id ids[t]; any count (launches of up
// to 256 rows). An id >= vocabulary_size throws before anything is enqueued. FP32 stores the decoded value, BF16
// rounds it to nearest even, exactly as launch_hip_graph_embedding stores it.
void launch_hip_embedding_rows(const HipEmbeddingTable& table, HipActivationStorage output,
    std::span<const std::uint32_t> ids, void* out, std::uint64_t pitch, void* stream);
// One row for a token word on the device (sync-free decode chain); an out-of-vocabulary word reads row 0 (the
// device-token rule of launch_hip_mtp_embedding_device: its producer rejects it).
void launch_hip_embedding_device_row(const HipEmbeddingTable& table, HipActivationStorage output,
    const std::uint32_t* device_id, void* out, void* stream);

// Tiled prompt priming: row t loads host token tokens[t] into outputs[t] (BF16 [hidden]) with
// launch_hip_mtp_embedding's row load/store; one owner, at most 8 rows.
void launch_hip_mtp_embedding_rows_to(const MtpNativeContext&,std::span<const std::uint32_t> tokens,
    std::span<std::uint16_t* const> outputs,void* stream);

}  // namespace strix

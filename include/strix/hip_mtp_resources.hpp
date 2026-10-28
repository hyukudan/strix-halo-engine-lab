#pragma once
#include "strix/compiled_mtp_draft_plan.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/mtp_draft_state_memory.hpp"
#include "strix/hip_mtp_input_fusion.hpp"
#include <array>
#include <memory>
#include <utility>

namespace strix {
class HipSetupQuiescenceGuard;
struct MtpSharedTargetBinding;
struct MtpDeviceView {
    std::uintptr_t address=0;
    std::uint64_t bytes=0,allocation_generation=0;
    bool operator==(const MtpDeviceView&) const = default;
};
struct MtpTensorView {
    const ManifestTensor* manifest=nullptr;
    const RuntimeTensorBinding* runtime=nullptr;
};
// Exact source tensor views, owned and revalidated by HipMtpResources. These
// are native sidecar roles; no ordinary serial layer/role is manufactured.
struct MtpNativeTensorViews {
    MtpTensorView embedding,enorm,hnorm,eh_projection;
    MtpTensorView attention_hc_norm,attention_hc_down,attention_hc_up,attention_hc_inject;
    MtpTensorView query,key,value,attention_output,query_norm,key_norm;
    MtpTensorView index_query,index_key,index_query_norm,index_key_norm;
    MtpTensorView mlp_hc_norm,mlp_hc_down,mlp_hc_up,mlp_hc_inject;
    MtpTensorView router,expert_gate,expert_up,expert_down;
    MtpTensorView shared_gate,shared_up,shared_down,shared_scalar_gate;
    MtpTensorView head_norm,head_down,head_up,lm_head;
};
// The three checkpoint enumerators retain their old numeric slots for source
// compatibility only. They have no allocation; view() rejects retired roles.
enum class HipMtpBuffer : std::uint32_t {
    wide_current,wide_next,wide_checkpoint,embedding,normalized_input,projected_input,
    fused_wide,wide_work,hidden0,hidden1,hidden2,injection,routes,route_weights,
    hyper_workspace,attention_workspace,moe_workspace,router_workspace,logits,
    kv,raw_index,kv_checkpoint,raw_index_checkpoint,status,count
};
struct HipMtpBudget {
    // Pool alignment reservation plus the finite reader's requested payload;
    // excludes separately retained weights and runtime/driver bookkeeping.
    std::uint64_t retained_bytes=0,context_capacity=0;
    std::uint32_t maximum_candidate_depth=0;
    // Explicit native numerical profile, never inferred from weight storage.
    HipMtpInputRounding normalized_rounding=HipMtpInputRounding::unresolved;
    HipMtpInputRounding projection_rounding=HipMtpInputRounding::unresolved;
};
struct HipMtpReservation {
    std::uint64_t device_pool_bytes=0,selector_payload_bytes=0,retained_payload_budget=0;
};
class HipMtpResources;
struct MtpNativeContext {
    const ModelContract* model=nullptr;
    const CompiledMtpDraftPlan* plan=nullptr;
    const MtpNativeTensorViews* tensors=nullptr;
    const HipMtpResources* resources=nullptr;
    std::uint64_t position=0;
};
// Independent proposal allocations; no SerialSessionContext or target state
// identity is synthesized. One caller owns progress and keeps metadata stable.
// Factory requires the verified explicit sidecar/shared-target binding and
// real residency owners. Native launchers consume only this owned view.
class HipMtpResources final : public std::enable_shared_from_this<HipMtpResources> {
public:
    [[nodiscard]] static std::shared_ptr<HipMtpResources> create(
        const ModelContract&,const CompiledMtpDraftPlan&,
        const ModelManifest& sidecar_manifest,std::shared_ptr<const HipModelResidency> sidecar,
        const ModelManifest& target_manifest,std::shared_ptr<const HipModelResidency> target,
        const MtpSharedTargetBinding&,HipMtpBudget,std::shared_ptr<HipRuntimeAllocator>,const RotaryIdentity& target_rotary);
    ~HipMtpResources();
    HipMtpResources(const HipMtpResources&)=delete;
    HipMtpResources& operator=(const HipMtpResources&)=delete;
    [[nodiscard]] MtpDeviceView view(HipMtpBuffer) const;
    [[nodiscard]] const ModelContract& model() const noexcept;
    [[nodiscard]] const CompiledMtpDraftPlan& plan() const noexcept;
    [[nodiscard]] const MtpDraftRuntimeBinding& binding() const noexcept;
    [[nodiscard]] const ModelManifest& manifest() const noexcept;
    [[nodiscard]] const ModelRuntimeBinding& runtime_binding() const noexcept;
    [[nodiscard]] const MtpNativeTensorViews& tensors() const noexcept;
    [[nodiscard]] const IndexedAttentionWorkspaceLayout& attention_workspace() const noexcept;
    [[nodiscard]] std::uint64_t context_capacity() const noexcept;
    [[nodiscard]] std::uint64_t physical_identity() const noexcept;
    [[nodiscard]] std::uint64_t retained_payload_bytes() const noexcept;
    [[nodiscard]] const HipMtpReservation& reservation() const noexcept;
    [[nodiscard]] const HipMtpBudget& budget() const noexcept;
    [[nodiscard]] const MtpSharedTargetBinding& shared_target() const noexcept;
    [[nodiscard]] const MtpDraftStateMemoryBinding& state_memory() const noexcept;
    [[nodiscard]] std::shared_ptr<const HipSetupQuiescenceGuard> guard() const noexcept;
    [[nodiscard]] void* stream() const noexcept;
    // Full source census/binding admission at each native entry boundary.
    void validate() const;
    class NativeValidationScope {
    public:
        ~NativeValidationScope();
        NativeValidationScope(NativeValidationScope&&) noexcept;
        NativeValidationScope(const NativeValidationScope&)=delete;
        NativeValidationScope& operator=(const NativeValidationScope&)=delete;
        NativeValidationScope& operator=(NativeValidationScope&&)=delete;
    private:
        friend class HipMtpResources;
        explicit NativeValidationScope(std::shared_ptr<const HipMtpResources> owner):owner_(std::move(owner)) {}
        std::shared_ptr<const HipMtpResources> owner_;
    };
    // Serial, non-nesting capability; successful full admission is required to
    // construct it. Keep this scope through all stages and actual completion.
    [[nodiscard]] NativeValidationScope admit_native_entry() const;
    // Cheap immutable-source/lease/current-device checks inside an admitted
    // entry. Standalone raw calls retain full validation when no scope exists.
    void validate_stage() const;
    // Native queue lifecycle only: not a semantic qualification or target
    // transaction. Failure remains sticky and cleanup proves quiescence.
    void require_idle() const;
    void mark_submitted();
    void complete();
    void fail() noexcept;
    [[nodiscard]] bool failed() const noexcept;
private:
    HipMtpResources();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix

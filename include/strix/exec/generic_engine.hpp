#pragma once

#include "strix/exec/compiler.hpp"
#include "strix/graph/model_decl.hpp"
#include "strix/session/engine.hpp"
#include "strix/state/txn.hpp"
#include "strix/weight/load.hpp"

namespace strix::exec {

class GenericEngine final : public session::Engine {
public:
    // Resolved host bindings for the primary text graph. The CPU window fits
    // in one X3 descriptor per lane. Borrowed declaration and recipe fields
    // obey their upstream lifetime contracts.
    struct CpuBinding {
        std::uint32_t context = 0;
        std::uint32_t slots = 0;
        weight::LoadPlan load;
        std::unique_ptr<weight::ImageSink> images;
        std::vector<state::Decl> states;
        std::vector<state::Update> updates;
        PlanProvider plan;
        BufferResolver buffers;
        // Construct the CPU catalogue binding against stable loaded images
        // and the read-only published S1 table.
        std::function<KernelResolver(const weight::ImageSink&, const state::CpuState&)> kernels;
        // Project this launch's arena outputs onto S1 state instances. Each
        // returned view is consumed by Txn::append before the next call.
        std::function<void(const Program&, std::span<const std::byte>, std::uint32_t,
                           std::span<state::Row>)> state_rows;
    };

    [[nodiscard]] static std::unique_ptr<GenericEngine>
    open_cpu(graph::ModelDecl, CpuBinding, const profile::Resolution&);
    ~GenericEngine() override;

    // The store is owned by the engine and borrowed by the serialized serve owner.
    [[nodiscard]] session::KvStore& store() noexcept;
    [[nodiscard]] session::Capabilities caps() const override;
    [[nodiscard]] profile::ExecutionReceipt receipt() const override;

    void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
    [[nodiscard]] session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
    [[nodiscard]] std::vector<session::AdvanceResult> advance(std::span<const session::AdvanceItem>) override;
    void prefill_batch(std::span<const session::PrefillItem>) override;
    [[nodiscard]] std::vector<session::StepResult> decode_batch(std::span<const session::DecodeItem>) override;
    void cancel_lane(session::KvHandle&) override;
    void resume_lane(session::KvHandle&) override;
    void append_external(std::span<const std::uint32_t>, session::KvHandle&) override;
    void commit_pending(session::KvHandle&) override;

private:
    struct Impl;
    explicit GenericEngine(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::exec

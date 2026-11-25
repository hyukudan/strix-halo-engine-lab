#pragma once

#include "strix/serve/session_runtime.hpp"

namespace strix::glm::serve {

// Request ownership only. Physical slots, grouped native work and rank control
// belong to Lead; logical handles never address native state.
class GlmSessionOwner {
public:
    GlmSessionOwner(strix::serve::GenerationQueue&, std::uint64_t context, std::uint32_t slots,
                    std::function<void(strix::serve::SessionRuntime&)> turn,
                    std::function<bool()> drained, std::function<bool()> stop,
                    std::function<void()> closing, std::function<void()> stopped,
                    std::function<void()> failed, bool embed_rows = false);
    strix::serve::SessionRuntime& runtime() noexcept { return runtime_; }
private:
    class ControlEngine final : public session::Engine {
    public:
        // embed_rows: the native engine takes provided embedding rows, so jobs with virtual ids are admitted.
        ControlEngine(std::uint64_t, std::uint32_t, bool embed_rows);
        void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
        session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
        session::Capabilities caps() const override { return caps_; }
        // The native turns read GenerationJob::embeds themselves (the lead binds each prompt's table to its engine);
        // the lane only records it.
        void bind_embeds(session::KvHandle&, std::shared_ptr<const session::EmbedTable> table) override {
            if (table && !caps_.embed_rows) throw std::invalid_argument("engine does not support embedding rows");
        }
        profile::ExecutionReceipt receipt() const override { return {}; }
    private:
        session::Capabilities caps_;
    } engine_;
    session::KvStore leases_;
    strix::serve::SessionRuntime runtime_;
};

} // namespace strix::glm::serve

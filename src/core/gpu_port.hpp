#pragma once

#include "strix/core/dense_reference.hpp"
#include "strix/session/backend.hpp"

#include <optional>

namespace strix::core::gpu_port {

// Observations describe one device. An absent property is unknown.
struct DeviceObservation {
    std::string name;
    bool compute = false;
    std::size_t subgroup_width = 0;
    std::optional<bool> cooperative_matrix;
};

struct Capabilities {
    std::size_t ordinal = 0;
    std::string name;
    std::optional<std::uint32_t> subgroup_width;
    std::optional<bool> cooperative_matrix;
};

[[noreturn]] void refuse(std::string_view backend, std::string_view reason);
std::optional<std::uint32_t> subgroup_width(std::size_t reported) noexcept;
Capabilities select_compute_device(std::string_view backend, std::span<const DeviceObservation> devices);

// Stable scalar layout for the two native RMSNorm bodies.
struct RmsNormParameters {
    std::uint32_t width;
    std::uint32_t rows;
    std::uint32_t use_scale;
    float epsilon;
};
static_assert(sizeof(RmsNormParameters) == 16);

RmsNormParameters rms_norm_parameters(std::size_t input, std::size_t scale, std::size_t output,
                                     std::uint32_t width, float epsilon);

class ReferenceBackend : public session::Backend, public DenseReference {
public:
    void gather(Q8Rows, std::uint32_t, std::span<float>) override;
    void linear(Q8Rows, std::span<const float>, std::span<float>) override;
    void rms_norm(std::span<const float>, std::span<const float>, std::span<float>, std::uint32_t, float) final;
    void rope(std::span<float>, std::span<const float>, std::uint32_t) override;
    void attention(const DenseAttentionRow&, std::span<float>) override;

protected:
    struct MappedBuffer : session::detail::BufferState { std::byte* mapped = nullptr; };
    using BoundBuffer = session::detail::BoundBuffer;

    void initialize_queue() { queue_ = create_queue(); }
    virtual void check_rms_norm(const RmsNormParameters&, std::size_t bytes) const = 0;
    virtual void launch_rms_norm(const RmsNormParameters&, const BoundBuffer& input,
                                 const BoundBuffer& scale, const BoundBuffer& output) = 0;

private:
    session::Queue queue_;
    std::shared_ptr<session::detail::QueueState> make_queue() final;
    std::shared_ptr<session::detail::EventState> record_event(session::detail::QueueState&) final;
    void wait_event(session::detail::QueueState&, session::detail::EventState&) final;
    void finish(session::detail::QueueState&) final;
    void transfer(session::detail::QueueState&, const BoundBuffer*, const BoundBuffer*,
                  void*, const void*, std::size_t) final;
    void execute_plan(session::ExecutionPlan&) final;
};

} // namespace strix::core::gpu_port

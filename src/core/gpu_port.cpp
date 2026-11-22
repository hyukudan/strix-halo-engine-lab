#include "gpu_port.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace strix::core::gpu_port {

void refuse(std::string_view backend, std::string_view reason) {
    throw session::UnsupportedOperation(std::string(backend) + ": " + std::string(reason));
}

std::optional<std::uint32_t> subgroup_width(std::size_t reported) noexcept {
    if (std::has_single_bit(reported) && reported <= std::numeric_limits<std::uint32_t>::max())
        return static_cast<std::uint32_t>(reported);
    return std::nullopt;
}

Capabilities select_compute_device(std::string_view backend, std::span<const DeviceObservation> devices) {
    for (std::size_t ordinal = 0; ordinal < devices.size(); ++ordinal) {
        const auto& device = devices[ordinal];
        if (!device.compute) continue;
        return {ordinal, device.name, subgroup_width(device.subgroup_width), device.cooperative_matrix};
    }
    refuse(backend, devices.empty() ? "no device found" : "no eligible compute device found");
}

RmsNormParameters rms_norm_parameters(std::size_t input, std::size_t scale, std::size_t output,
                                     std::uint32_t width, float epsilon) {
    if (!width || !input || input % width || output != input ||
        (scale && scale != width) || !(epsilon > 0.0F) || !std::isfinite(epsilon))
        throw std::invalid_argument("invalid reference RMSNorm shape or epsilon");
    if (input > std::numeric_limits<std::uint32_t>::max() ||
        input > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("reference RMSNorm element count");
    return {width, static_cast<std::uint32_t>(input / width), scale ? 1U : 0U, epsilon};
}

void ReferenceBackend::rms_norm(std::span<const float> input, std::span<const float> scale,
                               std::span<float> output, std::uint32_t width, float epsilon) {
    const auto parameters = rms_norm_parameters(input.size(), scale.size(), output.size(), width, epsilon);
    check_rms_norm(parameters, input.size_bytes());
    auto& queue = checked(queue_);
    const auto source = allocate(input.size_bytes());
    const auto result = allocate(output.size_bytes());
    const session::BufferSlice source_range{source, 0, input.size_bytes()};
    const session::BufferSlice result_range{result, 0, output.size_bytes()};
    session::Buffer multiplier;
    auto scale_range = source_range;
    upload(queue_, source_range, std::as_bytes(input));
    if (!scale.empty()) {
        multiplier = allocate(scale.size_bytes());
        scale_range = {multiplier, 0, scale.size_bytes()};
        upload(queue_, scale_range, std::as_bytes(scale));
    }
    const auto bound_input = checked(source_range);
    const auto bound_scale = checked(scale_range);
    const auto bound_output = checked(result_range);
    run(queue, [&] { launch_rms_norm(parameters, bound_input, bound_scale, bound_output); });
    download(queue_, result_range, std::as_writable_bytes(output));
}

void ReferenceBackend::gather(Q8Rows, std::uint32_t, std::span<float>) {
    refuse(backend_name(), "reference gather unavailable");
}
void ReferenceBackend::linear(Q8Rows, std::span<const float>, std::span<float>) {
    refuse(backend_name(), "reference linear unavailable");
}
void ReferenceBackend::rope(std::span<float>, std::span<const float>, std::uint32_t) {
    refuse(backend_name(), "reference RoPE unavailable");
}
void ReferenceBackend::attention(const DenseAttentionRow&, std::span<float>) {
    refuse(backend_name(), "reference attention unavailable");
}
void ReferenceBackend::execute_plan(session::ExecutionPlan&) {
    refuse(backend_name(), "family execution plan unavailable");
}
std::shared_ptr<session::detail::QueueState> ReferenceBackend::make_queue() {
    return std::make_shared<session::detail::QueueState>();
}
std::shared_ptr<session::detail::EventState> ReferenceBackend::record_event(session::detail::QueueState&) {
    return std::make_shared<session::detail::EventState>();
}
void ReferenceBackend::wait_event(session::detail::QueueState&, session::detail::EventState&) {}
void ReferenceBackend::finish(session::detail::QueueState&) {}
void ReferenceBackend::transfer(session::detail::QueueState&, const BoundBuffer* destination,
                                const BoundBuffer* source, void* host_destination,
                                const void* host_source, std::size_t bytes) {
    if (!bytes) return;
    const auto address = [](const BoundBuffer& region) {
        return static_cast<const MappedBuffer&>(*region.state).mapped + region.offset;
    };
    std::memcpy(destination ? address(*destination) : host_destination,
                source ? address(*source) : host_source, bytes);
}

} // namespace strix::core::gpu_port

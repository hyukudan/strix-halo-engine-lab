#include "metal/core/backend.hpp"
#include "vulkan/core/backend.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

namespace port = strix::core::gpu_port;
namespace session = strix::session;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Exception = session::UnsupportedOperation, class Action>
void expect_failure(Action action, std::string_view reason) {
    try {
        action();
    } catch (const Exception& error) {
        check(std::string_view(error.what()).find(reason) != std::string_view::npos, "wrong refusal");
        return;
    }
    throw std::runtime_error("expected refusal");
}

void capability_data() {
    const std::array<port::DeviceObservation, 3> devices{{
        {"no compute", false, 32, true},
        {"compute A", true, 16, false},
        {"compute B", true, 64, true},
    }};
    const auto caps_a = port::select_compute_device("Vulkan", devices);
    const auto caps_b = port::select_compute_device("Vulkan", std::span(devices).last(1));
    check(caps_a.ordinal == 1 && caps_a.name == "compute A", "first eligible ordinal");
    check(caps_a.subgroup_width == 16 && caps_a.cooperative_matrix == false, "device A observations");
    check(caps_b.subgroup_width == 64 && caps_b.cooperative_matrix == true, "device B observations");
    const std::array<port::DeviceObservation, 1> unknown{{{"unknown", true, 3, {}}}};
    const auto caps_unknown = port::select_compute_device("Metal", unknown);
    check(!caps_unknown.subgroup_width && !caps_unknown.cooperative_matrix, "unknown capability is preserved");
}

void no_device() {
    port::Capabilities caps{42, "sentinel", 16, true};
    expect_failure([&] { (void)strix::vulkan::open_reference(&caps); }, "Vulkan: native backend/loader unavailable");
    expect_failure([&] { (void)strix::metal::open_reference(&caps); }, "Metal: native backend unavailable");
    check(caps.ordinal == 42 && caps.name == "sentinel" && caps.subgroup_width == 16 &&
              caps.cooperative_matrix == true, "failure changed caller capabilities");
    expect_failure([] { (void)port::select_compute_device("Vulkan", {}); }, "Vulkan: no device found");
    expect_failure([] { (void)port::select_compute_device("Metal", {}); }, "Metal: no device found");
    const std::array<port::DeviceObservation, 1> unusable{{{"no compute", false, 32, true}}};
    expect_failure([&] { (void)port::select_compute_device("Vulkan", unusable); }, "no eligible compute device found");
}

class TestBackend final : public port::ReferenceBackend {
    struct Storage final : MappedBuffer {
        alignas(64) std::array<std::byte, 64> data{};
        Storage() { mapped = data.data(); }
    };
public:
    TestBackend() { initialize_queue(); }
    std::size_t launches = 0;
    bool fail_dispatch = false;

private:
    std::shared_ptr<session::detail::BufferState> allocate_buffer(std::size_t bytes, std::size_t alignment) override {
        check(bytes <= 64 && alignment <= 64, "test storage limits");
        return std::make_shared<Storage>();
    }
    void check_rms_norm(const port::RmsNormParameters&, std::size_t bytes) const override {
        check(bytes == 4 * sizeof(float), "native byte limit check");
    }
    void launch_rms_norm(const port::RmsNormParameters& params, const BoundBuffer& input,
                         const BoundBuffer& scale, const BoundBuffer& output) override {
        ++launches;
        if (fail_dispatch) throw std::runtime_error("test dispatch failed");
        check(params.width == 2 && params.rows == 2 && params.epsilon == 0.25F, "shader parameters");
        check(input.offset == 0 && scale.offset == 0 && output.offset == 0, "staged base offsets");
        check((input.state != scale.state) == (params.use_scale == 1), "optional scale binding");
        const auto* source = static_cast<const Storage&>(*input.state).mapped;
        const auto* multiplier = static_cast<const Storage&>(*scale.state).mapped;
        auto* destination = static_cast<const Storage&>(*output.state).mapped;
        if (params.use_scale) {
            const std::array expected{3.0F, 4.0F};
            check(scale.bytes == sizeof(expected) &&
                      std::memcmp(expected.data(), multiplier, sizeof(expected)) == 0, "scale upload");
        }
        // Echo the input to exercise staging without a second RMSNorm implementation.
        std::memcpy(destination, source, input.bytes);
    }
};

void reference_resources() {
    TestBackend backend;
    TestBackend other;
    auto queue = backend.create_queue();
    const auto source = backend.allocate(4);
    const auto target = backend.allocate(4);
    const auto foreign = other.allocate(4);
    const std::array input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> output{};
    backend.upload(queue, {source, 0, 4}, input);
    backend.copy(queue, {target, 0, 4}, {source, 0, 4});
    const auto event = backend.record(queue);
    backend.wait(queue, event);
    backend.download(queue, {target, 0, 4}, output);
    check(input == output, "synchronous buffer transfers");
    expect_failure<std::invalid_argument>([&] { backend.copy(queue, {target, 0, 4}, {foreign, 0, 4}); }, "foreign buffer");
    expect_failure<std::invalid_argument>([&] { backend.copy(queue, {source, 1, 3}, {source, 0, 3}); }, "overlapping");
    expect_failure<std::out_of_range>([&] { backend.download(queue, {target, 1, 4}, output); }, "buffer byte range");

    const std::array values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array scale{3.0F, 4.0F};
    auto result = values;
    backend.rms_norm(result, scale, result, 2, 0.25F);
    check(result == values, "aliased host staging");
    backend.rms_norm(values, {}, result, 2, 0.25F);
    check(result == values && backend.launches == 2, "unscaled staging");
    expect_failure<std::invalid_argument>([&] { backend.rms_norm(values, scale, result, 3, 0.25F); }, "invalid reference RMSNorm");
    expect_failure<std::invalid_argument>([&] {
        backend.rms_norm(values, {}, result, 2, std::numeric_limits<float>::infinity());
    }, "invalid reference RMSNorm");
    expect_failure([&] { backend.linear({}, values, result); }, "reference linear unavailable");
    check(backend.launches == 2, "invalid shape reached the native hook");

    backend.fail_dispatch = true;
    result.fill(0.0F);
    expect_failure<std::runtime_error>([&] { backend.rms_norm(values, {}, result, 2, 0.25F); }, "test dispatch failed");
    check(result == std::array<float, 4>{}, "failed dispatch published output");
    expect_failure<std::runtime_error>([&] { backend.rms_norm(values, {}, result, 2, 0.25F); }, "queue is poisoned");
    check(backend.launches == 3, "poisoned queue dispatched");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected test case");
        const std::string_view test = argv[1];
        if (test == "capabilities") capability_data();
        else if (test == "refusal") no_device();
        else if (test == "resources") reference_resources();
        else throw std::invalid_argument("unknown test case");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

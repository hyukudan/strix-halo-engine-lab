#include "strix/hip_dense_trellis.hpp"
#include "strix/hip_gdn_launcher.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace strix;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Buffer {
    void* data = nullptr; std::uint64_t bytes;
    explicit Buffer(std::uint64_t n) : bytes(n) { check(hipMalloc(&data, n)); }
    ~Buffer() { if (data) static_cast<void>(hipFree(data)); }
    Buffer(const Buffer&) = delete;
    void upload(const void* p) { check(hipMemcpy(data, p, bytes, hipMemcpyHostToDevice)); }
};
struct Stream {
    hipStream_t value{};
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { static_cast<void>(hipStreamDestroy(value)); }
};
float latency(const std::function<void()>& launch, hipStream_t stream, Buffer& flush, bool cold) {
    hipEvent_t begin{}, end{}; check(hipEventCreate(&begin)); check(hipEventCreate(&end));
    std::vector<float> samples;
    for (unsigned repeat = 0; repeat < 20U; ++repeat) {
        if (cold) check(hipMemsetAsync(flush.data, static_cast<int>(repeat), flush.bytes, stream));
        check(hipEventRecord(begin, stream)); launch(); check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
        float ms{}; check(hipEventElapsedTime(&ms, begin, end));
        if (repeat >= 4U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(begin)); check(hipEventDestroy(end));
    std::sort(samples.begin(), samples.end()); return samples[samples.size() / 2U];
}
void run(const char* path, bool benchmark) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file.good() && file.tellg() >= 64 && file.tellg() <= (256LL << 20U), "fixture file extent differs");
    std::vector<std::byte> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(file.good(), "fixture read failed");
    std::size_t cursor = 0;
    const auto take = [&](std::uint64_t n) {
        require(n <= bytes.size() - cursor, "fixture section exceeds file");
        const auto* p = bytes.data() + cursor; cursor += static_cast<std::size_t>(n); return p;
    };
    require(std::memcmp(take(8U), "TRLDNS1\0", 8U) == 0, "fixture magic differs");
    std::array<std::uint32_t, 4> shape{}; std::memcpy(shape.data(), take(16U), 16U);
    const auto [inputs, outputs, bits, rows] = shape;
    std::array<std::uint64_t, 3> sizes{}; std::memcpy(sizes.data(), take(24U), 24U);
    const auto [packed_bytes, q8_bytes, actual_f16] = sizes;
    const auto* reserved = take(16U);
    require(std::all_of(reserved, reserved + 16U, [](std::byte b) { return b == std::byte{}; }), "fixture reserved fields differ");
    require(inputs && inputs <= 16384U && inputs % 128U == 0U && outputs && outputs <= 65536U && outputs % 128U == 0U &&
        (bits == 5U || bits == 6U) && rows == 8U && actual_f16 <= 1U &&
        packed_bytes == static_cast<std::uint64_t>(inputs) * outputs * bits / 8U &&
        q8_bytes == static_cast<std::uint64_t>(inputs) * outputs / 32U * 34U, "fixture geometry differs");
    const auto count = static_cast<std::uint64_t>(inputs) * outputs;
    Buffer packed(packed_bytes), su(inputs * 2ULL), sv(outputs * 2ULL), input(rows * inputs * 2ULL),
        decoded(count * 2U), output(rows * outputs * 4ULL), q8(q8_bytes), control(rows * outputs * 4ULL),
        transformed(rows * inputs * 4ULL), partial(4ULL * rows * outputs * 4ULL);
    packed.upload(take(packed_bytes)); su.upload(take(su.bytes)); sv.upload(take(sv.bytes)); input.upload(take(input.bytes));
    const auto* expected_decode = take(decoded.bytes);
    std::vector<float> expected(static_cast<std::size_t>(rows) * outputs);
    std::memcpy(expected.data(), take(output.bytes), static_cast<std::size_t>(output.bytes));
    q8.upload(take(q8_bytes)); require(cursor == bytes.size(), "fixture has unrecognized trailing data");
    HipDenseTrellisWeights weights{static_cast<const std::uint32_t*>(packed.data), static_cast<const std::uint16_t*>(su.data),
        static_cast<const std::uint16_t*>(sv.data), inputs, outputs, bits};
    HipDenseTrellisWorkspace workspace{static_cast<float*>(transformed.data), static_cast<float*>(partial.data), transformed.bytes, partial.bytes};
    Stream stream;
    launch_hip_dense_trellis_decode(weights, static_cast<std::uint16_t*>(decoded.data), decoded.bytes, stream.value);
    check(hipStreamSynchronize(stream.value));
    std::vector<std::byte> actual_decode(static_cast<std::size_t>(decoded.bytes));
    check(hipMemcpy(actual_decode.data(), decoded.data, decoded.bytes, hipMemcpyDeviceToHost));
    require(std::memcmp(actual_decode.data(), expected_decode, actual_decode.size()) == 0, "rotated F16 decode differs from independent bit-array oracle");
    Buffer flush(256ULL << 20U);
    std::vector<std::uint16_t> original_input(static_cast<std::size_t>(rows) * inputs);
    check(hipMemcpy(original_input.data(), input.data, input.bytes, hipMemcpyDeviceToHost));
    std::vector<float> previous_rows;
    for (const unsigned active : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
        const auto candidate = [&] { launch_hip_dense_trellis(weights, static_cast<const std::uint16_t*>(input.data), inputs,
            static_cast<float*>(output.data), outputs, active, workspace, stream.value); };
        candidate(); check(hipStreamSynchronize(stream.value));
        std::vector<float> actual(static_cast<std::size_t>(active) * outputs);
        check(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> actual_partial(static_cast<std::size_t>(4U) * active * outputs);
        check(hipMemcpy(actual_partial.data(), partial.data, actual_partial.size() * sizeof(float), hipMemcpyDeviceToHost));
        require(previous_rows.empty() || std::memcmp(actual.data(), previous_rows.data(), previous_rows.size() * sizeof(float)) == 0,
            "compact projection changed existing rows when the physical row count changed");
        previous_rows = actual;
        launch_hip_dense_trellis_scalar_reference(weights, static_cast<const std::uint16_t*>(input.data), inputs,
            static_cast<float*>(control.data), outputs, active, workspace, stream.value);
        check(hipStreamSynchronize(stream.value));
        std::vector<float> scalar(actual.size());
        check(hipMemcpy(scalar.data(), control.data, scalar.size() * sizeof(float), hipMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), scalar.data(), actual.size() * sizeof(float)) == 0,
            "optimized compact projection changed scalar-codec reference bits");
        std::vector<float> scalar_partial(actual_partial.size());
        check(hipMemcpy(scalar_partial.data(), partial.data, scalar_partial.size() * sizeof(float), hipMemcpyDeviceToHost));
        require(std::memcmp(actual_partial.data(), scalar_partial.data(), actual_partial.size() * sizeof(float)) == 0,
            "optimized compact projection changed a split partial before output transform");
        hipGraph_t graph{}; hipGraphExec_t executable{};
        check(hipStreamBeginCapture(stream.value, hipStreamCaptureModeThreadLocal));
        candidate();
        check(hipStreamEndCapture(stream.value, &graph));
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        auto changed_input = original_input;
        std::vector<float> replay(actual.size());
        for (unsigned repeat = 0; repeat < 16U; ++repeat) {
            for (std::size_t i = 0; i < changed_input.size(); ++i)
                changed_input[i] = static_cast<std::uint16_t>(original_input[i] ^ ((repeat & 1U) ? 0x8000U : 0U));
            check(hipMemcpyAsync(input.data, changed_input.data(), input.bytes, hipMemcpyHostToDevice, stream.value));
            check(hipMemsetAsync(partial.data, 0xff, partial.bytes, stream.value));
            check(hipGraphLaunch(executable, stream.value));
            check(hipStreamSynchronize(stream.value));
            check(hipMemcpy(replay.data(), output.data, replay.size() * sizeof(float), hipMemcpyDeviceToHost));
            launch_hip_dense_trellis_scalar_reference(weights, static_cast<const std::uint16_t*>(input.data), inputs,
                static_cast<float*>(control.data), outputs, active, workspace, stream.value);
            check(hipStreamSynchronize(stream.value));
            check(hipMemcpy(scalar.data(), control.data, scalar.size() * sizeof(float), hipMemcpyDeviceToHost));
            require(std::memcmp(replay.data(), scalar.data(), replay.size() * sizeof(float)) == 0,
                "captured compact projection changed bits or reused poisoned scratch");
        }
        input.upload(original_input.data());
        check(hipGraphExecDestroy(executable)); check(hipGraphDestroy(graph));
        double square = 0, denominator = 0, maximum = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite fixture result");
            const double error = static_cast<double>(actual[i]) - expected[i];
            square += error * error; denominator += static_cast<double>(expected[i]) * expected[i]; maximum = std::max(maximum, std::abs(error));
        }
        const double relative = std::sqrt(square / std::max(denominator, 1e-30));
        std::cout << "{\"rows\":" << active << ",\"rotated_decode_bits_equal\":true,\"scalar_reference_bits_equal\":true,\"row_count_invariant\":true,\"oracle_materialized_f16\":" << (actual_f16 ? "true" : "false")
            << ",\"split_partial_bits_equal\":true,\"graph_changed_input_replays\":16,\"relative_l2\":" << relative << ",\"maximum_abs_error\":" << maximum << ",\"model_quality_qualified\":false}\n";
        require(relative <= 0.001, "layer fixture exceeds preregistered relative L2 diagnostic envelope");
        HipDenseTrellisRows views{};
        for (unsigned r = 0; r < active; ++r) {
            views.input[r] = static_cast<const std::uint16_t*>(input.data) + static_cast<std::uint64_t>(r) * inputs;
            views.output[r] = static_cast<float*>(control.data) + static_cast<std::uint64_t>(r) * outputs;
        }
        launch_hip_dense_trellis_rows(weights, views, active, workspace, stream.value);
        check(hipStreamSynchronize(stream.value));
        std::vector<float> row_results(actual.size());
        check(hipMemcpy(row_results.data(), control.data, row_results.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto b = std::bit_cast<std::uint32_t>(actual[i]);
            require(std::bit_cast<std::uint32_t>(row_results[i]) == ((b + 0x7fffU + ((b >> 16U) & 1U)) & 0xffff0000U),
                "native row views differ from BF16-rounded contiguous operator");
        }
        if (!benchmark || (active != 1U && active != 2U && active != 4U && active != 8U)) continue;
        std::vector<GdnKnownWindowProjectionView> x, y;
        for (unsigned r = 0; r < active; ++r) {
            x.push_back({reinterpret_cast<std::uintptr_t>(input.data) + r * inputs * 2ULL, inputs * 2ULL});
            y.push_back({reinterpret_cast<std::uintptr_t>(control.data) + r * outputs * 4ULL, outputs * 4ULL});
        }
        const auto reference = [&] { gdn_known_window_rows_projection_probe(GdnKnownWindowProjectionDirection::bf16_to_fp32, x,
            {reinterpret_cast<std::uintptr_t>(q8.data), q8.bytes}, y, inputs, outputs, stream.value, GdnKnownWindowRowsProbeMode::rows_a8); };
        for (const bool cold : {false, true}) for (unsigned pair = 0; pair < 3U; ++pair) {
            float a{}, b{};
            if (pair & 1U) { b = latency(candidate, stream.value, flush, cold); a = latency(reference, stream.value, flush, cold); }
            else { a = latency(reference, stream.value, flush, cold); b = latency(candidate, stream.value, flush, cold); }
            std::cout << "{\"rows\":" << active << ",\"cold\":" << (cold ? "true" : "false") << ",\"pair\":" << pair
                << ",\"q8_a8_control_us\":" << a << ",\"compact_raw_operator_us\":" << b << ",\"performance_qualified\":false}\n";
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--bench"), "usage: fixture.trldns [--bench]");
        run(argv[1], argc == 3); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

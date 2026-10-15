#include "strix/hip_repacked_hyper_projection_tests.hpp"
#include "strix/hc_repacked_down_pack.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace repacked_test {
using strix::hc_projection_diagnostic::Shape;
using strix::hc_projection_diagnostic::Span;
using strix::hc_repacked_projection_diagnostic::Schedule;
constexpr std::size_t guard_bytes = 256;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
std::vector<Schedule> parse_schedules(const std::string& text) {
    require(!text.empty(), "explicit admitted schedules required");
    std::vector<Schedule> result;
    unsigned previous = 0;
    for (std::size_t at = 0; at < text.size(); at += 2) {
        require(text[at] >= '1' && text[at] <= '3', "invalid admitted schedule");
        const unsigned value = static_cast<unsigned>(text[at] - '0');
        require(value > previous, "admitted schedules must be distinct ascending 1,2,3");
        result.push_back(static_cast<Schedule>(value)); previous = value;
        if (at + 1 < text.size()) require(text[at + 1] == ',' && at + 2 < text.size(), "invalid schedule separator");
    }
    return result;
}
std::size_t buffer_bytes(std::size_t payload) {
    require(payload <= std::numeric_limits<std::size_t>::max() - 2 * guard_bytes, "fixture allocation overflow");
    return payload ? payload + 2 * guard_bytes : 0;
}
struct Buffer {
    strix::HipSetupQuiescenceGuard guard;
    void* allocation = nullptr;
    void* p = nullptr;
    std::size_t n = 0;
    int pattern = 0xd7;
    explicit Buffer(std::size_t bytes) : n(bytes) {
        if (n) {
            check(hipMalloc(&allocation, buffer_bytes(n)), "allocate guarded fixture owner");
            require(allocation != nullptr, "null fixture allocation");
            p = static_cast<std::byte*>(allocation) + guard_bytes;
            try { guard.run([&] { check(hipMemset(allocation, pattern, buffer_bytes(n)), "initialize fixture guards"); }); }
            catch (...) { guard.recover_or_fail_stop(); static_cast<void>(hipFree(allocation)); allocation = nullptr; throw; }
        }
    }
    ~Buffer() {
        if (allocation) { guard.recover_or_fail_stop(); static_cast<void>(hipFree(allocation)); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Span span() const { return {reinterpret_cast<std::uintptr_t>(p), n}; }
    void upload(const std::vector<std::byte>& data) {
        require(data.size() == n, "fixture upload extent");
        if (n) guard.run([&] { check(hipMemcpy(p, data.data(), n, hipMemcpyHostToDevice), "upload fixture input"); });
    }
    void poison(int value, hipStream_t stream) {
        pattern = value;
        if (n) check(hipMemsetAsync(allocation, value, buffer_bytes(n), stream), "poison complete guarded output");
    }
    void guards() const {
        if (!n) return;
        std::array<std::byte, guard_bytes> front{}, tail{};
        guard.run([&] {
            check(hipMemcpy(front.data(), allocation, guard_bytes, hipMemcpyDeviceToHost), "read leading guard");
            check(hipMemcpy(tail.data(), static_cast<const std::byte*>(p) + n, guard_bytes, hipMemcpyDeviceToHost), "read trailing guard");
        });
        const auto expected = static_cast<std::byte>(pattern);
        require(std::all_of(front.begin(), front.end(), [&](auto x) { return x == expected; }) &&
                std::all_of(tail.begin(), tail.end(), [&](auto x) { return x == expected; }), "fixture owner guard overwritten");
    }
};
std::vector<std::byte> read(const Buffer& owner) {
    std::vector<std::byte> data(owner.n);
    if (owner.n) owner.guard.run([&] { check(hipMemcpy(data.data(), owner.p, owner.n, hipMemcpyDeviceToHost), "read fixture payload"); });
    return data;
}
std::vector<std::byte> bytes(const std::vector<float>& values) {
    const auto view = std::as_bytes(std::span(values)); return {view.begin(), view.end()};
}
std::string digest(const std::vector<std::byte>& data) { return strix::model_content_identity_hex(strix::sha256_bytes(data)); }
std::vector<std::uint8_t> u8(const std::vector<std::byte>& data) {
    std::vector<std::uint8_t> result(data.size()); if (!data.empty()) std::memcpy(result.data(), data.data(), data.size()); return result;
}
std::vector<std::byte> byte_vector(const std::vector<std::uint8_t>& data) {
    std::vector<std::byte> result(data.size()); if (!data.empty()) std::memcpy(result.data(), data.data(), data.size()); return result;
}
std::size_t weight_bytes(Shape s, bool inject) {
    const auto g = strix::hc_repacked_down::geometry(s.width, s.rank);
    return buffer_bytes(static_cast<std::size_t>(g.original_bytes)) + buffer_bytes(static_cast<std::size_t>(g.plane_bytes)) +
        buffer_bytes(inject ? static_cast<std::size_t>(s.streams) * s.width * 4U : 0);
}
std::size_t fixture_bytes(Shape s, bool inject) {
    return buffer_bytes(static_cast<std::size_t>(s.rows) * s.width * 4U) +
        2 * buffer_bytes(static_cast<std::size_t>(s.rows) * s.rank * 4U) +
        2 * buffer_bytes(inject ? static_cast<std::size_t>(s.rows) * s.streams * 4U : 0);
}
struct WeightOwner {
    Shape shape;
    strix::hc_repacked_down::Geometry g;
    Buffer original, planes, inject;
    std::string original_sha, packed_sha, inject_sha, codes_sha, scales_sha;
    WeightOwner(Shape s, const std::vector<std::byte>& down, const std::vector<std::byte>& injection)
        : shape(s), g(strix::hc_repacked_down::geometry(s.width, s.rank)),
          original(down.size()), planes(static_cast<std::size_t>(g.plane_bytes)), inject(injection.size()) {
        require(down.size() == g.original_bytes && (injection.empty() || injection.size() == static_cast<std::size_t>(s.streams) * s.width * 4U), "weight fixture extents differ");
        auto source = u8(down); std::vector<std::uint8_t> packed(static_cast<std::size_t>(g.plane_bytes)), roundtrip(source.size());
        strix::hc_repacked_down::pack(g, source, packed);
        require(strix::hc_repacked_down::padding_zero(g, packed), "packed padding not zero");
        strix::hc_repacked_down::unpack(g, packed, roundtrip); require(roundtrip == source, "fixture repack roundtrip differs");
        original.upload(down); planes.upload(byte_vector(packed)); inject.upload(injection);
        original_sha = digest(down); packed_sha = digest(byte_vector(packed)); inject_sha = digest(injection);
        const auto code_end = packed.begin() + static_cast<std::ptrdiff_t>(g.codes_bytes);
        codes_sha = digest(byte_vector(std::vector<std::uint8_t>(packed.begin(), code_end)));
        scales_sha = digest(byte_vector(std::vector<std::uint8_t>(packed.begin() + static_cast<std::ptrdiff_t>(g.scales_offset), packed.begin() + static_cast<std::ptrdiff_t>(g.scales_offset + g.scales_bytes))));
        readonly();
    }
    WeightOwner(Shape s, const std::vector<std::byte>& down, const std::vector<float>& injection) : WeightOwner(s, down, bytes(injection)) {}
    Span codes() const { return {planes.span().address, g.codes_bytes}; }
    Span scales() const { return {planes.span().address + g.scales_offset, g.scales_bytes}; }
    void readonly() const {
        require(digest(read(original)) == original_sha && digest(read(planes)) == packed_sha && digest(read(inject)) == inject_sha, "read-only original/packed/injection bytes changed");
        original.guards(); planes.guards(); inject.guards();
    }
    std::size_t resident_bytes() const { return buffer_bytes(original.n) + buffer_bytes(planes.n) + buffer_bytes(inject.n); }
};
struct Stream {
    strix::HipSetupQuiescenceGuard guard; hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "create fixture stream"); }
    ~Stream() { if (value) { guard.recover_or_fail_stop(); static_cast<void>(hipStreamDestroy(value)); } }
};
struct Graph {
    strix::HipSetupQuiescenceGuard guard; hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
    ~Graph() { guard.recover_or_fail_stop(); if (executable) static_cast<void>(hipGraphExecDestroy(executable)); if (graph) static_cast<void>(hipGraphDestroy(graph)); }
};
struct Events {
    hipEvent_t start = nullptr, stop = nullptr;
    Events() { check(hipEventCreate(&start), "create start event"); try { check(hipEventCreate(&stop), "create stop event"); } catch (...) { static_cast<void>(hipEventDestroy(start)); throw; } }
    ~Events() { if (start) static_cast<void>(hipEventDestroy(start)); if (stop) static_cast<void>(hipEventDestroy(stop)); }
};
struct Fixture {
    Shape shape; bool injection;
    std::shared_ptr<WeightOwner> weights;
    Buffer normalized, base_low, base_inject, test_low, test_inject;
    std::string normalized_sha;
    std::vector<std::byte> expected;
    Fixture(Shape s, const std::vector<std::byte>& n, const std::vector<std::byte>& w, const std::vector<std::byte>& i, std::shared_ptr<WeightOwner> shared = {})
        : shape(s), injection(shared ? shared->inject.n != 0 : !i.empty()),
          weights(shared ? std::move(shared) : std::make_shared<WeightOwner>(s, w, i)),
          normalized(n.size()), base_low(static_cast<std::size_t>(s.rows) * s.rank * 4U),
          base_inject(injection ? static_cast<std::size_t>(s.rows) * s.streams * 4U : 0),
          test_low(base_low.n), test_inject(base_inject.n) {
        require(s.width == weights->shape.width && s.rank == weights->shape.rank && s.streams == weights->shape.streams &&
                n.size() == static_cast<std::size_t>(s.rows) * s.width * 4U, "shared fixture shape/normalized extent differs");
        if (!w.empty()) require(digest(w) == weights->original_sha, "shared original tensor differs");
        if (!i.empty()) require(digest(i) == weights->inject_sha, "shared injection tensor differs");
        normalized.upload(n); normalized_sha = digest(n);
        poison(false, nullptr); reference(nullptr); check(hipDeviceSynchronize(), "drain rows8 baseline"); expected = semantic(false); guards(false); readonly();
    }
    Fixture(Shape s, const std::vector<float>& n, const std::vector<std::byte>& w, const std::vector<float>& i, std::shared_ptr<WeightOwner> shared = {})
        : Fixture(s, bytes(n), w, bytes(i), std::move(shared)) {}
    void set_injection_enabled(bool enabled) {
        require(!enabled || weights->inject.n != 0, "no injection owner to enable"); injection = enabled;
        poison(false, nullptr); reference(nullptr); check(hipDeviceSynchronize(), "drain changed injection baseline"); expected = semantic(false); guards(false);
    }
    Span inject_span() const { return injection ? weights->inject.span() : Span{}; }
    Span output_span(bool candidate) const { return injection ? (candidate ? test_inject.span() : base_inject.span()) : Span{}; }
    void reference(hipStream_t stream) {
        strix::hc_projection_diagnostic::enqueue(shape, normalized.span(), weights->original.span(), inject_span(), base_low.span(), output_span(false), 0, 1, true, stream);
    }
    void launch(Schedule schedule, hipStream_t stream) {
        strix::hc_repacked_projection_diagnostic::enqueue(shape, normalized.span(), weights->codes(), weights->scales(), weights->g.rank_pad, inject_span(), test_low.span(), output_span(true), schedule, stream);
    }
    void poison(bool candidate, hipStream_t stream) {
        auto& low = candidate ? test_low : base_low; auto& inj = candidate ? test_inject : base_inject;
        low.poison(candidate ? 0x5a : 0xa5, stream); inj.poison(candidate ? 0x5a : 0xa5, stream);
    }
    std::vector<std::byte> semantic(bool candidate) const {
        auto result = read(candidate ? test_low : base_low);
        if (injection) { auto tail = read(candidate ? test_inject : base_inject); result.insert(result.end(), tail.begin(), tail.end()); }
        return result;
    }
    void guards(bool candidate) const { (candidate ? test_low : base_low).guards(); (candidate ? test_inject : base_inject).guards(); }
    void verify(bool candidate) const { require(semantic(candidate) == expected, "repacked projection differs from old rows8 complete FP32 bits"); guards(candidate); }
    void readonly() const { require(digest(read(normalized)) == normalized_sha, "normalized owner changed"); normalized.guards(); weights->readonly(); }
    std::size_t resident_bytes() const { return buffer_bytes(normalized.n) + buffer_bytes(base_low.n) + buffer_bytes(base_inject.n) + buffer_bytes(test_low.n) + buffer_bytes(test_inject.n); }
    void capture(Graph& graph, Schedule schedule, hipStream_t stream, bool candidate = true) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin repacked graph");
        try { if (candidate) launch(schedule, stream); else reference(stream); }
        catch (...) { static_cast<void>(hipStreamEndCapture(stream, &graph.graph)); throw; }
        check(hipStreamEndCapture(stream, &graph.graph), "end repacked graph");
        check(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate repacked graph");
    }
};
std::string quoted(const std::string& text) {
    std::string result = "\"";
    for (char raw : text) { const auto c = static_cast<unsigned char>(raw); require(c >= 32, "control character in receipt string"); if (c == '"' || c == '\\') result += '\\'; result += raw; }
    return result + '"';
}
double median(std::vector<double> values) { require(!values.empty(), "empty sample"); std::sort(values.begin(), values.end()); const auto n = values.size(); return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * 0.5; }
void array(std::ostream& out, const std::vector<double>& values) { out << '['; for (std::size_t n = 0; n < values.size(); ++n) out << (n ? "," : "") << values[n]; out << ']'; }

struct Data { std::vector<float> normalized, inject; std::vector<std::byte> down; };
void put_scale(Data& d, Shape s, unsigned output, unsigned block, unsigned bits) {
    const auto at = (static_cast<std::size_t>(output) * (s.width / 32U) + block) * 34U;
    d.down[at] = static_cast<std::byte>(bits & 255U); d.down[at + 1] = static_cast<std::byte>(bits >> 8U);
}
void put_code(Data& d, Shape s, unsigned output, unsigned feature, int code) {
    const auto at = (static_cast<std::size_t>(output) * (s.width / 32U) + feature / 32U) * 34U + 2U + feature % 32U;
    d.down[at] = static_cast<std::byte>(static_cast<std::uint8_t>(code));
}
Data data(Shape s, bool injection, unsigned flavor) {
    Data d; d.normalized.resize(static_cast<std::size_t>(s.rows) * s.width);
    d.inject.resize(injection ? static_cast<std::size_t>(s.streams) * s.width : 0);
    d.down.resize(static_cast<std::size_t>(s.rank) * (s.width / 32U) * 34U);
    for (std::size_t j = 0; j < d.normalized.size(); ++j) d.normalized[j] = static_cast<float>(static_cast<int>(j % 47U) - 23) * 0.03125F;
    for (std::size_t j = 0; j < d.inject.size(); ++j) d.inject[j] = static_cast<float>(static_cast<int>(j % 31U) - 15) * 0.0009765625F;
    for (unsigned o = 0; o < s.rank; ++o) for (unsigned b = 0; b < s.width / 32U; ++b) {
        put_scale(d, s, o, b, 0x2400U + (b + o) % 128U);
        for (unsigned j = 0; j < 32; ++j) put_code(d, s, o, b * 32U + j, static_cast<int>((b * 17U + o * 19U + j * 59U) & 255U) - 128);
    }
    const std::array<unsigned, 3> witnesses{0U, s.width / 2U, s.width - 1U};
    if (flavor == 1) { // Signed zeros, negative-scale code zero, exact cancellation.
        std::fill(d.normalized.begin(), d.normalized.end(), 0.0F);
        std::fill(d.inject.begin(), d.inject.end(), -0.0F);
        for (unsigned o = 0; o < s.rank; ++o) for (unsigned b = 0; b < s.width / 32U; ++b) {
            put_scale(d, s, o, b, 0xbc00U); for (unsigned j = 0; j < 32; ++j) put_code(d, s, o, b * 32U + j, 0);
        }
        for (unsigned row = 0; row < s.rows; ++row) { d.normalized[static_cast<std::size_t>(row) * s.width] = 1; d.normalized[static_cast<std::size_t>(row) * s.width + 1U] = 1; }
        // Even outputs contain only -0 products, distinguishing the required
        // +0 accumulator from a -0 start or a folded first product. Odd outputs
        // additionally retain the nonzero exact-cancellation witness.
        for (unsigned o = 1; o < s.rank; o += 2U) { put_code(d, s, o, 0, 1); put_code(d, s, o, 1, -1); }
    }
    if (flavor == 2 || flavor == 3) { // Distinct qNaN/sNaN payloads at early/middle/last features.
        for (unsigned row = 0; row < s.rows; ++row) for (unsigned k = 0; k < witnesses.size(); ++k)
            d.normalized[static_cast<std::size_t>(row) * s.width + witnesses[k]] = std::bit_cast<float>((flavor == 2 ? 0x7fc12345U : 0x7f812345U) + k * 0x111U);
    }
    if (flavor == 4 || flavor == 5) { // Same feature has two different NaNs.
        for (unsigned f : witnesses) {
            const unsigned b = f / 32U;
            for (unsigned o = 0; o < s.rank; ++o) put_scale(d, s, o, b, flavor == 4 ? 0xfe55U : 0x7c15U);
            // A NaN half scale affects ALL32 products, so every feature in its
            // block carries the distinct normalized NaN; later terms cannot
            // silently mask a wrong operand-priority witness within the block.
            for (unsigned row = 0; row < s.rows; ++row) for (unsigned j = 0; j < 32; ++j)
                d.normalized[static_cast<std::size_t>(row) * s.width + b * 32U + j] = std::bit_cast<float>(flavor == 4 ? 0x7fc34567U : 0xff834567U);
        }
    }
    if (flavor == 6) { // Inf*zero, Inf-Inf and overflow; compare the device payload/class bits.
        for (unsigned row = 0; row < s.rows; ++row) {
            const auto at = static_cast<std::size_t>(row) * s.width;
            d.normalized[at] = std::numeric_limits<float>::infinity(); d.normalized[at + 1] = -std::numeric_limits<float>::infinity();
            d.normalized[at + s.width - 1] = std::numeric_limits<float>::max();
        }
        for (unsigned o = 0; o < s.rank; ++o) { put_code(d, s, o, 0, o % 2U ? 0 : 127); put_code(d, s, o, 1, 127); put_scale(d, s, o, s.width / 32U - 1U, 0x7bffU); put_code(d, s, o, s.width - 1U, 127); }
    }
    if (flavor == 7) { // FP32/half subnormals, tiny sums/epilogues and signed versions.
        constexpr std::array<unsigned, 8> patterns{1U, 0x80000001U, 0x007fffffU, 0x807fffffU, 0x00800000U, 0x80800000U, 0x0d800001U, 0x8d800001U};
        for (std::size_t j = 0; j < d.normalized.size(); ++j) d.normalized[j] = std::bit_cast<float>(patterns[j % patterns.size()]);
        for (unsigned o = 0; o < s.rank; ++o) for (unsigned b = 0; b < s.width / 32U; ++b) { put_scale(d, s, o, b, b % 2U ? 0x03ffU : 0x8001U); for (unsigned j = 0; j < 32; ++j) put_code(d, s, o, b * 32U + j, 127); }
        for (std::size_t j = 0; j < d.inject.size(); ++j) d.inject[j] = std::bit_cast<float>(j % 2U ? 1U : 0x00800001U);
    }
    if (flavor == 8) { // Full-mantissa cancellation with a constructive fused/unfused witness.
        std::fill(d.normalized.begin(), d.normalized.end(), 0.0F); std::fill(d.inject.begin(), d.inject.end(), 0.0F);
        std::fill(d.down.begin(), d.down.end(), std::byte{0});
        const float x = std::bit_cast<float>(0x3f800001U), coefficient = (1.0F + 1.0F / 1024.0F) * 3.0F;
        volatile float rounded_product = x * coefficient;
        const float first = -rounded_product;
        const float fused = std::fmaf(x, coefficient, first); volatile float product = x * coefficient; const float unfused = first + product;
        const auto epilogue = [](float v) { const float z = v / 4.0F; return z * (1.0F / (1.0F + std::exp(-z))); };
        require(std::isfinite(fused) && std::bit_cast<std::uint32_t>(epilogue(fused)) != std::bit_cast<std::uint32_t>(epilogue(unfused)), "full-mantissa fixture lacks fused/unfused epilogue witness");
        for (unsigned row = 0; row < s.rows; ++row) { const auto at = static_cast<std::size_t>(row) * s.width; d.normalized[at] = first; d.normalized[at + 32U] = x; }
        for (unsigned o = 0; o < s.rank; ++o) { put_scale(d, s, o, 0, 0x3c00U); put_code(d, s, o, 0, 1); put_scale(d, s, o, 1, 0x3c01U); put_code(d, s, o, 32, 3); }
        if (injection) for (unsigned stream = 0; stream < s.streams; ++stream) { d.inject[static_cast<std::size_t>(stream) * s.width] = 1.0F; d.inject[static_cast<std::size_t>(stream) * s.width + 32U] = coefficient; }
    }
    if (flavor == 9) { // Extreme codes, scales and signed overflow.
        constexpr std::array<unsigned, 6> scales{0x7bffU, 0xfbffU, 0x0000U, 0x8000U, 0x7c00U, 0xfc00U};
        constexpr std::array<int, 3> codes{-128, 0, 127};
        for (unsigned o = 0; o < s.rank; ++o) for (unsigned b = 0; b < s.width / 32U; ++b) { put_scale(d, s, o, b, scales[(o + b) % scales.size()]); for (unsigned j = 0; j < 32; ++j) put_code(d, s, o, b * 32U + j, codes[j % codes.size()]); }
    }
    if (flavor == 10) { // Accumulator NaN followed by a differently sourced product NaN.
        for (unsigned row = 0; row < s.rows; ++row) {
            const auto at = static_cast<std::size_t>(row) * s.width;
            d.normalized[at] = std::bit_cast<float>(0x7fc13579U);
            for (unsigned j = 0; j < 32; ++j) d.normalized[at + s.width - 32U + j] = std::bit_cast<float>(0xffc24681U);
        }
        for (unsigned o = 0; o < s.rank; ++o) put_scale(d, s, o, s.width / 32U - 1U, 0x7e35U);
    }
    if (flavor == 11 && injection) { // F32 injection q/sNaNs at all three feature witnesses.
        for (unsigned stream = 0; stream < s.streams; ++stream) for (unsigned k = 0; k < witnesses.size(); ++k) {
            d.inject[static_cast<std::size_t>(stream) * s.width + witnesses[k]] = std::bit_cast<float>((stream % 2U ? 0xff812357U : 0x7fc35791U) + k);
            for (unsigned row = 0; row < s.rows; ++row) d.normalized[static_cast<std::size_t>(row) * s.width + witnesses[k]] = std::bit_cast<float>(0x7fc56789U + k);
        }
    }
    return d;
}
void exercise(Fixture& f, const std::vector<Schedule>& admitted, bool repeats = false) {
    Stream stream;
    for (Schedule schedule : admitted) {
        f.poison(true, stream.value); f.launch(schedule, stream.value); check(hipStreamSynchronize(stream.value), "drain eager repacked fixture"); f.verify(true); f.readonly();
        Graph graph; f.capture(graph, schedule, stream.value);
        for (unsigned replay = 0; replay < 3; ++replay) { f.poison(true, stream.value); check(hipGraphLaunch(graph.executable, stream.value), "replay repacked fixture"); check(hipStreamSynchronize(stream.value), "drain repacked replay"); f.verify(true); f.readonly(); }
        if (repeats) for (unsigned repeat = 0; repeat < 100; ++repeat) { f.poison(true, stream.value); f.launch(schedule, stream.value); check(hipStreamSynchronize(stream.value), "drain repeated eager fixture"); f.verify(true); }
        f.readonly();
    }
}
void first_last_controls(Shape s, bool injection, const std::vector<Schedule>& admitted) {
    Data d = data(s, injection, 0); std::fill(d.normalized.begin(), d.normalized.end(), 0.0F); std::fill(d.inject.begin(), d.inject.end(), 0.0F); std::fill(d.down.begin(), d.down.end(), std::byte{0});
    for (unsigned row = 0; row < s.rows; ++row) { const auto at = static_cast<std::size_t>(row) * s.width; d.normalized[at] = 1.0F; d.normalized[at + s.width - 1U] = 1.0F; }
    for (unsigned o = 0; o < s.rank; ++o) for (unsigned b = 0; b < s.width / 32U; ++b) put_scale(d, s, o, b, 0x3c00U);
    Fixture base(s, d.normalized, d.down, d.inject); exercise(base, admitted);
    const auto mutation = [&](unsigned output, unsigned feature) {
        Data changed = d; put_code(changed, s, output, feature, 1);
        Fixture witness(s, changed.normalized, changed.down, changed.inject);
        const auto offset = static_cast<std::size_t>(output) * 4U;
        require(!std::equal(base.expected.begin() + static_cast<std::ptrdiff_t>(offset), base.expected.begin() + static_cast<std::ptrdiff_t>(offset + 4U), witness.expected.begin() + static_cast<std::ptrdiff_t>(offset)), "first/last coefficient mutation is not visible in rows8 target output");
        exercise(witness, admitted);
    };
    mutation(s.rank - 1U, 0); mutation(s.rank - 1U, s.width - 1U);
    if (s.rank > 31U && (s.rank - 1U) % 32U != 31U) mutation(31U, s.width - 1U);
    if (injection) {
        Data changed = d; changed.inject.back() = 0.5F; Fixture witness(s, changed.normalized, changed.down, changed.inject);
        const auto offset = static_cast<std::size_t>(s.rows) * s.rank * 4U + (s.streams - 1U) * 4U;
        require(!std::equal(base.expected.begin() + static_cast<std::ptrdiff_t>(offset), base.expected.begin() + static_cast<std::ptrdiff_t>(offset + 4U), witness.expected.begin() + static_cast<std::ptrdiff_t>(offset)), "last injection weight mutation is not visible in rows8 stream3");
        exercise(witness, admitted);
    }
}

void negative_cases(const std::vector<Schedule>& admitted) {
    const Shape shape{256, 4, 1, 1}; const auto d = data(shape, true, 0); Fixture f(shape, d.normalized, d.down, d.inject);
    f.poison(true, nullptr); check(hipDeviceSynchronize(), "drain negative output poison");
    const auto low_before = read(f.test_low), injection_before = read(f.test_inject);
    const auto n = f.normalized.span(), c = f.weights->codes(), s = f.weights->scales(), i = f.inject_span(), l = f.test_low.span(), o = f.output_span(true);
    const auto rejected = [&](Shape sh, Span nn, Span cc, Span ss, unsigned rank_pad, Span ii, Span ll, Span oo, Schedule schedule, const char* message) {
        bool caught = false;
        try { strix::hc_repacked_projection_diagnostic::enqueue(sh, nn, cc, ss, rank_pad, ii, ll, oo, schedule, nullptr); }
        catch (const std::invalid_argument& error) { caught = std::string(error.what()) == message; }
        require(caught, "repacked negative must throw precise invalid_argument before enqueue");
        require(read(f.test_low) == low_before && read(f.test_inject) == injection_before, "negative admission modified poisoned payload");
        f.guards(true); f.readonly();
    };
    constexpr auto geometry = "HC repacked projection diagnostic unsupported shape/schedule";
    constexpr auto pair = "HC repacked projection diagnostic injection owner pair differs";
    constexpr auto extent = "HC repacked projection diagnostic owner extent/alignment differs";
    constexpr auto overlap = "HC repacked projection diagnostic owners overlap";
    for (Schedule schedule : admitted) {
        const auto reject = [&](Shape sh, Span nn, Span cc, Span ss, unsigned rp, Span ii, Span ll, Span oo, const char* message) { rejected(sh, nn, cc, ss, rp, ii, ll, oo, schedule, message); };
        for (unsigned width : {0U, 255U, 257U, 10496U}) { auto bad = shape; bad.width = width; reject(bad, n, c, s, 32, i, l, o, geometry); }
        for (unsigned streams : {0U, 1U, 2U, 3U, 5U, 8U}) { auto bad = shape; bad.streams = streams; reject(bad, n, c, s, 32, i, l, o, geometry); }
        for (unsigned rank : {0U, 513U}) { auto bad = shape; bad.rank = rank; reject(bad, n, c, s, 32, i, l, o, geometry); }
        for (unsigned rows : {0U, 65536U}) { auto bad = shape; bad.rows = rows; reject(bad, n, c, s, 32, i, l, o, geometry); }
        for (unsigned rp : {0U, 31U, 64U}) reject(shape, n, c, s, rp, i, l, o, geometry);
        for (unsigned bad : {0U, 4U, std::numeric_limits<unsigned>::max()}) rejected(shape, n, c, s, 32, i, l, o, static_cast<Schedule>(bad), geometry);
        reject(shape, n, c, s, 32, {}, l, o, pair); reject(shape, n, c, s, 32, i, l, {}, pair);
        reject(shape, n, c, s, 32, {0, 1}, l, {}, pair); reject(shape, n, c, s, 32, {}, l, {0, 1}, pair);
        // Exact required extents are used by every positive fixture. Here every
        // owner is independently one byte too small, null, misaligned, then
        // wrapping at address+FULLprovidedbytes (including unused tail bytes).
        const std::array<Span, 6> good{n, c, s, i, l, o};
        const std::array<std::uint64_t, 6> alignment{16, 256, 256, 16, 4, 4};
        const auto call = [&](const std::array<Span, 6>& a, const char* message) { reject(shape, a[0], a[1], a[2], 32, a[3], a[4], a[5], message); };
        for (std::size_t k = 0; k < good.size(); ++k) {
            auto bad = good; --bad[k].bytes; call(bad, extent);
            bad = good; bad[k].bytes = 0; call(bad, extent);
            bad = good; bad[k].address += alignment[k] == 4 ? 1U : 4U; call(bad, extent);
            bad = good; bad[k].address = 0; call(bad, k == 3 || k == 5 ? pair : extent);
            bad = good; bad[k].address = std::numeric_limits<std::uintptr_t>::max() & ~static_cast<std::uintptr_t>(alignment[k] - 1U); bad[k].bytes = std::numeric_limits<std::uint64_t>::max(); call(bad, extent);
        }
        for (std::size_t write = 4; write < 6; ++write) for (std::size_t read_index = 0; read_index < 4; ++read_index) { auto bad = good; bad[write].address = good[read_index].address; call(bad, overlap); }
        { auto bad = good; bad[5].address = l.address; call(bad, overlap); }
        { auto bad = good; bad[2].address = c.address; call(bad, overlap); }
        // Overlap exists only in a provided read owner's tail, beyond its
        // logical required data. The API promises to validate the FULL span.
        { auto bad = good; bad[0].bytes += 16U; bad[4].address = n.address + n.bytes; call(bad, overlap); }
        // Failed host API call establishes a stale error without any model or
        // new test kernel. If the runtime cannot establish it, fail honestly.
        static_cast<void>(hipGetLastError());
        const auto seed_error = hipSetDevice(-1);
        require(seed_error != hipSuccess && hipPeekAtLastError() != hipSuccess, "runtime did not expose deliberately seeded stale error");
        bool stale = false;
        try { strix::hc_repacked_projection_diagnostic::enqueue(shape, n, c, s, 32, i, l, o, schedule, nullptr); }
        catch (const std::runtime_error& error) { stale = std::string(error.what()).starts_with("HC repacked projection diagnostic stale error:"); }
        static_cast<void>(hipGetLastError());
        require(stale, "repacked admission did not reject stale error before launch");
        require(read(f.test_low) == low_before && read(f.test_inject) == injection_before, "stale-error admission changed outputs"); f.guards(true); f.readonly();
    }
}
void synthetic(const std::vector<Schedule>& admitted) {
    require(std::endian::native == std::endian::little, "repacked fixture requires little endian");
    if (const char* arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); arch && *arch) throw std::invalid_argument("architecture override forbidden");
    int device = -1; check(hipGetDevice(&device), "observe native fixture device"); hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, device), "observe native fixture architecture");
    require(std::string(properties.gcnArchName).starts_with("gfx1151") && properties.warpSize == 32, "repacked fixture needs observed gfx1151/wave32");
    struct Case { Shape shape; bool injection; };
    const std::array<Case, 9> cases{{{{256, 4, 1, 1}, true}, {{256, 4, 31, 2}, true}, {{2048, 4, 32, 1}, false}, {{2048, 4, 33, 43}, true}, {{10240, 4, 316, 1}, true}, {{10240, 4, 318, 2}, false}, {{10240, 4, 320, 1}, true}, {{10240, 4, 320, 43}, true}, {{10240, 4, 512, 1}, true}}};
    std::size_t count = 0;
    for (const auto& c : cases) {
        for (unsigned flavor = 0; flavor < 12; ++flavor) {
            const auto input = data(c.shape, c.injection, flavor); Fixture f(c.shape, input.normalized, input.down, input.inject);
            if (flavor == 1) for (unsigned row = 0; row < c.shape.rows; ++row) for (unsigned o = 0; o < c.shape.rank; o += 2U) {
                std::uint32_t bits = 0;
                const auto at = (static_cast<std::size_t>(row) * c.shape.rank + o) * sizeof(bits);
                std::memcpy(&bits, f.expected.data() + at, sizeof(bits));
                require(bits == 0x00000000U, "signed-zero fixture power: rows8 even output is not raw +0");
            }
            exercise(f, admitted, flavor == 0 && c.shape.width == 10240 && c.shape.rank == 320 && c.shape.rows == 1);
            ++count;
        }
        first_last_controls(c.shape, c.injection, admitted);
    }
    negative_cases(admitted);
    // Positive rows boundary uses real owners, not an out-of-bounds fabricated
    // pointer. No loops/replay multiplication beyond the ordinary A4 protocol.
    { const Shape edge{256, 4, 1, 65535}; const auto input = data(edge, false, 0); Fixture f(edge, input.normalized, input.down, input.inject); exercise(f, admitted); }
    std::cout << "HC repacked A4 synthetic cases=" << count << " shapes=9 value_classes=12 admitted_schedules=";
    for (std::size_t k = 0; k < admitted.size(); ++k) std::cout << (k ? "," : "") << static_cast<unsigned>(admitted[k]);
    std::cout << " fullbits eager+graph3 first/last-controls eager100 rows65535 negatives guards readonly PASS\n";
}
} // namespace repacked_test

#include "hip_repacked_hc_capture_qualification.hpp"

int main(int argc, char** argv) try {
    if (argc == 1) {
        const char* text = std::getenv("STRIX_HC_REPACKED_ADMITTED_SCHEDULES");
        repacked_test::require(text != nullptr, "STRIX_HC_REPACKED_ADMITTED_SCHEDULES required before HIP");
        const auto admitted = repacked_test::parse_schedules(text);
        repacked_test::synthetic(admitted); return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--synthetic" && std::string(argv[2]) == "--schedules") {
        const auto admitted = repacked_test::parse_schedules(argv[3]);
        repacked_test::synthetic(admitted); return 0;
    }
    repacked_test::require(repacked_test::run_capture_cli(argc, argv), "expected --synthetic --schedules or --capture-qualify");
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }

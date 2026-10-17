#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
using View = GdnKnownWindowProjectionView;
using Direction = GdnKnownWindowProjectionDirection;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
std::vector<std::byte> read(View view) {
    std::vector<std::byte> result(static_cast<std::size_t>(view.bytes));
    check(hipMemcpy(result.data(), reinterpret_cast<const void*>(view.address), result.size(), hipMemcpyDeviceToHost), "read wide GDN bytes");
    return result;
}
void write(View view, const std::vector<std::byte>& bytes) {
    require(view.bytes == bytes.size(), "wide GDN upload size differs");
    check(hipMemcpy(reinterpret_cast<void*>(view.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload wide GDN bytes");
}
template<class T> std::vector<std::byte> bytes_of(const std::vector<T>& values) {
    std::vector<std::byte> result(values.size() * sizeof(T));
    std::memcpy(result.data(), values.data(), result.size()); return result;
}
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create wide GDN stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (const auto graph : graphs) if (hipGraphDestroy(graph) != hipSuccess) std::_Exit(3);
        for (auto* allocation : allocations) if (hipFree(allocation) != hipSuccess) std::_Exit(3);
        if (hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    View allocate(std::uint64_t bytes) {
        require(bytes && bytes <= (1ULL << 24U), "wide GDN fixture allocation exceeds bound");
        void* allocation = nullptr;
        check(hipMalloc(&allocation, static_cast<std::size_t>(bytes + 512U)), "allocate guarded wide GDN owner");
        try { allocations.push_back(allocation); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipFree(allocation); throw; }
        check(hipMemset(allocation, 0xa5, static_cast<std::size_t>(bytes + 512U)), "initialize wide GDN guards");
        return {reinterpret_cast<std::uintptr_t>(allocation) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "complete wide GDN projection"); guard.require_original(); }
};
void guards(View view) {
    for (const auto address : {view.address - 256U, view.address + view.bytes}) {
        const auto bytes = read({address, 256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == std::byte{0xa5}; }),
            "wide GDN exterior guard changed");
    }
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
// Encode exact Q8_0 bytes, without host struct packing or implicit half casts.
std::vector<std::byte> packed_weights(std::uint32_t width, std::uint32_t outputs, unsigned flavor) {
    const auto blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::byte> result(blocks * 34U);
    constexpr std::array<std::uint16_t, 8> scales{0x0000,0x8000,0x0001,0x8001,0x0400,0x8400,0x2000,0xa000};
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto scale = flavor ? scales[block % scales.size()] : static_cast<std::uint16_t>(0x2400U + (block % 3U) * 0x400U);
        result[block * 34U] = static_cast<std::byte>(scale & 0xffU);
        result[block * 34U + 1U] = static_cast<std::byte>(scale >> 8U);
        for (std::size_t lane = 0; lane < 32U; ++lane) {
            const auto code = flavor == 2U ? (lane % 2U ? -128 : 127) :
                static_cast<int>((block * 7U + lane * 3U) % 29U) - 14;
            result[block * 34U + 2U + lane] = static_cast<std::byte>(static_cast<std::uint8_t>(static_cast<std::int8_t>(code)));
        }
    }
    return result;
}
std::vector<std::byte> input_bytes(std::uint32_t width, unsigned row, bool input_bf16, unsigned flavor) {
    std::vector<std::uint16_t> halves(width);
    std::vector<float> floats(width);
    constexpr std::array<std::uint16_t, 8> edges{0,0x8000,1,0x8001,0x0080,0x8080,0x3e80,0xbe80};
    for (std::uint32_t i = 0; i < width; ++i) {
        const auto bits = flavor == 1U ? edges[(i + row) % edges.size()] :
            bf16((flavor == 2U ? (i % 2U ? -0.125F : 0.125F) :
                static_cast<float>(static_cast<int>((i * 5U + row * 11U) % 31U) - 15) * 0.015625F) * static_cast<float>(row + 1U));
        halves[i] = bits;
        floats[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return input_bf16 ? bytes_of(halves) : bytes_of(floats);
}
struct Fixture {
    Owner owner;
    unsigned rows;
    std::uint32_t width, outputs;
    Direction direction;
    View weights;
    std::vector<View> input, output;
    std::vector<std::vector<std::byte>> original_input;
    std::vector<std::byte> original_weights;
    Fixture(unsigned count, std::uint32_t inner, std::uint32_t outer, Direction kind)
        : rows(count), width(inner), outputs(outer), direction(kind) {
        const bool bf = direction == Direction::bf16_to_fp32;
        weights = owner.allocate(static_cast<std::uint64_t>(width / 32U) * outputs * 34U);
        for (unsigned row = 0; row < rows; ++row) {
            input.push_back(owner.allocate(static_cast<std::uint64_t>(width) * (bf ? 2U : 4U)));
            output.push_back(owner.allocate(static_cast<std::uint64_t>(outputs) * (bf ? 4U : 2U)));
        }
    }
    void prepare(unsigned flavor) {
        original_weights = packed_weights(width, outputs, flavor); write(weights, original_weights);
        original_input.clear();
        for (unsigned row = 0; row < rows; ++row) {
            original_input.push_back(input_bytes(width, row, direction == Direction::bf16_to_fp32, flavor));
            write(input[row], original_input.back());
        }
    }
    void unchanged() const {
        require(read(weights) == original_weights, "wide GDN mutated packed weight bytes"); guards(weights);
        for (unsigned row = 0; row < rows; ++row) {
            require(read(input[row]) == original_input[row], "wide GDN mutated row input bytes");
            guards(input[row]); guards(output[row]);
        }
    }
    std::vector<std::vector<std::byte>> result() const {
        std::vector<std::vector<std::byte>> values;
        for (const auto view : output) values.push_back(read(view));
        return values;
    }
    void poison() const {
        for (const auto view : output)
            check(hipMemset(reinterpret_cast<void*>(view.address), 0xa5, static_cast<std::size_t>(view.bytes)), "poison complete wide GDN outputs");
    }
    std::vector<std::vector<std::byte>> run(bool wide) {
        poison();
        if (wide) gdn_known_window_rows_projection_probe(direction, input, weights, output, width, outputs, owner.stream);
        else for (unsigned first = 0; first < rows; first += 2U) {
            const std::array<View,2> x{input[first],input[first + 1U]}, y{output[first],output[first + 1U]};
            gdn_known_window_r2_projection_probe(direction, GdnKnownWindowProjectionMode::incumbent_row_loop,
                x, weights, y, width, outputs, owner.stream);
        }
        owner.finish(); unchanged(); return result();
    }
    void rejected(unsigned mutation, bool capture) {
        poison();
        const auto before = result();
        auto x = input, y = output; auto w = weights;
        auto inner = width, outer = outputs; auto kind = direction;
        void* stream = owner.stream;
        if (mutation == 0U) x.pop_back();
        if (mutation == 1U) y.pop_back();
        if (mutation == 2U) x.back().address += 2U;
        if (mutation == 3U) --x.back().bytes;
        if (mutation == 4U) --y.back().bytes;
        if (mutation == 5U) --w.bytes;
        if (mutation == 6U) y.back().address = x.front().address;
        if (mutation == 7U) y.back().address = w.address;
        if (mutation == 8U) y.back() = y.front();
        if (mutation == 9U) kind = static_cast<Direction>(999U);
        if (mutation == 10U) inner = 33U;
        if (mutation == 11U) outer = 0U;
        if (mutation == 12U) stream = nullptr;
        if (mutation == 13U) x.back().address = 0;
        if (mutation == 14U) { x.clear(); y.clear(); }
        if (mutation == 15U) { x.resize(3U); y.resize(2U); }
        if (mutation == 16U) y.back().address += 2U;
        if (mutation == 17U) w.address += 2U;
        bool invalid = false;
        if (capture) check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin invalid wide GDN capture");
        try { gdn_known_window_rows_projection_probe(kind, x, w, y, inner, outer, stream); }
        catch (const std::invalid_argument&) { invalid = true; }
        catch (...) {
            if (capture) {
                hipGraph_t graph = nullptr;
                const auto ended = hipStreamEndCapture(owner.stream, &graph);
                if (graph) owner.graphs.push_back(graph);
                if (ended != hipSuccess && ended != hipErrorStreamCaptureInvalidated) std::_Exit(3);
            }
            throw;
        }
        if (capture) {
            hipGraph_t graph = nullptr;
            check(hipStreamEndCapture(owner.stream, &graph), "end invalid wide GDN capture");
            require(graph != nullptr, "invalid wide GDN capture omitted its graph");
            owner.graphs.push_back(graph);
            std::size_t nodes = 0;
            check(hipGraphGetNodes(graph, nullptr, &nodes), "count invalid wide GDN graph nodes");
            require(nodes == 0U, "invalid wide GDN submitted work before rejection");
        }
        owner.finish();
        require(invalid, "malformed wide GDN metadata was not rejected");
        require(result() == before, "invalid wide GDN modified an output"); unchanged();
    }
};
} // namespace
int main() {
    try {
        for (const unsigned rows : {4U,8U}) for (const auto direction : {Direction::bf16_to_fp32,Direction::fp32_to_bf16})
            for (const std::uint32_t width : {2560U,6144U}) for (const std::uint32_t outputs : {1U,7U,9U,17U}) {
                Fixture fixture(rows,width,outputs,direction);
                for (unsigned flavor = 0; flavor < 3U; ++flavor) {
                    fixture.prepare(flavor);
                    const auto expected = fixture.run(false);
                    for (unsigned repeat = 0; repeat < 3U; ++repeat)
                        require(fixture.run(true) == expected, "wide GDN complete projection payload differs bitwise from selected incumbent rows");
                    if (!flavor) require(std::any_of(expected.begin() + 1, expected.end(),
                        [&](const auto& row) { return row != expected.front(); }), "wide GDN finite row fixture is degenerate");
                }
                if (width == 2560U && outputs == 9U) {
                    fixture.prepare(0);
                    for (unsigned mutation = 0; mutation < 18U; ++mutation)
                        for (const bool capture : {false,true}) fixture.rejected(mutation,capture);
                    const auto expected = fixture.run(false);
                    require(fixture.run(true) == expected, "wide GDN cannot retry after pure metadata rejection");
                }
            }
        std::cout << "wide GDN projections R4/R8 complete bitwise fixture passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

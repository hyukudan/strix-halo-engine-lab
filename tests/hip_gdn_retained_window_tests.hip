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
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
using View = GdnKnownWindowProjectionView;
using Row = GdnRecurrentWindowRowView;
using Mode = GdnRetainedWindowMode;
constexpr std::uint32_t keys = 16, heads = 48, dim = 128;
constexpr std::size_t state_floats = static_cast<std::size_t>(heads) * dim * dim;
constexpr std::size_t qkv_floats = (2U * keys + heads) * dim;
constexpr std::size_t core_floats = heads * dim;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
std::vector<std::byte> read(View view) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(view.bytes));
    check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(view.address), bytes.size(), hipMemcpyDeviceToHost), "read retained recurrence bytes");
    return bytes;
}
void write(View view, const std::vector<std::byte>& bytes) {
    require(bytes.size() == view.bytes, "retained recurrence upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(view.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload retained recurrence bytes");
}
template<class T> std::vector<std::byte> bytes_of(const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size()); return bytes;
}
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create retained recurrence stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (const auto graph : graphs) if (hipGraphDestroy(graph) != hipSuccess) std::_Exit(3);
        for (auto* allocation : allocations) if (hipFree(allocation) != hipSuccess) std::_Exit(3);
        if (hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    View allocate(std::size_t bytes) {
        require(bytes && bytes <= (1U << 23U), "retained recurrence allocation exceeds bound");
        void* p = nullptr;
        check(hipMalloc(&p, bytes + 512U), "allocate retained recurrence owner");
        try { allocations.push_back(p); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipFree(p); throw; }
        check(hipMemset(p, 0xa5, bytes + 512U), "poison retained recurrence guards");
        return {reinterpret_cast<std::uintptr_t>(p) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "finish retained recurrence"); guard.require_original(); }
};
void guards(View view) {
    for (const auto address : {view.address - 256U, view.address + view.bytes}) {
        const auto bytes = read({address,256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == std::byte{0xa5}; }), "retained recurrence exterior guard changed");
    }
}
struct Snapshot {
    std::vector<std::byte> matrix;
    std::vector<std::vector<std::byte>> core;
    bool operator==(const Snapshot&) const = default;
};
struct Fixture {
    Owner owner;
    View matrix;
    std::vector<Row> rows;
    std::vector<std::array<std::vector<std::byte>,3>> readonly;
    std::vector<std::byte> initial;
    explicit Fixture(unsigned count) {
        matrix = owner.allocate(state_floats * sizeof(float));
        for (unsigned row = 0; row < count; ++row)
            rows.push_back({owner.allocate(qkv_floats * sizeof(float)), owner.allocate(heads * sizeof(float)),
                owner.allocate(heads * sizeof(float)), owner.allocate(core_floats * sizeof(float))});
    }
    void prepare(unsigned seed) {
        std::vector<float> state(state_floats);
        constexpr std::array<std::uint32_t,8> finite_edges{0,0x80000000U,1,0x80000001U,0x00800000U,0x80800000U,0x501502f9U,0xd01502f9U};
        for (std::size_t i = 0; i < state.size(); ++i) {
            const auto key = (i / dim) % dim, value = i % dim, head = i / (dim * dim);
            state[i] = seed == 2U ? std::bit_cast<float>(finite_edges[(key * 3U + value + head) % finite_edges.size()]) :
                seed == 1U ? (i % 2U ? -0.5F : 0.5F) + static_cast<float>(value % 4U) * 0.0009765625F :
                static_cast<float>(static_cast<int>((key * 5U + value * 7U + head * 3U) % 31U) - 15) * 0.015625F;
        }
        initial = bytes_of(state); readonly.clear();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            std::vector<float> qkv(qkv_floats), beta(heads), gate(heads);
            for (std::size_t i = 0; i < qkv.size(); ++i) {
                qkv[i] = seed == 1U ? (i % 2U ? -0.125F : 0.125F) * static_cast<float>(row + 1U) :
                    static_cast<float>(static_cast<int>((i * 7U + row * 11U) % 29U) - 14) * (seed == 2U ? 0.00000095367431640625F : 0.00390625F);
                if (seed && i % 17U == 0U) qkv[i] = std::bit_cast<float>(static_cast<std::uint32_t>((i + row) % 2U ? 0x80000000U : 0U));
                if (seed == 2U && i % 23U == 0U) qkv[i] = std::bit_cast<float>(static_cast<std::uint32_t>(row % 2U ? 0x80000001U : 1U));
            }
            for (std::size_t h = 0; h < heads; ++h) {
                beta[h] = seed == 2U ? ((h + row) % 2U ? 1.0F : 0.0F) :
                    0.125F * static_cast<float>(1U + (h + row) % 6U);
                gate[h] = seed == 2U ? ((h + row) % 3U ? -80.0F : -0.0F) :
                    -0.0625F * static_cast<float>((h + row * 3U) % 9U);
            }
            readonly.push_back({bytes_of(qkv),bytes_of(beta),bytes_of(gate)});
            write(rows[row].qkv,readonly.back()[0]); write(rows[row].beta,readonly.back()[1]); write(rows[row].gate,readonly.back()[2]);
        }
    }
    void unchanged() const {
        guards(matrix);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            require(read(rows[i].qkv) == readonly[i][0] && read(rows[i].beta) == readonly[i][1] && read(rows[i].gate) == readonly[i][2],
                "retained recurrence modified read-only row inputs");
            guards(rows[i].qkv); guards(rows[i].beta); guards(rows[i].gate); guards(rows[i].core);
        }
    }
    void poison() const {
        for (const auto& row : rows)
            check(hipMemset(reinterpret_cast<void*>(row.core.address), 0xa5, static_cast<std::size_t>(row.core.bytes)), "poison retained core output");
    }
    Snapshot snapshot() const {
        Snapshot result; result.matrix = read(matrix);
        for (const auto& row : rows) result.core.push_back(read(row.core));
        return result;
    }
    std::vector<Snapshot> run(Mode mode, bool swapped = false) {
        write(matrix, initial);
        auto selected = rows;
        if (swapped) std::swap(selected.front(),selected.back());
        std::vector<Snapshot> results;
        // State is deliberately retained between these calls: only a new arm
        // resets it. Each observation includes ALL tiles/heads and every core.
        for (unsigned step = 0; step < 3U; ++step) {
            poison();
            gdn_recurrent_window_probe(selected,matrix,keys,heads,dim,mode,owner.stream);
            owner.finish(); unchanged(); results.push_back(snapshot());
        }
        return results;
    }
    void rejected(unsigned mutation, bool capture) {
        write(matrix,initial); poison(); const auto before = snapshot();
        auto selected = rows; auto state = matrix;
        std::uint32_t k = keys, v = heads, d = dim;
        auto mode = Mode::retained_rows; void* stream = owner.stream;
        if (mutation == 0U) selected.pop_back();
        if (mutation == 1U) selected.back().qkv.address += 2U;
        if (mutation == 2U) --selected.back().qkv.bytes;
        if (mutation == 3U) --selected.back().beta.bytes;
        if (mutation == 4U) --selected.back().gate.bytes;
        if (mutation == 5U) --selected.back().core.bytes;
        if (mutation == 6U) --state.bytes;
        if (mutation == 7U) state.address += 2U;
        if (mutation == 8U) selected.back().core.address = selected.front().qkv.address;
        if (mutation == 9U) selected.back().core.address = state.address;
        if (mutation == 10U) selected.back().core = selected.front().core;
        if (mutation == 11U) selected.back().qkv.address = state.address;
        if (mutation == 12U) k = 8U;
        if (mutation == 13U) v = 32U;
        if (mutation == 14U) d = 64U;
        if (mutation == 15U) mode = static_cast<Mode>(999U);
        if (mutation == 16U) stream = nullptr;
        if (mutation == 17U) selected.back().gate.address = 0;
        bool invalid = false;
        if (capture) check(hipStreamBeginCapture(owner.stream,hipStreamCaptureModeThreadLocal), "begin rejected recurrence capture");
        try { gdn_recurrent_window_probe(selected,state,k,v,d,mode,stream); }
        catch (const std::invalid_argument&) { invalid = true; }
        catch (...) {
            if (capture) {
                hipGraph_t graph = nullptr; const auto status = hipStreamEndCapture(owner.stream,&graph);
                if (graph) owner.graphs.push_back(graph);
                if (status != hipSuccess && status != hipErrorStreamCaptureInvalidated) std::_Exit(3);
            }
            throw;
        }
        if (capture) {
            hipGraph_t graph = nullptr; check(hipStreamEndCapture(owner.stream,&graph), "end rejected recurrence capture");
            require(graph != nullptr, "rejected recurrence capture omitted graph"); owner.graphs.push_back(graph);
            std::size_t nodes = 0; check(hipGraphGetNodes(graph,nullptr,&nodes), "count rejected recurrence nodes");
            require(nodes == 0U, "rejected recurrence enqueued work before admission");
        }
        owner.finish(); require(invalid, "invalid retained recurrence metadata was accepted");
        require(snapshot() == before, "rejected recurrence changed a core or matrix byte"); unchanged();
    }
    void rejected_capture() {
        write(matrix,initial);poison();const auto before = snapshot();
        check(hipStreamBeginCapture(owner.stream,hipStreamCaptureModeThreadLocal), "begin unsupported recurrence capture");
        bool invalid = false;
        try { gdn_recurrent_window_probe(rows,matrix,keys,heads,dim,Mode::retained_rows,owner.stream); }
        catch (const std::invalid_argument&) { invalid = true; }
        catch (...) {
            hipGraph_t graph = nullptr; const auto status = hipStreamEndCapture(owner.stream,&graph);
            if (graph) owner.graphs.push_back(graph);
            if (status != hipSuccess && status != hipErrorStreamCaptureInvalidated) std::_Exit(3);
            throw;
        }
        hipGraph_t graph = nullptr; check(hipStreamEndCapture(owner.stream,&graph), "end unsupported recurrence capture");
        require(graph != nullptr,"unsupported recurrence capture omitted graph");owner.graphs.push_back(graph);
        std::size_t nodes = 0;check(hipGraphGetNodes(graph,nullptr,&nodes),"count unsupported recurrence nodes");
        require(invalid && nodes == 0U,"valid metadata recurrence capture did not reject before enqueue");
        owner.finish();require(snapshot() == before,"unsupported recurrence capture changed state");unchanged();
    }
};
} // namespace
int main() {
    try {
        for (const unsigned count : {2U,4U}) {
            Fixture fixture(count);
            for (unsigned seed = 0; seed < 3U; ++seed) {
                fixture.prepare(seed);
                const auto expected = fixture.run(Mode::serial_rows);
                for (unsigned repeat = 0; repeat < 3U; ++repeat)
                    require(fixture.run(Mode::retained_rows) == expected,"retained recurrence matrix or full row core differs bitwise from serial kernel");
                if (!seed) require(fixture.run(Mode::serial_rows,true).back().matrix != expected.back().matrix,
                    "finite recurrence fixture does not distinguish causal row order");
            }
            fixture.prepare(0);
            for (unsigned mutation = 0; mutation < 18U; ++mutation)
                for (const bool capture : {false,true}) fixture.rejected(mutation,capture);
            fixture.rejected_capture();
            const auto expected = fixture.run(Mode::serial_rows);
            require(fixture.run(Mode::retained_rows) == expected,"retained recurrence is not reusable after rejected preflight");
        }
        std::cout << "retained GDN recurrence R2/R4 full-state fixture passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

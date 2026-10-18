#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_plan.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_shared_expert_launcher.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
struct View { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
std::vector<std::byte> read(View view) {
    std::vector<std::byte> bytes(static_cast<std::size_t>(view.bytes));
    check(hipMemcpy(bytes.data(), reinterpret_cast<const void*>(view.address), bytes.size(), hipMemcpyDeviceToHost), "read guarded rows view");
    return bytes;
}
void write(View view, const std::vector<std::byte>& bytes) {
    require(view.bytes == bytes.size(), "rows upload extent differs");
    check(hipMemcpy(reinterpret_cast<void*>(view.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "upload guarded rows view");
}
template<class T> std::vector<std::byte> bytes_of(const std::vector<T>& values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size()); return bytes;
}
struct Owner {
    HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create shared rows stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        for (auto* p : allocations) if (hipFree(p) != hipSuccess) std::_Exit(3);
        if (hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    View allocate(std::uint64_t bytes) {
        require(bytes && bytes < (1ULL << 28U), "rows fixture allocation exceeds bound");
        void* pointer = nullptr;
        check(hipMalloc(&pointer, static_cast<std::size_t>(bytes + 512U)), "allocate shared rows owner");
        try { allocations.push_back(pointer); }
        catch (...) { guard.recover_or_fail_stop(); static_cast<void>(hipFree(pointer)); throw; }
        check(hipMemset(pointer, 0xa5, static_cast<std::size_t>(bytes + 512U)), "poison shared rows owner");
        return {reinterpret_cast<std::uintptr_t>(pointer) + 256U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "drain shared rows stream"); guard.require_original(); }
};
void guards(View view) {
    for (const auto address : {view.address - 256U, view.address + view.bytes}) {
        const auto bytes = read({address, 256U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == std::byte{0xa5}; }), "rows outer guard changed");
    }
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
struct Q8 { __half scale; std::int8_t codes[32]; };
static_assert(sizeof(Q8) == 34);
std::vector<std::byte> q8(unsigned width, unsigned outputs, unsigned salt) {
    std::vector<Q8> blocks(static_cast<std::size_t>(width / 32U) * outputs);
    constexpr std::array<std::uint16_t, 7> scale_bits{0x0000,0x8000,0x0001,0x0400,0x1800,0x1c00,0x2000};
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const auto bits = scale_bits[(b + salt) % scale_bits.size()];
        std::memcpy(&blocks[b].scale, &bits, sizeof(bits));
        for (unsigned c = 0; c < 32; ++c)
            blocks[b].codes[c] = static_cast<std::int8_t>(static_cast<int>((b * 17U + c * 31U + salt) % 256U) - 128);
    }
    return bytes_of(blocks);
}
struct Fixture {
    Owner owner; // outlives every borrowing descriptor and outstanding enqueue
    ModelContract model = ModelContract::qwen38_flash_next();
    SerialExecutionPlan execution = build_serial_decode_execution_plan(model);
    SerialExecutionIoPlan io = build_serial_execution_io_plan(execution);
    ExecutionStep step;
    ExecutionArenaPlan serial;
    KnownWindowArena window;
    View scratch;
    ExecutionMemoryBinding memory;
    ModelManifest manifest;
    ModelRuntimeBinding weights;
    ExecutionTensorAccessPlan access;
    std::vector<View> weight_views;
    std::vector<std::vector<std::byte>> original_weights;
    std::vector<ExecutionRuntimeResources> resources;
    std::vector<ExecutionDispatchContext> contexts;
    std::vector<ExecutionBufferAddress> input, output, temporary;
    HipSharedExpertLauncherConfig config;
    std::string serial_id;
    ExecutionLauncher incumbent;
    std::vector<std::byte> seed;

    Fixture(unsigned rows, bool prefetch) {
        require(owner.guard.architecture() == "gfx1151", "shared rows fixture requires actual gfx1151");
        const auto found = std::find_if(execution.steps.begin(), execution.steps.end(),
            [](const auto& value) { return value.op == SerialOp::shared_expert && value.layer == 0U; });
        require(found != execution.steps.end(), "shared canonical step missing"); step = *found;
        const auto& row_io = io.steps[step.id];
        serial.alignment = 256;
        const auto allocation = [&](ExecutionBufferUse use, std::uint64_t bytes) {
            const auto offset = (serial.total_bytes + 255U) / 256U * 256U;
            serial.allocations.push_back({use, step.id, step.id, offset, bytes}); serial.total_bytes = offset + bytes;
        };
        allocation(row_io.reads[0], 5120); allocation(row_io.writes[0], 5120); allocation(row_io.temporaries[0], 5120);
        serial.naive_nonaliased_bytes = serial.total_bytes;
        window = build_known_window_arena(serial, 256U, rows);
        scratch = owner.allocate(window.physical.total_bytes);
        memory = {0x5253484152454401ULL, scratch.address, scratch.bytes, 23};
        manifest.architecture = "qwen4_exp";
        add("blk.0.shared_expert.gate_proj.weight", 8, {2560,640}, q8(2560,640,1));
        add("blk.0.shared_expert.up_proj.weight", 8, {2560,640}, q8(2560,640,2));
        add("blk.0.shared_expert.down_proj.weight", 8, {640,2560}, q8(640,2560,3));
        std::vector<float> scalar(2560);
        for (std::size_t i = 0; i < scalar.size(); ++i) scalar[i] = static_cast<float>(static_cast<int>(i % 7U) - 3) * 0.0001F;
        add("blk.0.shared_expert_gate.weight", 0, {2560}, bytes_of(scalar));
        manifest.layout_identity = model_manifest_layout_identity(manifest);
        weights.manifest_layout_identity = manifest.layout_identity; weights.model_content_identity.bytes[0] = 1;
        access.manifest_layout_identity = manifest.layout_identity; access.plan_identity = 0x5253484152454402ULL;
        access.accesses.push_back({step.id, step.op, 0, {0,1,2,3}});
        config.hidden_size = 2560; config.intermediate_size = 640; config.prefetch_q8 = prefetch;
        if (prefetch) config.prefetch_eligible_layers = {0};
        kernels::KernelRegistry registry;
        ExecutionBindingRequest request;
        request.prefetch_q8_shared_expert = prefetch; request.shared_expert_prefetch_layers = config.prefetch_eligible_layers;
        register_hip_shared_expert_kernel_variant(model, request, config, registry);
        serial_id = registry.variants().front().id; incumbent = make_hip_shared_expert_launcher(serial_id, &config);
        contexts.resize(rows); resources.resize(rows); input.resize(rows); output.resize(rows); temporary.resize(rows);
        for (unsigned r = 0; r < rows; ++r) {
            resources[r] = {&memory, &manifest, &weights, &access};
            auto& c = contexts[r];
            c.execution_identity = memory.execution_identity; c.step_id = step.id; c.op = step.op; c.layer = step.layer;
            c.kernel_id = serial_id; c.reads = row_io.reads; c.writes = row_io.writes; c.temporaries = row_io.temporaries;
            c.arena = &window.row_view[r]; c.memory = &memory; c.resources = &resources[r];
            c.position = 17U + r; c.input_token_id = 101U + r;
            c.reads_state = step.reads_state; c.writes_state = step.writes_state;
            input[r] = resolve_dispatch_read(c, 0); output[r] = resolve_dispatch_write(c, 0); temporary[r] = resolve_dispatch_temporary(c, 0);
        }
    }
    void add(std::string name, unsigned type, std::vector<std::uint64_t> dims, std::vector<std::byte> data) {
        ManifestTensor t; t.name = std::move(name); t.layer = 0; t.role = TensorRole::dense_projection;
        t.ggml_type = type; t.dimensions = std::move(dims); t.absolute_offset = manifest.total_tensor_bytes;
        t.byte_size = data.size(); manifest.total_tensor_bytes += t.byte_size;
        const auto device = owner.allocate(data.size()); write(device, data);
        const auto index = static_cast<std::uint32_t>(weight_views.size());
        weight_views.push_back(device); original_weights.push_back(std::move(data)); manifest.tensors.push_back(std::move(t));
        weights.tensors.push_back({index, RuntimeTensorStorage::resident, device.address, device.bytes, 31, 0});
    }
    HipLaunchContext scope() const { HipLaunchContext s; s.stream = owner.stream; return s; }
    void prepare(unsigned kind, bool reverse) {
        seed.assign(static_cast<std::size_t>(scratch.bytes), std::byte{0xa5});
        constexpr std::array<std::uint16_t, 12> special{0,0x8000,1,0x8001,0x007f,0x807f,0x3f80,0xbf80,0x7f80,0xff80,0x7fc1,0xff83};
        for (std::size_t r = 0; r < contexts.size(); ++r) {
            const auto tag = reverse ? contexts.size() - 1U - r : r;
            std::vector<std::uint16_t> values(2560);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = kind ? special[(i + tag * 5U) % special.size()] :
                    bf16(static_cast<float>(static_cast<int>((i * 29U + tag * 43U) % 97U) - 48) / static_cast<float>(97U + tag));
            std::memcpy(seed.data() + input[r].address - scratch.address, values.data(), input[r].bytes);
        }
    }
    std::vector<std::byte> run(bool wide) {
        write(scratch, seed);
        const auto current = scope();
        { const HipLaunchScope active(current);
          if (wide) {
              shared_expert_known_window_rows_preflight(contexts, config);
              require(read(scratch) == seed, "pure shared rows preflight changed payload");
              shared_expert_known_window_rows_launch(contexts, config);
          } else for (const auto& c : contexts) { incumbent.preflight(c, incumbent.user_data); incumbent.function(c, incumbent.user_data); } }
        owner.finish();
        const auto result = read(scratch);
        auto masked = result;
        for (std::size_t r = 0; r < contexts.size(); ++r)
            for (const auto view : {output[r], temporary[r]})
                std::copy_n(seed.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address),
                    static_cast<std::size_t>(view.bytes), masked.begin() + static_cast<std::ptrdiff_t>(view.address - scratch.address));
        require(masked == seed, "shared rows changed input or internal guard bytes");
        guards(scratch);
        for (std::size_t w = 0; w < weight_views.size(); ++w) {
            require(read(weight_views[w]) == original_weights[w], "shared rows changed a weight owner"); guards(weight_views[w]);
        }
        return result;
    }
    // The window owner's detached form: a second stream and a private workspace at the arena's row
    // stride. Returns the arena run's bytes with the workspace rows read from the private copy; the
    // arena workspace keeps its seed.
    std::vector<std::byte> run_detached() {
        write(scratch, seed);
        const auto stride = temporary[1].address - temporary[0].address;
        const auto workspace = owner.allocate(stride * (contexts.size() - 1U) + temporary.back().bytes);
        hipStream_t side = nullptr;
        check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create detached shared stream");
        try {
            shared_expert_known_window_rows_launch_detached(contexts, config, side, workspace.address, workspace.bytes);
            check(hipStreamSynchronize(side), "complete detached shared rows");
        } catch (...) { static_cast<void>(hipStreamSynchronize(side)); static_cast<void>(hipStreamDestroy(side)); throw; }
        check(hipStreamDestroy(side), "destroy detached shared stream");
        auto result = read(scratch);
        for (std::size_t r = 0; r < contexts.size(); ++r) {
            const auto at = static_cast<std::ptrdiff_t>(temporary[r].address - scratch.address);
            require(std::equal(seed.begin() + at, seed.begin() + at + static_cast<std::ptrdiff_t>(temporary[r].bytes), result.begin() + at),
                "detached shared rows wrote the arena workspace");
            const auto copied = read({workspace.address + r * stride, temporary[r].bytes});
            std::copy(copied.begin(), copied.end(), result.begin() + at);
        }
        guards(scratch); guards(workspace);
        return result;
    }
    template<class Mutate, class Restore> void rejected(Mutate mutate, Restore restore, bool capture,
        std::string_view expected, bool expect_overflow = false) {
        write(scratch, seed); const auto prior_weights = original_weights;
        auto current = scope(); hipGraph_t graph = nullptr;
        bool rejected = false; std::string reason;
        mutate();
        { const HipLaunchScope active(current);
          if (capture) check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin wide shared rejection capture");
          try { shared_expert_known_window_rows_launch(contexts, config); }
          catch (const std::invalid_argument& error) { rejected = !expect_overflow; reason = error.what(); }
          catch (const std::overflow_error& error) { rejected = expect_overflow; reason = error.what(); }
          if (capture) check(hipStreamEndCapture(owner.stream, &graph), "end wide shared rejection capture"); }
        restore();
        if (graph) {
            std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "count wide shared rejected nodes");
            require(nodes == 0, "rejected wide shared admission captured nodes");
            check(hipGraphDestroy(graph), "destroy wide shared rejected graph");
        }
        owner.finish();
        const bool unchanged = read(scratch) == seed;
        if (!rejected || reason.find(expected) == std::string::npos || !unchanged)
            throw std::runtime_error("wide shared rejection expected [" + std::string(expected) +
                "] but received [" + reason + "]; rejected=" + std::to_string(rejected) +
                "; payload_unchanged=" + std::to_string(unchanged));
        for (std::size_t w = 0; w < weight_views.size(); ++w)
            require(read(weight_views[w]) == prior_weights[w], "wide shared rejection mutated weights");
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2 && argc <= 3, "usage: shared-known-window-rows ROWS [PREFETCH0OR1]");
        const auto rows = static_cast<unsigned>(std::stoul(argv[1]));
        require(rows == 4 || rows == 8, "fixture requires R4 or R8");
        require(argc == 2 || std::string_view(argv[2]) == "0" || std::string_view(argv[2]) == "1",
            "fixture prefetch argument must be zero or one");
        const bool prefetch = argc == 3 && std::string_view(argv[2]) == "1";
        Fixture f(rows, prefetch);
        for (unsigned kind = 0; kind < 2; ++kind) for (const bool reverse : {false, true}) {
            f.prepare(kind, reverse); const auto reference = f.run(false);
            for (unsigned repeat = 0; repeat < 3; ++repeat)
                require(f.run(true) == reference, "wide shared result differs from complete incumbent arena bits");
            require(f.run_detached() == reference, "detached wide shared result differs from complete incumbent arena bits");
        }
        f.prepare(0, false);
        for (const bool capture : {false, true}) {
            const auto original = f.contexts;
            f.rejected([&] { ++f.contexts.back().position; }, [&] { f.contexts = original; }, capture, "R2 row identity/position differs");
            f.rejected([&] { f.contexts.back().execution_identity ^= 1; }, [&] { f.contexts = original; }, capture, "R2 row identity/position differs");
            f.rejected([&] { f.contexts.back().temporaries = {}; }, [&] { f.contexts = original; }, capture, "R2 requires matching C1 row IO");
            auto copied_memory = f.memory;
            f.rejected([&] { f.contexts.back().memory = &copied_memory; }, [&] { f.contexts = original; }, capture, "R2 scratch aliases another memory binding");
            const auto offset = f.window.row_view.back().allocations[2].offset;
            f.rejected([&] { f.window.row_view.back().allocations[2].offset = f.window.row_view.front().allocations[2].offset; },
                [&] { f.window.row_view.back().allocations[2].offset = offset; }, capture, "R2 row extent/order differs");
            auto copied_weights = f.weights;
            f.rejected([&] { f.resources.back().model_weights = &copied_weights; }, [&] { f.resources.back().model_weights = &f.weights; }, capture, "R2 weight/access owners differ");
            const auto type = f.manifest.tensors[0].ggml_type;
            f.rejected([&] { f.manifest.tensors[0].ggml_type = 20; }, [&] { f.manifest.tensors[0].ggml_type = type; }, capture, "R2 tensor type/layer/owner differs");
            f.rejected([&] { f.contexts[rows - 2U].position += 2; f.contexts.back().position += 2; },
                [&] { f.contexts = original; }, capture, "rows identity/position/IO differs");
            f.rejected([&] { f.contexts.front().position = UINT64_MAX; }, [&] { f.contexts = original; },
                capture, "rows positions overflow");
            f.rejected([&] { f.resources[rows - 2U].model_weights = &copied_weights; f.resources.back().model_weights = &copied_weights; },
                [&] { f.resources[rows - 2U].model_weights = &f.weights; f.resources.back().model_weights = &f.weights; },
                capture, "rows retained resource owners differ");
            const auto row_views = f.window.row_view;
            f.rejected([&] {
                const auto stride = f.input[1].address - f.input[0].address;
                for (unsigned r = 0; r < rows; ++r)
                    f.window.row_view[r].allocations[1].offset = f.window.row_view[r].allocations[0].offset + 2U * stride;
            }, [&] { f.window.row_view = row_views;
                for (unsigned r = 0; r < rows; ++r) f.contexts[r].arena = &f.window.row_view[r]; },
                capture, "rows cross-row writable/input alias");
            f.rejected([&] {
                for (unsigned r = 0; r < rows; ++r) f.window.row_view[r].allocations[0].offset += f.scratch.bytes + 256U;
            }, [&] { f.window.row_view = row_views;
                for (unsigned r = 0; r < rows; ++r) f.contexts[r].arena = &f.window.row_view[r]; },
                capture, "execution buffer exceeds bound arena allocation");
            const auto arena_bytes = f.memory.arena_bytes;
            f.rejected([&] { f.memory.arena_bytes = UINT64_MAX; }, [&] { f.memory.arena_bytes = arena_bytes; },
                capture, "execution arena address range overflows uintptr_t", true);
            f.rejected([&] { f.contexts.resize(3); }, [&] { f.contexts = original; }, capture, "rows count must be R4 or R8");
        }
        f.rejected([] {}, [] {}, true, "rows cannot run during capture"); // Valid metadata still cannot capture.
        const auto reference = f.run(false);
        require(f.run(true) == reference, "wide shared retry after rejected metadata is not reusable");
        std::cout << "shared known-window rows source fixture R" << rows << " passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

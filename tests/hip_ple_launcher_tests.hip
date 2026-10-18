#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            hipGetErrorString(error));
    }
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits +=
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        bits >> 16U);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(
            value) << 16U;
    float output = 0.0F;
    std::memcpy(
        &output,
        &bits,
        sizeof(output));
    return output;
}

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

static_assert(sizeof(Q8Block) == 34);

class FixtureLookup final
    : public strix::PleLookupBackend {
public:
    explicit FixtureLookup(
        std::vector<std::uint16_t> gathered)
        : gathered_(std::move(gathered)) {}

    std::uint64_t implementation_identity()
        const noexcept override {
        return 0x504c455445535401ULL;
    }

    strix::PleLookupHandle start(
        const strix::PleLookupRequest& request) override {
        if (active_) {
            throw std::logic_error(
                "fixture lookup already active");
        }
        expect(
            request.row_ids.size() == 16 &&
                request.row_bytes == 320,
            "PLE launcher requests canonical 16x320-byte rows");
        active_ = true;
        return 1;
    }

    void finish(
        strix::PleLookupHandle handle,
        std::span<std::byte> destination) override {
        if (!active_ || handle != 1) {
            throw std::invalid_argument(
                "fixture lookup handle mismatch");
        }
        const auto bytes =
            gathered_.size() *
            sizeof(std::uint16_t);
        if (destination.size() < bytes) {
            throw std::invalid_argument(
                "fixture lookup destination undersized");
        }
        std::memcpy(
            destination.data(),
            gathered_.data(),
            bytes);
        active_ = false;
    }

    void cancel(
        strix::PleLookupHandle handle) noexcept override {
        if (handle == 1) {
            active_ = false;
        }
    }

private:
    std::vector<std::uint16_t> gathered_;
    bool active_ = false;
};

const strix::ExecutionStep& ple_step(
    const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) {
        if (step.op ==
            strix::SerialOp::ple) {
            return step;
        }
    }
    throw std::runtime_error(
        "serial plan has no PLE step");
}

std::vector<Q8Block> make_q8(
    std::uint32_t rows,
    std::uint32_t width,
    bool zero_values) {
    const auto blocks_per_row =
        width / 32U;
    std::vector<Q8Block> blocks(
        static_cast<std::size_t>(rows) *
        blocks_per_row);

    for (std::uint32_t row = 0;
         row < rows;
         ++row) {
        const float scale =
            zero_values
                ? 0.001F
                : 0.0001F *
                      static_cast<float>(
                          row % 7U + 1U);
        for (std::uint32_t block = 0;
             block < blocks_per_row;
             ++block) {
            auto& value =
                blocks[
                    static_cast<std::size_t>(row) *
                        blocks_per_row +
                    block];
            value.scale =
                __float2half(scale);
            std::fill(
                std::begin(value.values),
                std::end(value.values),
                static_cast<std::int8_t>(
                    zero_values ? 0 : 1));
        }
    }
    return blocks;
}

template <typename T>
void upload(
    const std::vector<T>& values,
    void** device) {
    check(
        hipMalloc(
            device,
            values.size() *
                sizeof(T)),
        "hipMalloc PLE test weight");
    check(
        hipMemcpy(
            *device,
            values.data(),
            values.size() *
                sizeof(T),
            hipMemcpyHostToDevice),
        "copy PLE test weight");
}

// Uses the fixture's real device owners, but a retained small host table and
// the actual synchronous backend. This exercises only factory.preflight.
void pure_preflight_cases(const strix::ExecutionDispatchContext& source,
    strix::HipPleLauncherConfig config) {
    config.model.ngram_vocab_size_base = 17;
    config.upload_pool = nullptr;
    config.enable_owned_event_upload = false;
    config.enable_resident_graph = false;
    auto contract = strix::build_ple_hash_contract(config.model);
    auto manifest = *source.resources->model_manifest;
    auto runtime = *source.resources->model_weights;
    auto access = *source.resources->tensor_access;
    auto resources = *source.resources;
    auto context = source;
    strix::FileMappedPleLookupBackend backend;
    strix::PleLookupPrefetchState pending;
    // Nonfinite payloads are deliberate: metadata admission must never inspect
    // table/activation/history values or produce the future row's hash.
    std::vector<std::uint16_t> table(
        static_cast<std::size_t>(contract.minimum_table_rows * contract.head_dim), 0x7fc1U);
    manifest.tensors[0].dimensions = {contract.head_dim, contract.minimum_table_rows};
    manifest.tensors[0].ggml_type = 30;
    manifest.tensors[0].byte_size = table.size() * sizeof(std::uint16_t);
    runtime.tensors[0].storage = strix::RuntimeTensorStorage::file_mapped;
    runtime.tensors[0].address = reinterpret_cast<std::uintptr_t>(table.data());
    runtime.tensors[0].bytes = manifest.tensors[0].byte_size;
    const auto rebind_manifest = [&] {
        manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
        runtime.manifest_layout_identity = manifest.layout_identity;
        access.manifest_layout_identity = manifest.layout_identity;
    };
    rebind_manifest();
    resources.model_manifest = &manifest;
    resources.model_weights = &runtime;
    resources.tensor_access = &access;
    resources.ple_lookup = &backend;
    resources.ple_hash_contract = &contract;
    resources.ple_prefetch = &pending;
    context.resources = &resources;
    auto launcher = strix::make_hip_ple_launcher("test.ple.pure-preflight", &config);
    expect(launcher.preflight != nullptr, "PLE factory publishes real metadata preflight");
    if (!launcher.preflight) return;

    const auto& scratch = *resources.scratch;
    const auto& state = *resources.session_state_memory;
    check(hipMemset(reinterpret_cast<void*>(scratch.arena_base), 0xa7,
        static_cast<std::size_t>(scratch.arena_bytes)), "poison complete PLE preflight arena");
    check(hipMemset(reinterpret_cast<void*>(state.device_base), 0x6d,
        static_cast<std::size_t>(state.device_bytes)), "poison complete PLE preflight device state");
    std::memset(reinterpret_cast<void*>(state.host_base), 0xd9,
        static_cast<std::size_t>(state.host_bytes));
    std::vector<std::byte> arena_before(static_cast<std::size_t>(scratch.arena_bytes));
    std::vector<std::byte> device_before(static_cast<std::size_t>(state.device_bytes));
    std::vector<std::byte> host_before(static_cast<std::size_t>(state.host_bytes));
    check(hipMemcpy(arena_before.data(), reinterpret_cast<const void*>(scratch.arena_base),
        arena_before.size(), hipMemcpyDeviceToHost), "snapshot full PLE preflight arena");
    check(hipMemcpy(device_before.data(), reinterpret_cast<const void*>(state.device_base),
        device_before.size(), hipMemcpyDeviceToHost), "snapshot full PLE preflight device state");
    std::memcpy(host_before.data(), reinterpret_cast<const void*>(state.host_base), host_before.size());
    const auto table_before = table;
    const auto unchanged = [&] {
        std::vector<std::byte> actual_arena(arena_before.size()), actual_device(device_before.size());
        check(hipMemcpy(actual_arena.data(), reinterpret_cast<const void*>(scratch.arena_base),
            actual_arena.size(), hipMemcpyDeviceToHost), "read complete PLE admission arena");
        check(hipMemcpy(actual_device.data(), reinterpret_cast<const void*>(state.device_base),
            actual_device.size(), hipMemcpyDeviceToHost), "read complete PLE admission state");
        expect(actual_arena == arena_before, "PLE preflight preserves every scratch byte including padding");
        expect(actual_device == device_before, "PLE preflight preserves every device-state byte");
        expect(std::memcmp(host_before.data(), reinterpret_cast<const void*>(state.host_base),
            host_before.size()) == 0, "PLE preflight preserves every host-history byte");
        expect(table == table_before, "PLE preflight preserves retained table payload");
        expect(!pending.active && pending.handle == 0 && pending.backend_identity == 0 &&
            pending.lookup_bytes == 0 && pending.input_token_id == 0 && pending.position == 0 &&
            pending.hash.row_count == 0 && pending.hash.context_count == 0 &&
            std::all_of(pending.hash.row_ids.begin(), pending.hash.row_ids.end(), [](auto v) { return v == 0; }) &&
            std::all_of(pending.hash.next_context.begin(), pending.hash.next_context.end(), [](auto v) { return v == 0; }),
            "PLE preflight does not start, finish or publish pending lookup state");
    };
    launcher.preflight(context, launcher.user_data);
    unchanged();
    const auto saved_table = manifest.tensors[0];
    const auto saved_binding = runtime.tensors[0];
    const auto saved_contract = contract;
    // The same retained allocation can expose an exact smaller packed extent;
    // no quantized payload is decoded by metadata-only admission.
    manifest.tensors[0].ggml_type = 20;
    manifest.tensors[0].byte_size = contract.minimum_table_rows *
        (contract.head_dim / strix::iq4_nl::block_elements) * strix::iq4_nl::block_bytes;
    runtime.tensors[0].bytes = manifest.tensors[0].byte_size;
    rebind_manifest();
    launcher.preflight(context, launcher.user_data);
    unchanged();
    manifest.tensors[0] = saved_table;
    runtime.tensors[0] = saved_binding;
    rebind_manifest();
    const auto reject = [&](const char* expected_message) {
        bool rejected = false, intended = false;
        try { launcher.preflight(context, launcher.user_data); }
        catch (const std::invalid_argument& error) {
            rejected = true; intended = error.what() == std::string(expected_message);
        }
        expect(rejected, "PLE malformed metadata rejects before lookup/upload");
        expect(intended, "PLE rejection reaches the intended pure admission guard");
        unchanged();
        manifest.tensors[0] = saved_table;
        runtime.tensors[0] = saved_binding;
        contract = saved_contract;
        rebind_manifest();
    };
    constexpr auto metadata_error = "known window PLE table metadata or mapping is incompatible";
    constexpr auto extent_error = "known window PLE declared table rows exceed the retained mapping";
    constexpr auto hash_error = "known window PLE prefetch hash contract differs from the selected launcher";
    manifest.tensors[0].dimensions.push_back(1); rebind_manifest(); reject(metadata_error);
    manifest.tensors[0].ggml_type = 0; rebind_manifest(); reject(metadata_error);
    manifest.tensors[0].dimensions[1] = contract.minimum_table_rows - 1U;
    rebind_manifest(); reject(metadata_error);
    runtime.tensors[0].bytes = std::numeric_limits<std::uint64_t>::max(); reject(metadata_error);
    manifest.tensors[0].dimensions[1] = std::numeric_limits<std::uint64_t>::max();
    rebind_manifest(); reject(extent_error);
    --manifest.tensors[0].byte_size; rebind_manifest(); reject(extent_error);
    contract.ngram_size = 0; reject(hash_error);
    contract.ngram_size = strix::PleHashContract::maximum_ngram + 1U; reject(hash_error);
    contract.multipliers[0] ^= 2U; reject(hash_error);
    contract.row_bytes += 2U; reject(hash_error);
    contract.head_offsets[0] += 1U; reject(hash_error);
    contract.head_vocab_sizes[0] += 1U; reject(hash_error);

    hipStream_t stream = nullptr;
    void* arguments = nullptr; void* status = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create PLE preflight capture stream");
    const auto graph_layout = strix::graph_input_layout(1);
    check(hipMalloc(&arguments, static_cast<std::size_t>(graph_layout.bytes)), "allocate PLE capture arguments");
    check(hipMalloc(&status, sizeof(std::uint32_t)), "allocate PLE capture status");
    strix::ExecutionMemoryBinding graph_memory;
    graph_memory.execution_identity = scratch.execution_identity;
    graph_memory.arena_base = reinterpret_cast<std::uintptr_t>(arguments);
    graph_memory.arena_bytes = graph_layout.bytes; graph_memory.allocation_generation = 1;
    strix::HipLaunchContext scope;
    scope.stream = stream; scope.graph_capture = true;
    scope.arguments = strix::bind_graph_inputs(graph_layout, graph_memory);
    scope.status = graph_memory;
    scope.status.arena_base = reinterpret_cast<std::uintptr_t>(status);
    scope.status.arena_bytes = sizeof(std::uint32_t);
    scope.context_tokens = resources.session_state_layout->context_tokens;
    scope.vocabulary_size = static_cast<std::uint32_t>(config.model.vocab_size);
    hipGraph_t graph = nullptr;
    bool capture_rejected = false;
    {
        const strix::HipLaunchScope owned(scope);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin pure PLE rejected capture");
        try { launcher.preflight(context, launcher.user_data); }
        catch (const std::invalid_argument& error) {
            capture_rejected = std::string(error.what()) == "known window PLE preflight is eager only";
        }
        check(hipStreamEndCapture(stream, &graph), "end pure PLE rejected capture");
    }
    std::size_t nodes = 0;
    check(hipGraphGetNodes(graph, nullptr, &nodes), "count pure PLE rejection graph nodes");
    expect(capture_rejected && nodes == 0, "PLE pure capture rejection has zero enqueued nodes");
    check(hipGraphDestroy(graph), "destroy pure PLE rejection graph");
    check(hipFree(status), "free PLE capture status");
    check(hipFree(arguments), "free PLE capture arguments");
    check(hipStreamDestroy(stream), "destroy pure PLE capture stream");
    unchanged();

    // FileMappedPleLookupBackend is final. Its first real handle proves that
    // none of the preceding callbacks started even a subsequently cancelled
    // lookup. This request reads no table payload and performs no upload.
    const std::array<std::uint64_t, 1> row{0};
    strix::PleLookupRequest lookup;
    lookup.row_ids = row; lookup.row_bytes = contract.row_bytes;
    lookup.table_manifest = &manifest.tensors[0]; lookup.table_runtime = &runtime.tensors[0];
    const auto first_handle = backend.start(lookup);
    expect(first_handle == 1, "all PLE pure preflights leave the real backend completely unused");
    backend.cancel(first_handle);
    unchanged();
}

// PLE tail of prefill rows: the rows kernels must reproduce the per-row kernels bit for bit (every
// output row and the convolution state) across blocks shorter than, equal to and longer than the
// convolution history, chained blocks on one state, in-place outputs and F32/BF16 weights.
template <class T> struct TailBuffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit TailBuffer(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "hipMalloc PLE tail buffer");
        check(hipMemset(p, 0, count * sizeof(T)), "hipMemset PLE tail buffer");
    }
    ~TailBuffer() { if (p) static_cast<void>(hipFree(p)); }
    TailBuffer(const TailBuffer&) = delete;
    TailBuffer& operator=(const TailBuffer&) = delete;
    void put(const std::vector<T>& v) {
        if (v.size() > n) throw std::invalid_argument("PLE tail upload extent");
        check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload PLE tail buffer");
    }
    std::vector<T> read() const {
        std::vector<T> v(n);
        check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "read PLE tail buffer");
        return v;
    }
};

void ple_tail_rows_cases() {
    constexpr std::uint32_t hidden = 2560, streams = 4, taps = 4, ngram = 3, expanded = hidden * streams,
        history = (taps - 1U) * ngram, max_rows = 300;
    std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
    const auto uniform = [&] {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<std::uint32_t>(seed >> 40U)) / 16777216.0F * 2.0F - 1.0F;
    };
    const auto f32 = [&](std::size_t count, float scale) {
        std::vector<float> v(count);
        for (auto& e : v) e = uniform() * scale;
        return v;
    };
    const auto bf16 = [&](std::size_t count, float scale) {
        std::vector<std::uint16_t> v(count);
        for (auto& e : v) e = float_to_bf16(uniform() * scale);
        return v;
    };
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create PLE tail stream");
    for (const bool bf16_weights : {false, true}) {
        for (const bool in_place : {false, true}) {
            TailBuffer<std::uint16_t> nk_b(expanded), nq_b(expanded), nc_b(expanded), cv_b(expanded * taps);
            TailBuffer<float> nk_f(expanded), nq_f(expanded), nc_f(expanded), cv_f(expanded * taps);
            nk_b.put(bf16(expanded, 1.5F)); nq_b.put(bf16(expanded, 1.5F)); nc_b.put(bf16(expanded, 1.5F));
            cv_b.put(bf16(expanded * taps, 1.2F));
            nk_f.put(f32(expanded, 1.5F)); nq_f.put(f32(expanded, 1.5F)); nc_f.put(f32(expanded, 1.5F));
            cv_f.put(f32(expanded * taps, 1.2F));
            const void* nk = bf16_weights ? static_cast<const void*>(nk_b.p) : nk_f.p;
            const void* nq = bf16_weights ? static_cast<const void*>(nq_b.p) : nq_f.p;
            const void* nc = bf16_weights ? static_cast<const void*>(nc_b.p) : nc_f.p;
            const void* cv = bf16_weights ? static_cast<const void*>(cv_b.p) : cv_f.p;
            const std::uint32_t in_stride = expanded + 64U, out_stride = in_place ? in_stride : expanded + 32U;
            TailBuffer<float> keys_a(std::size_t{max_rows} * expanded), keys_b(keys_a.n), values(std::size_t{max_rows} * hidden);
            TailBuffer<float> scratch_a(std::size_t{max_rows} * streams + 2U * expanded), scratch_b(scratch_a.n);
            TailBuffer<std::uint16_t> in_a(std::size_t{max_rows} * in_stride), in_b(in_a.n);
            TailBuffer<std::uint16_t> out_a(std::size_t{max_rows} * out_stride), out_b(out_a.n);
            TailBuffer<std::uint16_t> state_a(std::size_t{expanded} * history), state_b(state_a.n);
            const auto state = bf16(state_a.n, 2.0F);
            state_a.put(state); state_b.put(state);
            bool same = true;
            std::size_t compared = 0;
            // Blocks of 1, 2, 9 (the history), 10 and 300 rows chained on the same state.
            for (const std::uint32_t rows : {1U, 2U, 9U, 10U, 300U}) {
                const auto keys = f32(std::size_t{rows} * expanded, 3.0F);
                const auto input = bf16(std::size_t{rows} * in_stride, 4.0F);
                keys_a.put(keys); keys_b.put(keys); values.put(f32(std::size_t{rows} * hidden, 3.0F));
                in_a.put(input); in_b.put(input);
                auto* oa = in_place ? in_a.p : out_a.p;
                auto* ob = in_place ? in_b.p : out_b.p;
                strix::testing::enqueue_ple_tail(false, keys_a.p, values.p, in_a.p, in_stride, oa, out_stride, state_a.p, nk, nq, nc,
                    cv, bf16_weights, rows, hidden, streams, taps, ngram, 1e-6F, scratch_a.p, stream);
                strix::testing::enqueue_ple_tail(true, keys_b.p, values.p, in_b.p, in_stride, ob, out_stride, state_b.p, nk, nq, nc,
                    cv, bf16_weights, rows, hidden, streams, taps, ngram, 1e-6F, scratch_b.p, stream);
                check(hipStreamSynchronize(stream), "PLE tail synchronize");
                const auto ra = in_place ? in_a.read() : out_a.read();
                const auto rb = in_place ? in_b.read() : out_b.read();
                for (std::uint32_t row = 0; row < rows; ++row)
                    same = same && std::memcmp(ra.data() + std::size_t{row} * out_stride,
                        rb.data() + std::size_t{row} * out_stride, expanded * 2U) == 0;
                const auto sa = state_a.read(), sb = state_b.read();
                same = same && sa == sb;
                compared += std::size_t{rows} * expanded + sa.size();
            }
            std::cout << "PLE tail rows " << (bf16_weights ? "BF16" : "F32") << " weights, "
                      << (in_place ? "in place" : "separate output") << ": " << compared << " values "
                      << (same ? "identical" : "DIFFER") << '\n';
            expect(same, "PLE tail rows form is bit-exact against the per-row kernels");
        }
    }
    check(hipStreamDestroy(stream), "destroy PLE tail stream");
}

}  // namespace

int main() try {
    ple_tail_rows_cases();
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto hash_contract =
        strix::build_ple_hash_contract(
            model);

    const auto plan =
        strix::build_serial_decode_execution_plan(
            model);
    const auto io =
        strix::build_serial_execution_io_plan(
            plan);
    const auto& step =
        ple_step(plan);
    const auto& step_io =
        io.steps[step.id];

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;
    request.activation_element_bytes = 2;
    const auto arena =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            plan,
            io,
            256);

    void* arena_pointer = nullptr;
    check(
        hipMalloc(
            &arena_pointer,
            static_cast<std::size_t>(
                arena.total_bytes)),
        "hipMalloc PLE arena");

    strix::ExecutionMemoryBinding memory;
    memory.execution_identity =
        0x504c455445535410ULL;
    memory.arena_base =
        reinterpret_cast<std::uintptr_t>(
            arena_pointer);
    memory.arena_bytes =
        arena.total_bytes;
    memory.allocation_generation = 1;

    const auto input =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.reads[0]);
    const auto output =
        strix::resolve_execution_buffer(
            memory.execution_identity,
            arena,
            memory,
            step_io.writes[0]);

    const auto expanded =
        static_cast<std::size_t>(
            model.hidden_size *
            model.hc_count);
    std::vector<std::uint16_t> input_values(
        expanded);
    for (std::size_t index = 0;
         index < input_values.size();
         ++index) {
        const float value =
            static_cast<float>(
                static_cast<int>(
                    index % 19U) -
                9) /
            32.0F;
        input_values[index] =
            float_to_bf16(value);
    }
    check(
        hipMemcpy(
            reinterpret_cast<void*>(
                input.address),
            input_values.data(),
            input_values.size() *
                sizeof(std::uint16_t),
            hipMemcpyHostToDevice),
        "copy PLE hyper input");

    std::vector<std::uint16_t> gathered(
        static_cast<std::size_t>(
            model.ple_embed_dim),
        float_to_bf16(0.01F));
    FixtureLookup lookup_backend(
        gathered);

    const auto key_weights =
        make_q8(
            static_cast<std::uint32_t>(
                expanded),
            static_cast<std::uint32_t>(
                model.ple_embed_dim),
            true);
    const auto value_weights =
        make_q8(
            static_cast<std::uint32_t>(
                model.hidden_size),
            static_cast<std::uint32_t>(
                model.ple_embed_dim),
            false);

    std::vector<float> norm_key(
        expanded, 0.0F);
    std::vector<float> norm_query(
        expanded, 0.0F);
    std::vector<float> norm_conv(
        expanded, 0.0F);
    std::vector<float> conv(
        static_cast<std::size_t>(
            model.ple_conv_kernel_size) *
            expanded,
        0.0F);

    strix::ModelManifest manifest;
    manifest.architecture =
        "qwen4_exp";

    auto add_tensor =
        [&](std::string name,
            strix::TensorRole role,
            std::uint32_t type,
            std::vector<std::uint64_t> dims,
            std::uint64_t bytes,
            bool layered) {
            strix::ManifestTensor tensor;
            tensor.name =
                std::move(name);
            tensor.role = role;
            tensor.ggml_type = type;
            tensor.dimensions =
                std::move(dims);
            tensor.byte_size = bytes;
            if (layered) {
                tensor.layer =
                    step.layer;
            }
            manifest.total_tensor_bytes +=
                bytes;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    const auto synthetic_table_bytes =
        hash_contract.minimum_table_rows *
        hash_contract.row_bytes;
    add_tensor(
        "per_layer_token_embd.weight",
        strix::TensorRole::ngram_table,
        30,
        {
            model.ple_head_dim(),
            hash_contract.minimum_table_rows,
        },
        synthetic_table_bytes,
        false);
    add_tensor(
        "blk.1.ple_key.weight",
        strix::TensorRole::ple_parameter,
        8,
        {
            model.ple_embed_dim,
            expanded,
        },
        key_weights.size() *
            sizeof(Q8Block),
        true);
    add_tensor(
        "blk.1.ple_value.weight",
        strix::TensorRole::ple_parameter,
        8,
        {
            model.ple_embed_dim,
            model.hidden_size,
        },
        value_weights.size() *
            sizeof(Q8Block),
        true);
    add_tensor(
        "blk.1.ple_norm_key.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_key.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_norm_query.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_query.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_norm_conv.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.hidden_size,
            model.hc_count,
        },
        norm_conv.size() *
            sizeof(float),
        true);
    add_tensor(
        "blk.1.ple_conv1d.weight",
        strix::TensorRole::ple_parameter,
        0,
        {
            model.ple_conv_kernel_size,
            expanded,
        },
        conv.size() *
            sizeof(float),
        true);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);

    std::vector<void*> device_weights(
        manifest.tensors.size(),
        nullptr);
    upload(
        key_weights,
        &device_weights[1]);
    upload(
        value_weights,
        &device_weights[2]);
    // Fold source zero-centered gammas only for serialized GGUF bindings.
    const std::vector<float> gguf_norm(expanded, 1.0F);
    upload(gguf_norm, &device_weights[3]);
    upload(gguf_norm, &device_weights[4]);
    upload(gguf_norm, &device_weights[5]);
    upload(
        conv,
        &device_weights[6]);

    strix::ModelRuntimeBinding runtime;
    runtime.manifest_layout_identity =
        manifest.layout_identity;
    runtime.model_content_identity.bytes[0] =
        1;
    runtime.tensors.reserve(
        manifest.tensors.size());

    runtime.tensors.push_back(
        {
            0,
            strix::RuntimeTensorStorage::file_mapped,
            static_cast<std::uintptr_t>(
                0x100000ULL),
            synthetic_table_bytes,
            1,
            0,
        });
    for (std::uint32_t index = 1;
         index <
            manifest.tensors.size();
         ++index) {
        runtime.tensors.push_back(
            {
                index,
                strix::RuntimeTensorStorage::resident,
                reinterpret_cast<std::uintptr_t>(
                    device_weights[index]),
                manifest.tensors[index].byte_size,
                1,
                0,
            });
    }

    strix::ExecutionTensorAccessPlan access;
    access.manifest_layout_identity =
        manifest.layout_identity;
    access.plan_identity =
        0x504c455445535411ULL;
    access.accesses.push_back(
        {
            step.id,
            step.op,
            0,
            {0},
        });
    access.accesses.push_back(
        {
            step.id,
            step.op,
            1,
            {1, 2, 3, 4, 5, 6},
        });

    const auto conv_bytes =
        strix::ple_conv_state_bytes_per_session(
            model);
    void* conv_state_pointer = nullptr;
    check(
        hipMalloc(
            &conv_state_pointer,
            static_cast<std::size_t>(
                conv_bytes)),
        "hipMalloc PLE conv state");
    check(
        hipMemset(
            conv_state_pointer,
            0,
            static_cast<std::size_t>(
                conv_bytes)),
        "clear PLE conv state");

    std::array<std::uint32_t, 2>
        history{
            123U,
            456U,
        };

    strix::SerialSessionStateLayout state_layout;
    state_layout.context_tokens = 4;
    state_layout.device_alignment = 256;
    state_layout.host_alignment = 64;
    state_layout.device_total_bytes =
        conv_bytes;
    state_layout.host_total_bytes =
        history.size() *
        sizeof(std::uint32_t);
    state_layout.layout_identity =
        0x504c455445535412ULL;
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::ple_convolution,
            strix::PersistentState::convolution_history,
            strix::SessionStateStorage::device,
            0,
            conv_bytes,
            false,
        });
    state_layout.allocations.push_back(
        {
            strix::SessionStateComponent::ngram_history,
            strix::PersistentState::ngram_history,
            strix::SessionStateStorage::host,
            0,
            history.size() *
                sizeof(std::uint32_t),
            false,
        });

    strix::SessionStateMemoryBinding state_memory;
    state_memory.layout_identity =
        state_layout.layout_identity;
    state_memory.device_base =
        reinterpret_cast<std::uintptr_t>(
            conv_state_pointer);
    state_memory.device_bytes =
        conv_bytes;
    state_memory.device_allocation_generation = 1;
    state_memory.host_base =
        reinterpret_cast<std::uintptr_t>(
            history.data());
    state_memory.host_bytes =
        history.size() *
        sizeof(std::uint32_t);
    state_memory.host_allocation_generation = 1;

    strix::ExecutionRuntimeResources resources;
    resources.scratch = &memory;
    resources.model_manifest =
        &manifest;
    resources.model_weights =
        &runtime;
    resources.tensor_access =
        &access;
    resources.session_state_layout =
        &state_layout;
    resources.session_state_memory =
        &state_memory;
    resources.ple_lookup =
        &lookup_backend;

    strix::ExecutionDispatchContext context;
    context.step_id =
        step.id;
    context.op =
        step.op;
    context.layer =
        step.layer;
    context.reads =
        step_io.reads;
    context.writes =
        step_io.writes;
    context.temporaries =
        step_io.temporaries;
    context.arena =
        &arena;
    context.memory =
        &memory;
    context.session_state_layout =
        &state_layout;
    context.session_state_memory =
        &state_memory;
    context.resources =
        &resources;
    context.reads_state =
        step.reads_state;
    context.writes_state =
        step.writes_state;
    context.logical_rows = 1;
    context.request_concurrency = 1;
    context.input_token_id = 5;
    context.position = 0;

    strix::HipPleLauncherConfig config;
    config.model = model;
    config.activation =
        strix::HipActivationStorage::bf16;
    strix::HipPleLauncherBundle synchronous_bundle(config);
    expect(synchronous_bundle.owned_upload_admission_bytes() == 0,
           "default PLE bundle creates no extra pinned-host upload pool");
    auto owned_config = config;
    owned_config.enable_owned_event_upload = true;
    strix::HipPleLauncherBundle owned_bundle(owned_config);
    expect(owned_bundle.owned_upload_admission_bytes() ==
               strix::HipPleLauncherConfig::owned_event_upload_maximum_bytes,
           "opt-in PLE pool exposes its fixed additional admission maximum");
    strix::AllocationGenerationSource upload_generations;
    strix::HipHostUploadBackend upload_backend(1);
    strix::HostUploadPool uploads(upload_backend, upload_generations,
        {1, strix::PleHashContract::maximum_heads * 512,
         strix::PleHashContract::maximum_heads * 512, 4096});
    config.upload_pool = &uploads;

    auto launcher =
        strix::make_hip_ple_launcher(
            "test.ple",
            &config);
    launcher.function(
        context,
        launcher.user_data);
    expect(uploads.active_uploads() == 0 &&
               uploads.allocated_bytes() == strix::PleHashContract::maximum_heads * 512,
           "PLE event upload retires its exact-generation host source before projections");

    std::vector<std::uint16_t> actual(
        expanded);
    check(
        hipMemcpy(
            actual.data(),
            reinterpret_cast<const void*>(
                output.address),
            actual.size() *
                sizeof(std::uint16_t),
            hipMemcpyDeviceToHost),
        "copy PLE output");

    double gathered_sum = 0.0;
    for (const auto value :
         gathered) {
        gathered_sum +=
            bf16_to_float(value);
    }

    float maximum_error = 0.0F;
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        const auto feature =
            static_cast<std::uint32_t>(
                index %
                model.hidden_size);
        const auto block =
            feature / 32U;
        const auto& q8 =
            value_weights[
                static_cast<std::size_t>(
                    feature) *
                    (model.ple_embed_dim / 32U) +
                block];
        static_cast<void>(q8);
        const float scale =
            __half2float(
                value_weights[
                    static_cast<std::size_t>(
                        feature) *
                        (model.ple_embed_dim / 32U)].
                    scale);
        const float projected =
            static_cast<float>(
                gathered_sum) *
            scale;
        const float expected =
            bf16_to_float(
                input_values[index]) +
            0.5F *
                projected;
        const float actual_value =
            bf16_to_float(
                actual[index]);
        maximum_error =
            std::max(
                maximum_error,
                std::abs(
                    actual_value -
                    expected));
    }

    expect(
        maximum_error <= 0.02F,
        "PLE zero-key fixture matches residual plus half-gated value");
    expect(
        history[0] ==
            model.ple_eos_token_id &&
        history[1] == 5,
        "PLE position-zero commit writes EOS-padded next n-gram history");

    std::vector<std::uint16_t> conv_state(
        static_cast<std::size_t>(
            conv_bytes /
            sizeof(std::uint16_t)));
    check(
        hipMemcpy(
            conv_state.data(),
            conv_state_pointer,
            static_cast<std::size_t>(
                conv_bytes),
            hipMemcpyDeviceToHost),
        "copy PLE conv state");
    const auto history_len =
        static_cast<std::size_t>(
            (model.ple_conv_kernel_size -
             1ULL) *
            model.ngram_size);
    expect(
        conv_state[
            history_len - 1U] != 0,
        "PLE convolution state commits normalized current token");

    // A nonzero-key case exercises distinct serialized key/query/conv scales,
    // including zero and negative scales. The oracle uses generated source
    // gammas and FP64 reductions, independently of the device norm kernels.
    const auto nonzero_key = make_q8(static_cast<std::uint32_t>(expanded),
        static_cast<std::uint32_t>(model.ple_embed_dim), false);
    check(hipMemcpy(device_weights[1], nonzero_key.data(), nonzero_key.size() * sizeof(Q8Block),
                    hipMemcpyHostToDevice), "upload nonzero PLE key");
    std::vector<float> key_scale(expanded), query_scale(expanded), conv_scale(expanded);
    for (std::size_t index = 0; index < expanded; ++index) {
        norm_key[index] = static_cast<float>(static_cast<int>((index * 19U + index / 13U) % 23U) - 11) / 8.0F;
        norm_query[index] = static_cast<float>(static_cast<int>((index * 29U + index / 17U + 5U) % 19U) - 9) / 8.0F;
        norm_conv[index] = static_cast<float>(static_cast<int>((index * 31U + index / 11U + 7U) % 21U) - 10) / 8.0F;
        key_scale[index] = 1.0F + norm_key[index];
        query_scale[index] = 1.0F + norm_query[index];
        conv_scale[index] = 1.0F + norm_conv[index];
    }
    for (const auto& pair : {std::pair{3U, &key_scale}, std::pair{4U, &query_scale}, std::pair{5U, &conv_scale}})
        check(hipMemcpy(device_weights[pair.first], pair.second->data(), expanded * sizeof(float),
                        hipMemcpyHostToDevice), "upload distinct folded PLE scale");
    context.position = 1;
    context.input_token_id = 6;
    launcher.function(context, launcher.user_data);
    check(hipMemcpy(actual.data(), reinterpret_cast<void*>(output.address), expanded * sizeof(std::uint16_t),
                    hipMemcpyDeviceToHost), "read nonzero-norm PLE output");
    check(hipMemcpy(conv_state.data(), conv_state_pointer, static_cast<std::size_t>(conv_bytes),
                    hipMemcpyDeviceToHost), "read nonzero-norm PLE convolution state");
    bool output_equal = true, state_equal = true;
    const auto width = static_cast<std::size_t>(model.hidden_size);
    for (std::size_t stream = 0; stream < model.hc_count; ++stream) {
        std::vector<double> projected_key(width), gated_value(width);
        double key_square = 0.0, query_square = 0.0;
        for (std::size_t feature = 0; feature < width; ++feature) {
            const auto index = stream * width + feature;
            projected_key[feature] = gathered_sum * __half2float(nonzero_key[index * (model.ple_embed_dim / 32U)].scale);
            key_square += projected_key[feature] * projected_key[feature];
            const double query = bf16_to_float(input_values[index]);
            query_square += query * query;
        }
        const double inv_key = 1.0 / std::sqrt(key_square / static_cast<double>(width) + model.rms_norm_epsilon);
        const double inv_query = 1.0 / std::sqrt(query_square / static_cast<double>(width) + model.rms_norm_epsilon);
        double dot = 0.0;
        for (std::size_t feature = 0; feature < width; ++feature) {
            const auto index = stream * width + feature;
            dot += projected_key[feature] * inv_key * (1.0 + norm_key[index]) *
                   bf16_to_float(input_values[index]) * inv_query * (1.0 + norm_query[index]);
        }
        const double score = dot / std::sqrt(static_cast<double>(width));
        const double signed_root = score == 0.0 ? 0.0 : std::copysign(std::sqrt(std::max(std::fabs(score), 1e-6)), score);
        const double gate = 1.0 / (1.0 + std::exp(-signed_root));
        double value_square = 0.0;
        for (std::size_t feature = 0; feature < width; ++feature) {
            gated_value[feature] = gathered_sum * __half2float(value_weights[feature * (model.ple_embed_dim / 32U)].scale) * gate;
            value_square += gated_value[feature] * gated_value[feature];
        }
        const double inv_value = 1.0 / std::sqrt(value_square / static_cast<double>(width) + model.rms_norm_epsilon);
        for (std::size_t feature = 0; feature < width; ++feature) {
            const auto index = stream * width + feature;
            const double expected_output = bf16_to_float(input_values[index]) + gated_value[feature];
            const double expected_state = gated_value[feature] * inv_value * (1.0 + norm_conv[index]);
            output_equal = output_equal && std::fabs(bf16_to_float(actual[index]) - expected_output) <= 0.02;
            state_equal = state_equal && std::fabs(bf16_to_float(conv_state[index * history_len + history_len - 1U]) - expected_state) <= 0.02;
        }
    }
    expect(output_equal, "PLE distinct folded key/query scales match FP64 gate oracle");
    expect(state_equal, "PLE folded convolution scales match independently normalized current value");

    pure_preflight_cases(context, config);

    for (std::size_t index = 1;
         index < device_weights.size();
         ++index) {
        check(
            hipFree(
                device_weights[index]),
            "hipFree PLE weight");
    }
    check(
        hipFree(
            conv_state_pointer),
        "hipFree PLE conv state");
    check(
        hipFree(
            arena_pointer),
        "hipFree PLE arena");

    if (failures == 0) {
        std::cout
            << "HIP PLE launcher tests passed\n";
    }
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr
        << "HIP PLE launcher tests: "
        << error.what() << '\n';
    return 2;
}

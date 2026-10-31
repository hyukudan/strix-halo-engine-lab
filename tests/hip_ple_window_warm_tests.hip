// STRIX_PLE_WINDOW_WARM fixture: verify-window PLE rows with the draft loop's token reports (decode cache warming)
// versus without, compared bitwise: every row's output, the convolution state after each launch and after commit,
// and the published n-gram history. R in {1,2,4,8}; token ids at the vocabulary edges, EOS and repeated tokens; a
// position-zero window; warming with a wrong token and with a history changed afterwards (stale warm pages).
// Synthetic weights; the IQ4_NL table lives in a sparse anonymous mapping. Every case runs on the plain file-mapped
// backend (warming is a no-op there) and on a rolling-residency decode turn (the MTP CLI default). Also checks that
// Turn::warm reads uncached pages once and then finds them cached. GPU fixture.
#include "strix/dispatch_resources.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_io.hpp"
#include "strix/execution_memory.hpp"
#include "strix/execution_plan.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/hip_ple_launcher.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/ple_rolling.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};
static_assert(sizeof(Q8Block) == 34);

// Element-dependent weights: a staged byte in the wrong place changes the projection.
std::vector<Q8Block> make_q8(std::uint32_t rows, std::uint32_t width, std::uint32_t seed) {
    const auto per_row = width / 32U;
    std::vector<Q8Block> blocks(static_cast<std::size_t>(rows) * per_row);
    for (std::uint32_t row = 0; row < rows; ++row)
        for (std::uint32_t block = 0; block < per_row; ++block) {
            auto& b = blocks[static_cast<std::size_t>(row) * per_row + block];
            b.scale = __float2half(0.0005F * static_cast<float>((row * 7U + block + seed) % 13U + 1U));
            for (std::uint32_t i = 0; i < 32U; ++i)
                b.values[i] = static_cast<std::int8_t>(static_cast<int>((row * 3U + block * 5U + i * 7U + seed) % 15U) - 7);
        }
    return blocks;
}

template <typename T>
void* upload(const std::vector<T>& values) {
    void* device = nullptr;
    check(hipMalloc(&device, values.size() * sizeof(T)), "hipMalloc PLE fixture weight");
    check(hipMemcpy(device, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice), "upload PLE fixture weight");
    return device;
}

const strix::ExecutionStep& ple_step(const strix::SerialExecutionPlan& plan) {
    for (const auto& step : plan.steps) if (step.op == strix::SerialOp::ple) return step;
    throw std::runtime_error("serial plan has no PLE step");
}

// One deterministic, finite IQ4_NL row per table row id.
void fill_row(std::byte* table, std::uint64_t row) {
    auto* p = table + row * 90U;
    for (std::uint64_t block = 0; block < 5U; ++block) {
        const auto scale = static_cast<std::uint16_t>(0x2000U + (row * 13U + block) % 1024U);
        std::memcpy(p + block * 18U, &scale, sizeof(scale));
        for (std::uint64_t j = 0; j < 16U; ++j)
            p[block * 18U + 2U + j] = static_cast<std::byte>((row * 2654435761ULL >> ((j + block) % 24U)) + j * 37U);
    }
}

struct Window {
    std::vector<std::uint32_t> tokens;
    std::uint64_t position = 0;
    std::uint32_t keep = 1;
    enum class Prefetch { none, same, other_token, stale_position, changed_history } prefetch = Prefetch::none;
};

struct Record {
    std::vector<std::uint16_t> outputs;
    std::vector<std::byte> conv_launch, conv_commit;
    std::array<std::uint32_t, 2> history{};
};

struct Fixture {
    strix::ModelContract model = strix::ModelContract::qwen38_flash_next();
    strix::PleHashContract contract = strix::build_ple_hash_contract(model);
    strix::SerialExecutionPlan plan = strix::build_serial_decode_execution_plan(model);
    strix::SerialExecutionIoPlan io = strix::build_serial_execution_io_plan(plan);
    const strix::ExecutionStep& step = ple_step(plan);
    strix::ExecutionArenaPlan arena;
    std::array<void*, 8> arenas{};
    std::array<strix::ExecutionMemoryBinding, 8> memories{};
    std::byte* table = nullptr;
    std::uint64_t table_bytes = 0;
    std::vector<void*> weights;
    strix::ModelManifest manifest;
    strix::ModelRuntimeBinding runtime;
    strix::ExecutionTensorAccessPlan access;
    void* conv = nullptr;
    std::uint64_t conv_bytes = 0;
    std::array<std::uint32_t, 2> history{};
    strix::SerialSessionStateLayout state_layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::FileMappedPleLookupBackend backend;
    std::array<strix::ExecutionRuntimeResources, 8> resources{};
    // Rolling-residency decode turn over the same table (plain mapped reads; no owned O_DIRECT cache).
    std::shared_ptr<strix::ple_reclaim::ResidencyManager> manager;
    std::shared_ptr<strix::ple_reclaim::ResidencyManager::Turn> turn;
    std::size_t expanded = 0;

    Fixture() {
        expanded = static_cast<std::size_t>(model.hidden_size * model.hc_count);
        strix::ExecutionBufferRequest request;
        request.logical_rows = 1;
        request.activation_element_bytes = 2;
        arena = strix::build_serial_execution_arena_plan(model, request, plan, io, 256);
        for (std::size_t r = 0; r < arenas.size(); ++r) {
            check(hipMalloc(&arenas[r], static_cast<std::size_t>(arena.total_bytes)), "hipMalloc PLE fixture arena");
            memories[r].execution_identity = 0x504c454541524c59ULL;
            memories[r].arena_base = reinterpret_cast<std::uintptr_t>(arenas[r]);
            memories[r].arena_bytes = arena.total_bytes;
            memories[r].allocation_generation = r + 1U;
        }
        const auto rows = contract.minimum_table_rows;
        table_bytes = rows * 90U;
        void* mapped = ::mmap(nullptr, static_cast<std::size_t>(table_bytes), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("map the sparse PLE fixture table");
        table = static_cast<std::byte*>(mapped);
        const auto add = [&](std::string name, strix::TensorRole role, std::uint32_t type, std::vector<std::uint64_t> dims,
                             std::uint64_t bytes, bool layered) {
            strix::ManifestTensor tensor;
            tensor.name = std::move(name); tensor.role = role; tensor.ggml_type = type;
            tensor.dimensions = std::move(dims); tensor.byte_size = bytes;
            if (layered) tensor.layer = step.layer;
            manifest.total_tensor_bytes += bytes;
            manifest.tensors.push_back(std::move(tensor));
        };
        const auto width = static_cast<std::uint32_t>(model.ple_embed_dim);
        const auto key = make_q8(static_cast<std::uint32_t>(expanded), width, 1U);
        const auto value = make_q8(static_cast<std::uint32_t>(model.hidden_size), width, 2U);
        std::vector<float> norm_key(expanded), norm_query(expanded), norm_conv(expanded),
            conv_weights(static_cast<std::size_t>(model.ple_conv_kernel_size) * expanded);
        for (std::size_t i = 0; i < expanded; ++i) {
            norm_key[i] = static_cast<float>(static_cast<int>(i * 19U % 23U) - 11) / 8.0F;
            norm_query[i] = static_cast<float>(static_cast<int>((i * 29U + 5U) % 19U) - 9) / 8.0F;
            norm_conv[i] = static_cast<float>(static_cast<int>((i * 31U + 7U) % 21U) - 10) / 8.0F;
        }
        for (std::size_t i = 0; i < conv_weights.size(); ++i)
            conv_weights[i] = static_cast<float>(static_cast<int>(i * 11U % 17U) - 8) / 16.0F;
        add("per_layer_token_embd.weight", strix::TensorRole::ngram_table, 20, {model.ple_head_dim(), rows}, table_bytes, false);
        add("blk.1.ple_key.weight", strix::TensorRole::ple_parameter, 8, {model.ple_embed_dim, expanded}, key.size() * sizeof(Q8Block), true);
        add("blk.1.ple_value.weight", strix::TensorRole::ple_parameter, 8, {model.ple_embed_dim, model.hidden_size},
            value.size() * sizeof(Q8Block), true);
        add("blk.1.ple_norm_key.weight", strix::TensorRole::ple_parameter, 0, {model.hidden_size, model.hc_count}, expanded * 4U, true);
        add("blk.1.ple_norm_query.weight", strix::TensorRole::ple_parameter, 0, {model.hidden_size, model.hc_count}, expanded * 4U, true);
        add("blk.1.ple_norm_conv.weight", strix::TensorRole::ple_parameter, 0, {model.hidden_size, model.hc_count}, expanded * 4U, true);
        add("blk.1.ple_conv1d.weight", strix::TensorRole::ple_parameter, 0, {model.ple_conv_kernel_size, expanded},
            conv_weights.size() * 4U, true);
        manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
        weights = {nullptr, upload(key), upload(value), upload(norm_key), upload(norm_query), upload(norm_conv), upload(conv_weights)};
        runtime.manifest_layout_identity = manifest.layout_identity;
        runtime.model_content_identity.bytes[0] = 1;
        runtime.tensors.push_back({0, strix::RuntimeTensorStorage::file_mapped, reinterpret_cast<std::uintptr_t>(table), table_bytes, 1, 0});
        for (std::uint32_t i = 1; i < manifest.tensors.size(); ++i)
            runtime.tensors.push_back({i, strix::RuntimeTensorStorage::resident, reinterpret_cast<std::uintptr_t>(weights[i]),
                manifest.tensors[i].byte_size, 1, 0});
        access.manifest_layout_identity = manifest.layout_identity;
        access.plan_identity = 0x504c454541524c5aULL;
        access.accesses.push_back({step.id, step.op, 0, {0}});
        access.accesses.push_back({step.id, step.op, 1, {1, 2, 3, 4, 5, 6}});
        conv_bytes = strix::ple_conv_state_bytes_per_session(model);
        check(hipMalloc(&conv, static_cast<std::size_t>(conv_bytes)), "hipMalloc PLE fixture conv state");
        state_layout.context_tokens = 64;
        state_layout.device_alignment = 256;
        state_layout.host_alignment = 64;
        state_layout.device_total_bytes = conv_bytes;
        state_layout.host_total_bytes = sizeof(history);
        state_layout.layout_identity = 0x504c454541524c5bULL;
        state_layout.allocations.push_back({strix::SessionStateComponent::ple_convolution, strix::PersistentState::convolution_history,
            strix::SessionStateStorage::device, 0, conv_bytes, false});
        state_layout.allocations.push_back({strix::SessionStateComponent::ngram_history, strix::PersistentState::ngram_history,
            strix::SessionStateStorage::host, 0, sizeof(history), false});
        state_memory.layout_identity = state_layout.layout_identity;
        state_memory.device_base = reinterpret_cast<std::uintptr_t>(conv);
        state_memory.device_bytes = conv_bytes;
        state_memory.device_allocation_generation = 1;
        state_memory.host_base = reinterpret_cast<std::uintptr_t>(history.data());
        state_memory.host_bytes = sizeof(history);
        state_memory.host_allocation_generation = 1;
        for (std::size_t r = 0; r < resources.size(); ++r) {
            auto& res = resources[r];
            res.scratch = &memories[r];
            res.model_manifest = &manifest; res.model_weights = &runtime; res.tensor_access = &access;
            res.session_state_layout = &state_layout; res.session_state_memory = &state_memory;
            res.ple_lookup = &backend;
        }
        strix::ple_reclaim::RollingPages::Io pages;
        pages.prefetch = [](auto) -> std::uint64_t { return 0; };
        pages.evict = [](auto) {};
        pages.resident = []() -> std::uint64_t { return 0; };
        pages.reclaimable = [](auto) { return true; };
        manager = std::make_shared<strix::ple_reclaim::ResidencyManager>("ple-window-warm-fixture",
            reinterpret_cast<std::uintptr_t>(table), 1, 256, 4096, 0, 90, rows, 16, 1ULL << 30U, 64ULL << 20U, 1, std::move(pages));
        turn = manager->begin();
    }
    ~Fixture() {
        for (auto* p : weights) if (p) (void)hipFree(p);
        for (auto* p : arenas) if (p) (void)hipFree(p);
        if (conv) (void)hipFree(conv);
        if (table) (void)::munmap(table, static_cast<std::size_t>(table_bytes));
    }

    // Writes every table row any n-gram over `symbols` can select (EOS cuts included).
    void fill(const std::set<std::uint32_t>& symbols) {
        std::set<std::uint64_t> rows;
        for (const auto t : symbols) for (const auto a : symbols) for (const auto b : symbols) {
            const std::array<std::uint32_t, 2> previous{a, b};
            const auto hash = strix::hash_ple_token(contract, t, previous);
            for (std::uint32_t h = 0; h < hash.row_count; ++h) rows.insert(hash.row_ids[h]);
        }
        for (const auto row : rows) fill_row(table, row);
    }

    std::vector<std::array<std::uint64_t, 16>> ids(const std::vector<std::uint32_t>& tokens, std::uint64_t position,
        std::array<std::uint32_t, 2> previous) const {
        if (position == 0) previous.fill(static_cast<std::uint32_t>(model.ple_eos_token_id));
        std::vector<std::array<std::uint64_t, 16>> out;
        for (const auto token : tokens) {
            const auto hash = strix::hash_ple_token(contract, token, previous);
            std::array<std::uint64_t, 16> row{};
            std::copy_n(hash.row_ids.begin(), 16, row.begin());
            out.push_back(row);
            std::copy_n(hash.next_context.begin(), 2, previous.begin());
        }
        return out;
    }

    void reset() {
        check(hipMemset(conv, 0, static_cast<std::size_t>(conv_bytes)), "clear PLE fixture conv state");
        history = {123U, 456U};
    }

    std::vector<strix::ExecutionDispatchContext> contexts(const Window& w) {
        const auto& step_io = io.steps[step.id];
        std::vector<strix::ExecutionDispatchContext> out(w.tokens.size());
        for (std::size_t r = 0; r < out.size(); ++r) {
            auto& c = out[r];
            c.step_id = step.id; c.op = step.op; c.layer = step.layer;
            c.reads = step_io.reads; c.writes = step_io.writes; c.temporaries = step_io.temporaries;
            c.arena = &arena; c.memory = &memories[r];
            c.session_state_layout = &state_layout; c.session_state_memory = &state_memory;
            c.resources = &resources[r]; c.reads_state = step.reads_state; c.writes_state = step.writes_state;
            c.logical_rows = 1; c.request_concurrency = 1;
            c.input_token_id = w.tokens[r]; c.position = w.position + r;
            // Each row's own input: deterministic per row and window.
            const auto input = strix::resolve_dispatch_read(c, 0);
            std::vector<std::uint16_t> values(expanded);
            for (std::size_t i = 0; i < expanded; ++i) {
                const float v = static_cast<float>(static_cast<int>((i * 7U + r * 13U + w.position) % 19U) - 9) / 32.0F;
                std::uint32_t bits = 0; std::memcpy(&bits, &v, 4);
                values[i] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
            }
            check(hipMemcpy(reinterpret_cast<void*>(input.address), values.data(), expanded * 2U, hipMemcpyHostToDevice),
                "upload PLE fixture row input");
        }
        return out;
    }

    std::vector<std::byte> conv_state() const {
        std::vector<std::byte> out(static_cast<std::size_t>(conv_bytes));
        check(hipMemcpy(out.data(), conv, out.size(), hipMemcpyDeviceToHost), "read PLE fixture conv state");
        return out;
    }

    std::vector<Record> run(std::uint32_t rows, const std::vector<Window>& windows, bool warm, bool leased) {
        reset();
        backend.bind_residency(leased ? turn : nullptr);
        strix::HipPleKnownWindow owner(model, strix::HipActivationStorage::bf16, rows);
        std::vector<Record> records;
        for (const auto& w : windows) {
            auto rows_of = contexts(w);
            // The draft loop's calls: row 0's token, then each draft (row k + 1) as the host learns it.
            if (warm && w.prefetch != Window::Prefetch::none) {
                auto tokens = w.tokens;
                if (w.prefetch == Window::Prefetch::other_token) tokens.back() = tokens.back() == 7U ? 8U : 7U;
                strix::ple_window_warm::begin(tokens[0]);
                for (std::uint32_t k = 0; k + 1U < tokens.size(); ++k) strix::ple_window_warm::draft(k, tokens[k + 1U]);
            }
            if (w.prefetch == Window::Prefetch::changed_history) history = {history[1], 99U};
            owner.launch(rows_of);
            check(hipDeviceSynchronize(), "complete PLE fixture window");
            Record record;
            for (std::size_t r = 0; r < w.tokens.size(); ++r) {
                const auto output = strix::resolve_dispatch_write(rows_of[r], 0);
                std::vector<std::uint16_t> values(expanded);
                check(hipMemcpy(values.data(), reinterpret_cast<const void*>(output.address), expanded * 2U, hipMemcpyDeviceToHost),
                    "read PLE fixture row output");
                record.outputs.insert(record.outputs.end(), values.begin(), values.end());
            }
            record.conv_launch = conv_state();
            owner.commit(w.keep, nullptr);
            check(hipDeviceSynchronize(), "complete PLE fixture commit");
            record.conv_commit = conv_state();
            record.history = history;
            records.push_back(std::move(record));
        }
        return records;
    }
};

}  // namespace

int main() try {
    // The switch is read once; the plain arm never reports tokens (as with the switch off).
    ::setenv("STRIX_PLE_WINDOW_WARM", "1", 1);
    Fixture f;
    const auto v = static_cast<std::uint32_t>(f.model.vocab_size), eos = static_cast<std::uint32_t>(f.model.ple_eos_token_id);
    const std::vector<std::uint32_t> edges{0U, v - 1U, eos, 0U, v - 1U, eos, 1U, v - 2U};
    const std::vector<std::uint32_t> repeated(8, 7U);
    const std::vector<std::uint32_t> mixed{eos, 5U, eos, 5U, 8U, 8U, v - 1U, 0U};
    std::set<std::uint32_t> symbols{123U, 456U, 99U, 7U, 8U, 5U, 1U, 0U, v - 1U, v - 2U, eos};
    f.fill(symbols);
    std::size_t compared = 0;
    {
        // Turn::warm loads pages without a copy: the first call reads, a repeat finds them in the decode cache.
        const auto row = f.ids({7U}, 10U, {123U, 456U}).front();
        const std::vector<std::uint64_t> ids(row.begin(), row.end());
        const auto address = reinterpret_cast<std::uintptr_t>(f.table);
        const auto first = f.turn->warm(address, 1, ids), again = f.turn->warm(address, 1, ids);
        expect(first > 0U && again == 0U, "decode warming admits uncached pages once, then finds them cached");
    }
    for (const std::uint32_t rows : {1U, 2U, 4U, 8U}) {
        const auto take = [rows](const std::vector<std::uint32_t>& t) { return std::vector<std::uint32_t>(t.begin(), t.begin() + rows); };
        const auto partial = rows > 1U ? rows - 1U : 1U;
        using P = Window::Prefetch;
        std::uint64_t p = 10;
        std::vector<Window> windows;
        const auto next = [&](std::vector<std::uint32_t> tokens, std::uint32_t keep, P prefetch, std::optional<std::uint64_t> at = {}) {
            const auto position = at ? *at : p;
            windows.push_back({std::move(tokens), position, keep, prefetch});
            p = position + keep;
        };
        next(take(edges), rows, P::none);                  // primes the owner (first launch gathers inline)
        next(take(repeated), partial, P::same);
        next(take(mixed), rows, P::same);
        next(take(edges), rows, P::same, 0U);              // position zero: EOS-filled history
        next(take(mixed), partial, P::other_token);
        next(take(repeated), rows, P::stale_position);
        next(take(mixed), rows, P::changed_history);
        next(take(edges), rows, P::same);
        next(take(repeated), rows, P::same);
      for (const bool leased : {false, true}) {
        const auto plain = f.run(rows, windows, false, leased);
        const auto warmed = f.run(rows, windows, true, leased);
        for (std::size_t i = 0; i < windows.size(); ++i) {
            const auto& x = plain[i];
            const auto& y = warmed[i];
            const auto where = std::string(leased ? "leased " : "mapped ") + "R" + std::to_string(rows) + " window " + std::to_string(i);
            expect(x.outputs == y.outputs, where + ": row outputs are bitwise equal");
            expect(x.conv_launch == y.conv_launch && x.conv_commit == y.conv_commit, where + ": convolution states are bitwise equal");
            expect(x.history == y.history, where + ": published n-gram history is equal");
            ++compared;
        }
      }
    }
    f.backend.bind_residency(nullptr);
    expect(f.manager->healthy(), "the decode turn stays healthy across warmed and plain verify leases");
    if (failures == 0) std::cout << "HIP PLE window warm tests passed (" << compared << " windows)\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "HIP PLE window warm tests: " << error.what() << '\n';
    return 2;
}

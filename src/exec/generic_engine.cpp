#include "strix/exec/generic_engine.hpp"
#include "strix/fnv.hpp"

#include "executor_cpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <utility>

namespace strix::exec {
namespace {

using session::FinishReason;

enum class Mode { mixed, input, seal };

void validate_sampler(const session::SamplerState& sampler) {
    const auto& p = sampler.parameters;
    if (sampler.rng_draws || sampler.constraint || p.temperature != 0.0 || p.top_p != 1.0 ||
        p.top_k || p.min_p != 0.0 || p.seed || p.seeded)
        throw std::invalid_argument("generic CPU engine requires greedy selection with cursor zero");
}

bool cancelled(const session::AdvanceItem& item) {
    return item.sampler.cancellation.stop_requested() || (item.cancelled && item.cancelled());
}

[[noreturn]] void unsupported_snapshot() {
    throw std::invalid_argument("generic CPU state does not support snapshots");
}

}  // namespace

struct GenericEngine::Impl {
    static constexpr auto vocab_index = static_cast<std::size_t>(weight::tensor_map::Sym::vocab);

    struct Lane {
        std::optional<std::uint32_t> retained;
        FinishReason finish = FinishReason::none;
        bool needs_fresh_cancel = false;
    };

    struct Work {
        const session::AdvanceItem* item = nullptr;
        session::AdvanceResult* result = nullptr;
        std::vector<std::uint32_t> input;
        std::optional<std::uint32_t> pick;
        std::unique_ptr<state::Txn> transaction;
        bool decode = false;
    };

    Impl(graph::ModelDecl model, CpuBinding binding, const profile::Resolution& resolution)
        : declaration(std::move(model)), vocabulary(declaration.symbols.value[vocab_index]),
          images(std::move(binding.images)), tier(resolution.profile),
          state_rows(std::move(binding.state_rows)) {
        if (!binding.context || binding.context > kStepMaxRows || !binding.slots ||
            !vocabulary || vocabulary > session::kEmbedIdBit || declaration.stages.select_after ||
            declaration.stages.group || !images || !binding.kernels || !binding.buffers || !binding.plan ||
            !state_rows || (resolution.is_turbo() && !resolution.engine))
            throw std::invalid_argument("invalid generic CPU model, context or binding");

        weight::LoadReceipt loaded;
        const auto status = weight::load(binding.load, *images, loaded);
        if (!status) throw std::runtime_error("generic CPU weight load: " + status.detail);
        std::string load_identity = profile::hex_identity(loaded.plan_identity);
        for (const auto& image : loaded.images) load_identity += ":" + profile::hex_identity(image.crc32c);

        state::LayoutOptions options;
        options.max_pending = binding.context;
        physical = std::make_unique<state::CpuState>(
            state::layout(binding.states, declaration, binding.context, binding.slots, options), binding.updates);
        CompilerOptions compile_options;
        const auto resolved = profile::hex_identity(resolution.fingerprint());
        const auto declared = profile::hex_identity(declaration.identity);
        compile_options.declaration_identity = declared + ":" + resolved;
        const auto loaded_identity = host_fnv::field(host_fnv::offset64, load_identity);
        compile_options.weight_format_identity = loaded_identity;
        compile_options.state_format_identity = physical->plan().identity;
        compiler = std::make_unique<Compiler>(declaration.stages, std::move(binding.buffers),
            binding.kernels(*images, *physical), std::move(compile_options), std::move(binding.plan));

        run_receipt.engine = declaration.name;
        run_receipt.implementation = "exec.generic.cpu.v1";
        run_receipt.device = "cpu";
        run_receipt.topology = "single";
        run_receipt.identity.pack_hash = profile::hex_identity(loaded_identity);
        run_receipt.identity.state_format = profile::hex_identity(physical->plan().identity);
        run_receipt.digests = {{"declaration", declared}, {"resolution", resolved},
                               {"load_plan", profile::hex_identity(loaded.plan_identity)}};

        lanes.resize(binding.slots);
        session::Hooks hooks;
        hooks.initialize = [this](session::KvHandle handle) {
            physical->reset(handle.slot);
            lanes[handle.slot] = {};
        };
        hooks.reset = hooks.initialize;
        hooks.release = [](session::KvHandle) {};
        hooks.view = hooks.capture = [](session::KvHandle, std::uint32_t) -> session::RankState {
            unsupported_snapshot();
        };
        hooks.restore = [](session::KvHandle, std::uint32_t, const session::RankState&) { unsupported_snapshot(); };
        hooks.truncate = [](session::KvHandle, std::uint32_t, std::uint64_t) { unsupported_snapshot(); };
        kv = std::make_unique<session::KvStore>(binding.slots, 1, std::move(hooks));
    }

    [[nodiscard]] session::Frontier frontier(session::KvHandle handle) const {
        const session::Frontier result{kv->consumed(handle), kv->pending(handle)};
        if (physical->entry(0, handle.slot).frontier != result.consumed_tokens)
            throw std::logic_error("generic CPU state and token frontiers disagree");
        if (result.pending && *result.pending >= vocabulary)
            throw std::invalid_argument("generic CPU pending token exceeds vocabulary");
        return result;
    }

    [[nodiscard]] std::uint32_t context() const { return physical->plan().context; }
    [[nodiscard]] std::uint32_t slots() const { return physical->plan().slots; }

    [[nodiscard]] std::vector<session::AdvanceResult> run(std::span<const session::AdvanceItem> items, Mode mode) {
        if (items.size() > std::min(slots(), kStepMaxRows)) throw std::invalid_argument("generic CPU batch limit");
        std::vector<session::AdvanceResult> results(items.size());
        std::vector<Work> work(items.size());
        std::array<std::uint32_t, kStepMaxRows> seen{};
        std::size_t count = 0;
        for (const auto& item : items) {
            auto& w = work[count];
            auto& result = results[count];
            w.item = &item;
            w.result = &result;
            result.output.before = result.output.after = frontier(item.state);
            const auto& before = result.output.before;
            const auto& lane = lanes[item.state.slot];
            auto used = std::span(seen).first(count);
            if (std::ranges::find(used, item.state.slot) != used.end())
                throw std::invalid_argument("duplicate generic CPU lane");
            seen[count++] = item.state.slot;
            validate_row_hints(item.row_hints, false);
            validate_sampler(item.sampler);
            if (mode == Mode::seal ? lane.finish == FinishReason::none || lane.finish == FinishReason::cancelled
                                   : lane.finish != FinishReason::none)
                throw std::invalid_argument("invalid generic CPU lane lifecycle for this operation");
            const auto logical = before.consumed_tokens + before.pending.has_value();
            if (logical > context() || item.prompt.size() > context() - logical ||
                std::ranges::any_of(item.prompt, [this](auto id) { return id >= vocabulary; }))
                throw std::invalid_argument("generic CPU prompt exceeds context or vocabulary");

            w.decode = mode == Mode::mixed && item.prompt.empty();
            if (w.decode && lane.needs_fresh_cancel && item.sampler.cancellation.stop_requested())
                throw std::invalid_argument("generic CPU resume requires a fresh cancellation source");
            result.rows = {0, 0, 0};
        }

        for (auto& w : work) {
            const auto& item = *w.item;
            auto& out = w.result->output;
            const auto& before = out.before;
            if (mode == Mode::mixed) {
                if (cancelled(item)) out.finish = FinishReason::cancelled;
                else if (!item.sampler.max_tokens) out.finish = FinishReason::length;
                else if (w.decode && before.consumed_tokens + before.pending.has_value() == context())
                    out.finish = FinishReason::context;
                if (out.finish != FinishReason::none) continue;
            }
            if (w.decode || mode == Mode::seal || !item.prompt.empty()) {
                if (before.pending) w.input.push_back(*before.pending);
                w.input.insert(w.input.end(), item.prompt.begin(), item.prompt.end());
            }
            if (w.decode) {
                if (w.input.empty() && !lanes[item.state.slot].retained)
                    throw std::invalid_argument("generic CPU decode requires prefill");
                out.tokens.reserve(1);
            }
            if (!w.input.empty()) kv->reserve_consumed(item.state, w.input.size());
        }

        // All executor work and epilogues remain tentative until both phases
        // succeed. Destruction aborts every open S1 window if anything throws.
        run_phase(work, false, mode);
        run_phase(work, true, mode);
        for (auto& w : work) publish(w, mode);
        return results;
    }

    void run_phase(std::span<Work> work, bool decoding, Mode mode) {
        std::array<Work*, kStepMaxRows> group{};
        std::size_t size = 0, rows = 0;
        for (auto& w : work) {
            auto& out = w.result->output;
            if (w.decode != decoding || out.finish != FinishReason::none || (!decoding && w.input.empty())) continue;
            if (rows + w.input.size() > kStepMaxRows) {
                submit(std::span(group).first(size), static_cast<std::uint32_t>(rows), decoding);
                size = rows = 0;
            }
            if (mode == Mode::mixed && cancelled(*w.item)) {
                out.finish = FinishReason::cancelled;
                w.input.clear();
                continue;
            }
            if (w.input.empty()) {
                w.pick = lanes[w.item->state.slot].retained;
            } else {
                group[size++] = &w;
                rows += w.input.size();
            }
        }
        if (size) submit(std::span(group).first(size), static_cast<std::uint32_t>(rows), decoding);
    }

    void submit(std::span<Work* const> group, std::uint32_t width, bool decoding) {
        const ProgramKey key{.phase = decoding ? Phase::decode : Phase::prefill,
                             .rows = decoding ? std::bit_ceil(width) : width, .tier = tier};
        const auto& program = compiler->get(key);
        std::array<StepRow, kStepMaxRows> rows{};
        std::uint32_t index = 0;
        for (const auto* w : group) {
            const auto first = index;
            std::uint32_t ancestors = 0;
            auto position = static_cast<std::uint32_t>(w->result->output.before.consumed_tokens);
            for (const auto id : w->input) {
                auto flags = static_cast<std::uint32_t>(step_valid);
                if (index - first + 1 == w->input.size()) flags |= step_sample;
                rows[index] = {id, position++, w->item->state.slot,
                               index == first ? kStepNoParent : index - 1U, ancestors, flags, 0, 0};
                ancestors |= std::uint32_t{1} << index;
                ++index;
            }
        }
        if (program.arena.total_bytes > workspace.max_size() - program.arena.alignment)
            throw std::length_error("generic CPU arena exceeds host capacity");
        const auto bytes = static_cast<std::size_t>(program.arena.total_bytes);
        const auto alignment = static_cast<std::size_t>(program.arena.alignment);
        if (workspace.size() < bytes + alignment) workspace.resize(bytes + alignment);
        void* address = workspace.data();
        auto space = workspace.size();
        const auto arena = std::span(static_cast<std::byte*>(std::align(alignment, bytes, address, space)), bytes);
        CpuExecutor executor(arena);
        std::array<RowOut, kStepMaxRows> picks{};
        const auto ticket = executor.submit(program, {std::span(rows).first(width)});
        executor.wait(ticket, std::span(picks).first(key.rows));

        std::vector<state::Row> epilogues(physical->plan().instances.size());
        index = 0;
        for (auto* w : group) {
            const auto consumed = static_cast<std::uint32_t>(w->input.size());
            const auto pick = picks[index + consumed - 1U];
            if ((pick.status != RowStatus::picked && pick.status != RowStatus::stop) || pick.id >= vocabulary)
                throw std::runtime_error("generic CPU program did not emit a valid last-row pick");
            w->pick = pick.id;
            w->transaction = std::make_unique<state::Txn>(*physical, w->item->state.slot);
            w->transaction->begin(consumed);
            for (std::uint32_t row = 0; row < consumed; ++row, ++index) {
                std::ranges::fill(epilogues, state::Row{});
                state_rows(program, arena, index, epilogues);
                w->transaction->append(epilogues);
            }
        }
        group.front()->result->rows.padding_rows = key.rows - width;
    }

    void publish(Work& w, Mode mode) {
        const auto& item = *w.item;
        auto& result = *w.result;
        auto& out = result.output;
        auto& lane = lanes[item.state.slot];
        const auto count = static_cast<std::uint32_t>(w.input.size());
        if (w.transaction) {
            // Full-window commit executes no replay callbacks. History storage
            // was reserved above, and the first ID is the validated anchor.
            w.transaction->commit(count);
            kv->append_consumed(item.state, w.input);
            lane.retained = w.pick;
            result.input_tokens = mode == Mode::seal ? 0 : item.prompt.size();
            out.after = {out.before.consumed_tokens + count, std::nullopt};
            result.rows.useful_rows = count;
            result.rows.physical_rows = count + *result.rows.padding_rows;
        }
        if (w.decode && w.pick) {
            const auto id = *w.pick;
            kv->set_pending(item.state, id);
            out.after.pending = id;
            out.tokens.push_back(id);
            lane.needs_fresh_cancel = false;
            if (std::ranges::find(item.sampler.stop_tokens, id) != item.sampler.stop_tokens.end())
                out.finish = FinishReason::stop;
            else if (out.after.consumed_tokens + 1U == context()) out.finish = FinishReason::context;
            else if (item.sampler.max_tokens == 1) out.finish = FinishReason::length;
        }
        if (mode == Mode::mixed) lane.finish = out.finish;
    }

    graph::ModelDecl declaration;
    const std::uint64_t vocabulary;
    std::unique_ptr<weight::ImageSink> images;
    profile::Name tier;
    decltype(CpuBinding::state_rows) state_rows;
    std::unique_ptr<state::CpuState> physical;
    std::unique_ptr<Compiler> compiler;
    std::vector<std::byte> workspace;
    std::vector<Lane> lanes;
    profile::ExecutionReceipt run_receipt;
    std::unique_ptr<session::KvStore> kv;
};

GenericEngine::GenericEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GenericEngine::~GenericEngine() = default;

std::unique_ptr<GenericEngine> GenericEngine::open_cpu(graph::ModelDecl model, CpuBinding binding,
                                                       const profile::Resolution& resolution) {
    return std::unique_ptr<GenericEngine>(new GenericEngine(
        std::make_unique<Impl>(std::move(model), std::move(binding), resolution)));
}

session::KvStore& GenericEngine::store() noexcept { return *impl_->kv; }

session::Capabilities GenericEngine::caps() const {
    return {.max_context = impl_->context(), .max_slots = impl_->slots(),
            .max_batch = std::min(impl_->slots(), kStepMaxRows), .batch_prefill = true,
            .batch_decode = true, .lane_cancellation = true, .external_append = true,
            .lane_resume = true, .mixed_advance = true, .turn_commit = true};
}

profile::ExecutionReceipt GenericEngine::receipt() const { return impl_->run_receipt; }

void GenericEngine::prefill(std::span<const std::uint32_t> ids, session::KvHandle& handle) {
    const session::PrefillItem item{ids, handle};
    prefill_batch(std::span(&item, 1));
}

void GenericEngine::append_external(std::span<const std::uint32_t> ids, session::KvHandle& handle) {
    prefill(ids, handle);
}

void GenericEngine::prefill_batch(std::span<const session::PrefillItem> items) {
    const session::SamplerState sampler;
    std::vector<session::AdvanceItem> group;
    group.reserve(items.size());
    for (const auto& item : items) group.push_back({item.state, item.tokens, sampler, {}, {}});
    (void)impl_->run(group, Mode::input);
}

session::StepResult GenericEngine::decode(session::KvHandle& handle, const session::SamplerState& sampler) {
    const session::DecodeItem item{handle, sampler};
    auto out = decode_batch(std::span(&item, 1));
    return std::move(out.front());
}

std::vector<session::StepResult> GenericEngine::decode_batch(std::span<const session::DecodeItem> items) {
    std::vector<session::AdvanceItem> group;
    group.reserve(items.size());
    for (const auto& item : items) group.push_back({item.state, {}, item.sampler, {}, item.row_hints});
    std::vector<session::StepResult> results(items.size());
    auto out = advance(group);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].output.rows = out[i].rows;
        results[i] = std::move(out[i].output);
    }
    return results;
}

std::vector<session::AdvanceResult> GenericEngine::advance(std::span<const session::AdvanceItem> items) {
    return impl_->run(items, Mode::mixed);
}

void GenericEngine::cancel_lane(session::KvHandle& handle) {
    (void)impl_->frontier(handle);
    impl_->lanes[handle.slot].finish = FinishReason::cancelled;
}

void GenericEngine::resume_lane(session::KvHandle& handle) {
    (void)impl_->frontier(handle);
    auto& lane = impl_->lanes[handle.slot];
    if (lane.finish == FinishReason::cancelled) lane.needs_fresh_cancel = true;
    lane.finish = FinishReason::none;
}

void GenericEngine::commit_pending(session::KvHandle& handle) {
    const session::SamplerState sampler;
    const session::AdvanceItem item{handle, {}, sampler, {}, {}};
    (void)impl_->run(std::span(&item, 1), Mode::seal);
}

}  // namespace strix::exec

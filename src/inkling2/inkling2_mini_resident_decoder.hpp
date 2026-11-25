#pragma once

#include "core/resident_greedy.hpp"
#include "core/weight_slice_loader.hpp"
#include "inkling2_mini_resident_backend.hpp"
#include "inkling2_mini_schedule.hpp"

#if defined(__HIPCC__)
#include "core/hip_screened_head.hpp"
#include "core/hip_weight_uploader.hpp"
#else
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/weight_uploader.hpp"
#endif

#include "strix/exec/cohort_programs.hpp"
#include "strix/head_argmax.hpp"
#include "strix/logit_rows.hpp"
#include "strix/session/embed_splice_device.hpp"

#if !defined(__HIPCC__)
#include "cuda/core/step_inputs.cuh"
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strix::inkling2 {

namespace detail {

struct MiniPinnedLogitsDeleter {
    void operator()(float* address) const noexcept {
#if defined(__HIPCC__)
        (void)hipHostFree(address);
#else
        (void)cudaFreeHost(address);
#endif
    }
};
}  // namespace detail

// One unpadded token stream, bound to the selected device and stream for its
// lifetime. All checkpoint tensors and live cache slots stay on the device.
// The owner is serial. Calls return after the stream and state commit finish.
class MiniResidentDecoder {
public:
    static constexpr unsigned kMaxVerifyRows = kMiniMaxVerifyRows;
    // Multi-row score buffers (cohort lanes, verify rows) start every row on the shared head pitch.
    static constexpr std::size_t kLogitPitch = ::strix::head::logit_row_pitch(kOutputVocabulary);
    static_assert(kLogitPitch >= kOutputVocabulary &&
                  kLogitPitch * sizeof(float) % ::strix::head::kLogitRowAlignmentBytes == 0U,
                  "every score row starts on the shared head alignment");
    using Rows = std::span<const unsigned>;
    enum class LogitRows { all };

    // tail_rows: prefill() lets its final chunk take up to tail_rows rows beyond max_chunk_rows when that keeps the
    // bits of the greedy schedule (mini_prefill_tail_rows: the tail takes the same route forms in both schedules and
    // the enlarged chunk passes the device admission); otherwise the tail is dropped and the greedy chunks run. The
    // device chunk capacity is max_chunk_rows + the admitted tail. Only the HIP routes are qualified; CUDA keeps
    // greedy chunks.
    MiniResidentDecoder(const std::string& checkpoint, unsigned max_positions, unsigned max_chunk_rows,
                        bool packed_decode = false, bool step_graphs = MiniResidentBackend::default_step_graphs,
                        weight::FormatId storage = weight::kBf16WeightStorage, unsigned verify_rows = 0U,
                        bool capture_draft_taps = false, unsigned decode_rows = 1U, unsigned tail_rows = 0U)
        : prefill_rows_(MiniResidentBackend::checked_chunk_rows(max_chunk_rows)),
          tail_rows_(admitted_tail_rows(prefill_rows_, tail_rows <= std::numeric_limits<unsigned>::max() - prefill_rows_
                     ? tail_rows : throw std::invalid_argument("Inkling 2 Mini: invalid prefill tail"))),
          decode_rows_(MiniResidentBackend::checked_chunk_rows(decode_rows)),
          max_rows_(std::max({MiniResidentBackend::checked_chunk_rows(prefill_rows_ + tail_rows_),
                              std::min(kMaxVerifyRows, verify_rows), decode_rows_})),
          layout_(max_positions, max_rows_), weight_plan_(read_plan(checkpoint, storage)),
          weight_storage_(weight_plan_.bytes()),
          activations_(mini_resident_activation_elements(max_rows_)),
          history_backup_(session::cache_buffer_elements<float>(layout_.convolution_elements())),
          kv_backup_(mini_resident_max_kv_checkpoint_elements()), embeddings_(std::size_t{max_rows_} * kHiddenWidth),
          hidden_(embeddings_.span().size()), logits_(std::size_t{decode_rows_} * kLogitPitch),
          draft_taps_(capture_draft_taps ? MiniDSparkTapPlan::elements(max_positions) : 0U),
#if !defined(__HIPCC__)
          token_ids_(max_rows_),
#endif
          weights_(upload_weights()), backend_(weights_, max_rows_, stream_.get(), packed_decode, step_graphs),
          greedy_(decode_rows_, kOutputVocabulary) {
#if defined(__HIPCC__)
        // EXACT screened greedy head (core/hip_screened_head.hpp) over the BF16 head: same token and status as the
        // full head. The compact (Q8_0) head keeps its own full head.
        if (!weights_.compact() && core::hip_screened_head::enabled()) {
            screen_.emplace(weights_.bf16("unembed.weight").data(), kOutputVocabulary, kHiddenWidth, kLogitSoftcap,
                            stream_.get());
            if (!screen_->usable()) screen_.reset();
        }
#endif
#if !defined(__HIPCC__)
        cohort_context_ = max_positions;
#endif
        allocate_slots(1U);
        reset_gate_session();
    }

    ~MiniResidentDecoder() { backend_.abandon_capture(); }

    // Return the token after the last prompt input. Only that row goes through
    // the output projection. If a later chunk fails after an earlier one
    // committed, retire the session so a retry cannot append the prefix twice.
    [[nodiscard]] unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits = {}) {
        backend_.abandon_capture();
        auto& state = slots_.front()->state();
        validate_inputs(state, prompt, logits);
        return state.append_sequence(prompt, [&](std::span<const unsigned> chunk, bool last) {
            return append(state, chunk, last, last ? logits : std::span<float>{});
        }, prefill_rows_, tail_rows_);
    }

    // Prefill whose `splice` rows (e.g. the projected rows of an image) replace the looked-up rows. They pass
    // embed_norm like the token rows: the checkpoint keeps the table and proj_out rows unit-norm and puts the
    // channel profile only in the embed_norm gain. Text rows keep their arithmetic; an empty splice is prefill().
    [[nodiscard]] unsigned prefill_spliced(std::span<const unsigned> prompt, const session::EmbedSplice& splice,
                                   std::span<float> logits = {}) {
        if (splice.empty()) return prefill(prompt, logits);
        backend_.abandon_capture();
        auto& state = slots_.front()->state();
        validate_inputs(state, prompt, logits);
        splice.validate(prompt.size(), kHiddenWidth);
        if (splice.format != session::EmbedRowFormat::bf16)
            throw std::invalid_argument("Inkling 2 Mini: provided rows must be BF16");
        return state.append_sequence(prompt, [&](std::span<const unsigned> chunk, bool last) {
            // The chunk's first prompt row from the chunk itself, not from what the session committed so far.
            const auto first = static_cast<std::uint64_t>(chunk.data() - prompt.data());
            const auto part = splice.slice(first, first + chunk.size());
            return append(state, chunk, last, last ? logits : std::span<float>{}, MiniAppendKind::prefill, {}, &part);
        }, prefill_rows_, tail_rows_);
    }

    [[nodiscard]] unsigned decode(unsigned token, std::span<float> logits = {}) try {
        auto& state = slots_.front()->state();
        const std::span<const unsigned> ids{&token, 1U};
        validate_inputs(state, ids, logits);
        return append(state, ids, true, logits, MiniAppendKind::decode);
    } catch (...) {
        backend_.abandon_capture();
        throw;
    }

    // Independent serving lanes. Slot 0 is also the single-session slot used by
    // prefill/decode/verify above and the only one that records DSpark taps.
    [[nodiscard]] std::uint64_t slot_bytes() const { return layout_.kv_bytes() + layout_.convolution_bytes(); }

    void allocate_slots(unsigned slots) {
        if (slots == 0U || slots > decode_rows_) throw std::invalid_argument("invalid Mini slot count");
        slots_.reserve(slots);
        while (slots_.size() < slots) slots_.push_back(std::make_unique<Slot>(layout_));
#if !defined(__HIPCC__)
        publish_slot_directory();
#endif
    }

    void prefill_slot(unsigned slot, std::span<const unsigned> ids, bool first) {
        if (first) reset_slot(slot);
        auto& state = slots_.at(slot)->state();
        validate_inputs(state, ids, {});
        (void)state.append_sequence(ids, [&](std::span<const unsigned> chunk, bool) {
            return append(state, chunk, false, {});
        }, prefill_rows_);
    }

#if !defined(__HIPCC__)
    [[nodiscard]] MiniResidentBackend& step_program_backend() noexcept { return backend_; }
    [[nodiscard]] const MiniResidentWeights& step_program_weights() const noexcept { return weights_; }
    [[nodiscard]] MiniResidentSessionState& step_program_state() noexcept { return slots_.front()->state(); }
    [[nodiscard]] MiniResidentForwardWorkspace step_program_workspace() noexcept {
        return workspace();
    }
    [[nodiscard]] std::span<float> step_program_embeddings() noexcept { return embeddings_.span().first(kHiddenWidth); }
    [[nodiscard]] std::span<float> step_program_hidden() noexcept { return hidden_.span().first(kHiddenWidth); }
    [[nodiscard]] std::span<float> step_program_logits() noexcept { return logits_.span().first(kOutputVocabulary); }
    [[nodiscard]] core::ResidentGreedy& step_program_greedy() noexcept { return greedy_; }
#endif

    // One forward for all lanes. Every row keeps single-row decode arithmetic,
    // so a lane matches its own decode() stream. With DSpark tap capture on,
    // lanes step one at a time so each committed position records its taps.
    void decode_slots(std::span<const unsigned> slots, std::span<const unsigned> ids, std::span<unsigned> top) {
        if (slots.empty() || slots.size() != ids.size() || slots.size() != top.size() || slots.size() > decode_rows_)
            throw std::invalid_argument("Inkling 2 Mini: invalid decode batch size");

        last_logit_rows_ = 0U;
        pending_scores_ = {};  // every forward below rewrites hidden_
        groups_.clear();
        for (std::size_t row = 0; row < slots.size(); ++row) {
            auto& state = slots_.at(slots[row])->state();
            validate_inputs(state, ids.subspan(row, 1U), {});
            groups_.push_back({&state, {state.committed(), 1U}});
        }
#if !defined(__HIPCC__)
        // A cohort decode program per width (captured once, replayed for any slots and positions). The lone
        // slot-0 lane keeps its one-row step graph; DSpark taps step lanes one at a time.
        if (draft_taps_.span().empty() && !(slots.size() == 1U && slots.front() == 0U)) {
            const auto width = cohort_policy_.width(static_cast<std::uint32_t>(slots.size()), decode_rows_);
            if (width != 0U) {
                decode_cohort(slots, ids, top, width);
                return;
            }
        }
#endif
        if (groups_.size() == 1U || draft_taps_.span().size() != 0U) {
            for (std::size_t row = 0; row < slots.size(); ++row)
                top[row] = append(*groups_[row].state, ids.subspan(row, 1U), true, {}, MiniAppendKind::decode);
            // The head scores of a lone lane stay in row 0. Several lanes stepped one at a time overwrite it.
            last_logit_rows_ = slots.size() == 1U ? 1U : 0U;
            return;
        }

        const auto rows = static_cast<unsigned>(ids.size());
        const auto embeddings = embeddings_.span().first(ids.size() * kHiddenWidth);
        const auto hidden = hidden_.span().first(embeddings.size());
        const auto logits = logits_.span().first(::strix::head::logit_rows_elements(rows, kOutputVocabulary, kLogitPitch));
        // Verify routing gives each packed row the decode MoE and attention arithmetic.
        const auto kind = rows <= kMaxVerifyRows ? MiniAppendKind::verify : MiniAppendKind::decode;
        MiniResidentForward forward(backend_, workspace());
        forward.append_batch(groups_, rows, embeddings, hidden, [&] {
            embed(ids, embeddings, rows, kind);
        }, [&](MiniConstRows output) {
            for (unsigned row = 0; row < rows; ++row)
                project_head(output.data() + std::size_t{row} * kHiddenWidth,
                             logits.subspan(::strix::head::logit_row_offset(row, kLogitPitch), kOutputVocabulary), true);
            greedy_.enqueue_rows(logits, rows, kLogitPitch, stream_.get());
        }, [&] {
            for (std::size_t row = 0; row < top.size(); ++row) (void)checked_token(greedy_, row);
        });
        for (std::size_t row = 0; row < top.size(); ++row) top[row] = greedy_.token(row);
        last_logit_rows_ = rows;
    }

#if !defined(__HIPCC__)
    // Effective cohort program policy and the programs installed so far (receipts / effective config).
    [[nodiscard]] const exec::CohortProgramPolicy& cohort_programs() const noexcept { return cohort_policy_; }
    [[nodiscard]] const exec::WidthStepGraphs& cohort_graphs() const noexcept { return backend_.cohort_graphs(); }
#endif

    // Final-row scores of the last decode_slots() call, exactly as the greedy pick saw them (after the
    // output softcap). Lane `row` follows the slot order of that call. Valid until the next decode_slots().
    [[nodiscard]] bool final_logits_supported() const { return draft_taps_.span().empty(); }
    [[nodiscard]] std::uint32_t logit_width() const { return kOutputVocabulary; }
    // The last pick was screened and its score row waits for the first score read (materialize_scores): a greedy
    // caller that never reads scores never pays the full head projection.
    [[nodiscard]] bool scores_deferred() const noexcept { return pending_scores_.input != nullptr; }
    void read_slot_logits(std::uint32_t row, std::span<float> output) {
        materialize_scores();
        if (row >= last_logit_rows_ || output.size() != kOutputVocabulary)
            throw std::logic_error("Inkling 2 Mini: no final logits for that decode lane");
        reserve_staged_logits(kOutputVocabulary);
        mini_gpu::copy(staged_logits_.get(), logits_.data() + ::strix::head::logit_row_offset(row, kLogitPitch),
                       std::size_t{kOutputVocabulary} * sizeof(float), mini_gpu::device_to_host, stream_.get());
        mini_gpu::synchronize(stream_.get());
        std::copy_n(staged_logits_.get(), kOutputVocabulary, output.data());
    }

    // Read-only view of one lane's device state for state digests (tests, receipts): its cache plan, K and V
    // planes, recurrent histories and committed frontier. Slots never move; the view holds no copy.
    struct SlotView {
        const MiniSessionStatePlan& plan;
        session::KvRowPlanes kv;
        std::span<const float> history;
        unsigned committed;
    };
    [[nodiscard]] SlotView slot_view(unsigned slot) {
        auto& owner = *slots_.at(slot);
        return {layout_, owner.planes(), owner.history(), owner.state().committed()};
    }

    [[nodiscard]] unsigned max_verify_rows() const { return std::min(max_rows_, kMaxVerifyRows); }
    // The DSpark tap rows are valid only up to the slot-0 frontier. A partial
    // verify commit replays the kept positions into the same slots.
    [[nodiscard]] std::span<const std::uint16_t> committed_taps() const {
        if (draft_taps_.span().empty()) return {};
        const auto& state = slots_.front()->state();
        if (state.poisoned()) throw std::logic_error("invalid Mini resident tap frontier");
        return draft_taps_.span().first(MiniDSparkTapPlan::elements(state.committed()));
    }

    // EngineVerifyTarget binding. All replies are validated before exposing the
    // staged rows. A caller may also request the raw scores for every row.
    Rows forward_rows(const unsigned* ids, unsigned rows, LogitRows, std::span<float> logits = {}) {
        if (rows == 0U || rows > max_verify_rows())
            throw std::invalid_argument("invalid Mini resident verify width");
        auto& state = slots_.front()->state();
        const Rows input{ids, rows};
        validate_inputs(state, input, {});
        if (!logits.empty() && logits.size() != std::size_t{rows} * kOutputVocabulary)
            throw std::invalid_argument("invalid Mini verify logits shape");
        auto top = std::span(verify_top_).first(rows);
        (void)append(state, input, true, logits, MiniAppendKind::verify, top);
        std::ranges::copy(input, verify_ids_.begin());
        return top;
    }

    void read_top(Rows top, unsigned rows, unsigned* output) {
        if (!slots_.front()->state().staged() || top.data() != verify_top_.data() || top.size() != rows)
            throw std::logic_error("invalid Mini resident verify reply");
        std::ranges::copy(top, output);
    }

    // A one-row verify window (a speculator serial step) is the plain decode step, committed at once.
    [[nodiscard]] unsigned decode_committed_row(unsigned token, std::span<float> logits) {
        return decode(token, logits);
    }

    void commit(unsigned rows) {
        auto& state = slots_.front()->state();
        state.commit_staged(backend_, rows, [&](unsigned keep) {
            (void)append(state, std::span(verify_ids_).first(keep), keep == 1U, {}, MiniAppendKind::verify);
        });
    }

    // Forward and prefix commit already wait before publishing results.
    void sync() const noexcept {}

    void reset_gate_session() {
        backend_.reset_step_graph();
        reset_slot(0U);
        mini_gpu::synchronize(stream_.get());
    }

    void set_gate_route_tap(moe::RouteTap* tap) { backend_.set_gate_route_tap(tap); }
    void set_gate_order_null(bool enabled) { backend_.set_gate_order_null(enabled); }
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off); set once after
    // construction. No bool form: a bool would drop the class and run nearest@1 for mse / mse-gate-up.
    void set_moe_w4a4(moe_core::grouped_w4a4::Activations activations) { backend_.set_moe_w4a4(activations); }
    // Resolved exact down-image lever (mini_combine_image); set once after construction. True when the image route is
    // active (HIP, qualified target); every entry point that resolves the profile passes the resolved state.
    bool set_combine_image(bool selected) { return backend_.set_combine_image(selected); }
    [[nodiscard]] std::uint64_t gate_null_applications() const { return backend_.gate_null_applications(); }

    void prefill_gate_rows(std::span<const unsigned> prompt, std::span<float> logits) {
        backend_.abandon_capture();
        pending_scores_ = {};
        auto& state = slots_.front()->state();
        validate_inputs(state, prompt, {});
        const auto rows = logits.size() / kOutputVocabulary;
        if (rows == 0U || rows > prompt.size() || logits.size() % kOutputVocabulary != 0U)
            throw std::invalid_argument("invalid Mini prefill gate head shape");
        const auto start = state.committed();
        const auto first_tail_row = prompt.size() - rows;
        (void)state.append_sequence(prompt, [&](std::span<const unsigned> chunk, bool /*last*/) {
            const auto first = std::size_t{state.committed() - start};
            const auto end = first + chunk.size();
            if (end <= first_tail_row) return append(state, chunk, false, {});
            const auto offset = std::max(first, first_tail_row) - first_tail_row;
            const auto count = end - std::max(first, first_tail_row);
            return append(state, chunk, false, logits.subspan(offset * kOutputVocabulary, count * kOutputVocabulary));
        }, prefill_rows_);
    }

private:
    friend class MiniResidentDSpark;
    using Slot = session::ResidentRowSlot<MiniSessionStatePlan, mini_gpu::Buffer>;

#if !defined(__HIPCC__)
    // Directory entry of every allocated slot (K plane, V plane, histories): the state::StateTable layout the
    // cohort programs read. Slots never move, so programs stay valid when slots are added.
    void publish_slot_directory() {
        slot_directory_host_.assign(std::size_t{kMiniLaneStates} * decode_rows_, state::SlotEntry{});
        for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
            const auto planes = slots_[slot]->planes();
            const auto entry = [&](std::uint32_t lane_state, const void* base) {
                slot_directory_host_[std::size_t{lane_state} * decode_rows_ + slot] = {
                    reinterpret_cast<std::uint64_t>(base), 0U, 0U, cohort_context_, 0U};
            };
            entry(kMiniLaneKeys, planes.keys.data());
            entry(kMiniLaneValues, planes.values.data());
            entry(kMiniLaneHistory, slots_[slot]->history().data());
        }
        mini_gpu::copy(slot_directory_.data(), slot_directory_host_.data(),
                       slot_directory_host_.size() * sizeof(state::SlotEntry), mini_gpu::host_to_device, stream_.get());
        mini_gpu::synchronize(stream_.get());
    }

    [[nodiscard]] MiniCohortLanes cohort_lanes(unsigned slot, std::uint32_t width) {
        auto& owner = *slots_.at(slot);
        const auto planes = owner.planes();
        return {exec::step_lane_view(backend_.cohort_block(), slot_directory_.data(), decode_rows_, width),
                {reinterpret_cast<std::uintptr_t>(planes.keys.data()), reinterpret_cast<std::uintptr_t>(planes.values.data()),
                 reinterpret_cast<std::uintptr_t>(owner.history().data())}};
    }

    // Embedding of a cohort program: the ids come from the device step block (padded rows read id 0).
    void embed_lanes(MiniMutableRows output, unsigned rows, MiniAppendKind kind) {
        const auto table = weights_.bf16("embed.weight");
        const auto gain = weights_.bf16("embed_norm.weight");
        backend_.start_chunk(rows, kind);
        backend_.check(cuda::core::step_inputs::enqueue(backend_.cohort_block(),
            cuda::core::step_inputs::Linear{{lane_anchor_.data(), lane_ids_.data()}}, stream_.get()));
        backend_.check(enqueue_mini_cuda_embeddings(table.data(), gain.data(), lane_ids_.data(), output.data(), rows,
                                                    stream_.get()));
    }

    // One cohort decode step through the program of `width` (R = slots.size() <= width lanes; rows past R
    // are padding that repeats lane 0 and writes no state). The program covers the forward; the head runs
    // on the R real rows only, so padding never costs a head projection. When the program declines, the
    // same transaction runs the incumbent eager forward for the R lanes. Each lane's bits are those of its
    // eager row (lane stages: shared EXACT lane forms; row-wise stages: batch invariant).
    void decode_cohort(std::span<const unsigned> slots, std::span<const unsigned> ids, std::span<unsigned> top,
                       std::uint32_t width) {
        const auto rows = static_cast<unsigned>(slots.size());
        cohort_lanes_.clear();
        for (unsigned row = 0; row < rows; ++row)
            cohort_lanes_.push_back({slots[row], groups_[row].state->committed(), ids[row]});
        const auto block = exec::make_cohort_step(cohort_lanes_, width, static_cast<std::uint32_t>(slots_.size()),
                                                  cohort_context_);
        // Verify routing gives each packed row the decode MoE and attention arithmetic (as the eager batch).
        const auto kind_of = [](unsigned count) {
            return count <= kMaxVerifyRows ? MiniAppendKind::verify : MiniAppendKind::decode;
        };
        const auto logits = logits_.span().first(::strix::head::logit_rows_elements(rows, kOutputVocabulary, kLogitPitch));
        MiniResidentForward forward(backend_, workspace());
        const auto& graphs = backend_.cohort_graphs();
        const bool had_graph = graphs.policy().ready(width);
        const bool had_failed = graphs.policy().failed(width);
        bool launched = false;
        MiniResidentSessionState::append_batch(groups_, [&] { backend_.finish(); }, [&] {
            const auto program_rows = std::size_t{width} * kHiddenWidth;
            const auto program = [&] {
                padded_groups_.assign(groups_.begin(), groups_.end());
                padded_groups_.resize(width, groups_.front());
                forward.run_batch(padded_groups_, width, embeddings_.span().first(program_rows),
                                  hidden_.span().first(program_rows), [&] {
                    embed_lanes(embeddings_.span().first(program_rows), width, kind_of(width));
                });
            };
            backend_.prepare_route(width, kind_of(width));
            launched = backend_.try_cohort_graph(cohort_policy_.enabled, block, [&] {
                return cohort_lanes(slots.front(), width);
            }, program);
            if (!launched) {
                if (!had_failed && graphs.policy().failed(width))
                    std::fprintf(stderr, "Inkling CUDA cohort width=%u capture=declined fallback=eager\n", width);
                const auto eager_rows = std::size_t{rows} * kHiddenWidth;
                forward.run_batch(groups_, rows, embeddings_.span().first(eager_rows), hidden_.span().first(eager_rows),
                                  [&] { embed(ids, embeddings_.span().first(eager_rows), rows, kind_of(rows)); });
            }
            for (unsigned row = 0; row < rows; ++row)
                project_head(hidden_.data() + std::size_t{row} * kHiddenWidth,
                             logits.subspan(::strix::head::logit_row_offset(row, kLogitPitch), kOutputVocabulary), true);
            greedy_.enqueue_rows(logits, rows, kLogitPitch, stream_.get());
        }, [&] {
            for (std::size_t row = 0; row < top.size(); ++row) (void)checked_token(greedy_, row);
        });
        if (launched && had_graph) {
            auto& reported = rows < width ? reported_padded_cohort_replays_ : reported_cohort_replays_;
            if (!reported.test(width - 1U)) {
                reported.set(width - 1U);
                std::fprintf(stderr, "Inkling CUDA cohort width=%u replay=complete rows=%u nodes=%zu\n",
                             width, rows, graphs.nodes(width));
            }
        }
        for (std::size_t row = 0; row < top.size(); ++row) top[row] = greedy_.token(row);
        last_logit_rows_ = rows;
    }
#endif

    void reset_slot(unsigned slot) {
        // KV visibility is determined by the fresh frontier. No old row is read.
        slots_.at(slot)->reset([&](std::span<float> history) {
            mini_gpu::clear(history.data(), history.size_bytes(), stream_.get());
        });
    }

    [[nodiscard]] MiniResidentForwardWorkspace workspace(const MiniResidentSessionState* state = nullptr) {
        const bool taps = state && state == &slots_.front()->state();
        return {activations_.span(), history_backup_.span(), kv_backup_.span(),
                taps ? draft_taps_.span() : std::span<std::uint16_t>{}};
    }

    [[nodiscard]] static unsigned checked_token(const core::ResidentGreedy& greedy, std::size_t row) {
        const auto selected = greedy.token(row);
        if (selected >= kOutputVocabulary) throw std::runtime_error("Inkling 2 Mini: non-finite output logits");
        return selected;
    }

    void embed(std::span<const unsigned> ids, MiniMutableRows output, unsigned rows, MiniAppendKind kind) {
        const auto table = weights_.bf16("embed.weight");
        const auto gain = weights_.bf16("embed_norm.weight");
        // During a step-graph capture the token comes from the device descriptor.
        const auto* step = backend_.capture_descriptor();
        backend_.start_chunk(rows, kind);
#if defined(__HIPCC__)
        if (step) {
            core::enqueue_hip_bf16_embedding_device_row(table, kVocabulary, kHiddenWidth, &step->token,
                                                        output, stream_.get());
        } else {
            enqueue_mini_hip_embeddings(table, ids, output, stream_.get());
        }
        backend_.rms(output.data(), gain.data(), output.data(), rows);
#else
        if (!step) {
            // The caller's token span can be pageable.
            mini_gpu::copy(token_ids_.data(), ids.data(), ids.size_bytes(), mini_gpu::host_to_device, stream_.get());
        }
        backend_.check(enqueue_mini_cuda_embeddings(table.data(), gain.data(),
            step ? &step->token : token_ids_.data(), output.data(), rows, stream_.get()));
#endif
    }

    // Provided rows of one chunk: staged on the device, widened over the looked-up rows, then normalized with
    // the embed_norm gain (the same RMS launch the HIP token rows take). Prefill only: never inside a capture.
    void splice_rows(const session::EmbedSplice& splice, MiniMutableRows embeddings) {
        if (backend_.capture_descriptor()) throw std::logic_error("Inkling 2 Mini: provided rows inside a step capture");
        const auto gain = weights_.bf16("embed_norm.weight");
        const auto total = splice.rows();
        if (!splice_staging_) splice_staging_.emplace(std::size_t{max_rows_} * kHiddenWidth);
        if (total > max_rows_ || embeddings.size() < total * kHiddenWidth)
            throw std::invalid_argument("Inkling 2 Mini: provided rows exceed the chunk");
        std::size_t staged = 0;
        for (const auto& segment : splice.segments) {
            auto* device = splice_staging_->data() + staged * kHiddenWidth;
            mini_gpu::copy(device, segment.data, std::size_t{segment.rows} * kHiddenWidth * sizeof(std::uint16_t),
                           mini_gpu::host_to_device, stream_.get());
            session::enqueue_embed_splice(embeddings.data(), 1U, device, kHiddenWidth, splice.format, segment.run_row,
                                          segment.rows, stream_.get());
            auto* rows = embeddings.data() + std::size_t{segment.run_row} * kHiddenWidth;
            backend_.rms(rows, gain.data(), rows, segment.rows);
            staged += segment.rows;
        }
    }

    [[nodiscard]] static MiniResidentWeightPlan read_plan(const std::string& path, weight::FormatId storage) {
        MiniTextWeights source;
        const auto status = source.open(path);
        if (!status) throw std::runtime_error(status.detail);
        return MiniResidentWeightPlan::make(source, storage, MiniResidentBackend::resident_row_pitch_rule());
    }

    [[nodiscard]] MiniResidentWeights upload_weights() {
#if defined(__HIPCC__)
        core::HipWeightUploader uploader;
#else
        strix::cuda::core::WeightUploader uploader;
#endif
        core::WeightSliceLoader loader(uploader, {});
        auto weights = MiniResidentWeights::bind(weight_plan_, weight_storage_.span());
        loader.load_resident(weight_plan_.slices(), weight_storage_.span());
#if !defined(__HIPCC__)
        // Only the CUDA attention declaration consumes the certificate: it scans the staged attention tensors in bounded
        // chunks (no host copy of a whole tensor; no scratch means three planes), so the HIP load path is unchanged.
        core::Bf16StagedScan scan;
        weights.certify_attention(scan, [&](std::span<const std::uint16_t> staged, std::span<std::uint16_t> host) {
            mini_gpu::copy(host.data(), staged.data(), staged.size_bytes(), mini_gpu::device_to_host, stream_.get());
            mini_gpu::synchronize(stream_.get());
        });
#endif
        return weights;
    }

    void validate_inputs(const MiniResidentSessionState& state, std::span<const unsigned> tokens,
                         std::span<float> logits) const {
        if (state.poisoned()) throw std::runtime_error("Inkling 2 Mini: device session is invalid");
        if (state.staged()) throw std::logic_error("Inkling 2 Mini: commit the staged verify rows first");
        if (!layout_.admits_sequence(state.committed(), tokens.size()))
            throw std::invalid_argument("Inkling 2 Mini: input tokens exceed the session context");
        if (!logits.empty() && logits.size() != kOutputVocabulary)
            throw std::invalid_argument("Inkling 2 Mini: logits output must hold one vocabulary row");
        if (std::ranges::any_of(tokens, [](unsigned id) { return id >= kVocabulary; }))
            throw std::out_of_range("token id exceeds the Mini vocabulary");
    }

    // `rows` consecutive input rows (rows > 1 only without the softcap) into rows x vocabulary scores.
    void project_head(const float* input, std::span<float> row_scores, bool softcap, unsigned rows = 1U) {
        const auto stream = stream_.get();
        if (rows == 0U || (softcap && rows != 1U) || row_scores.size() / kOutputVocabulary < rows)
            throw std::invalid_argument("invalid head projection rows");
        if (weights_.compact()) {
            const auto& head = weights_.compact_head();
            if (softcap)
                backend_.check(core::enqueue_resident_q8_softcap_head(head, kOutputVocabulary, input, 1U,
                                                                     row_scores, kLogitSoftcap, stream));
            else
                backend_.check(core::enqueue_resident_q8_rows(head, kOutputVocabulary, input, rows,
                                                             row_scores.data(), stream));
            return;
        }
        const auto matrix = weights_.bf16("unembed.weight");
        if (!softcap) {
            const std::array project{core::Bf16RowProjection{matrix.data(), kOutputVocabulary, row_scores.data()}};
#if defined(__HIPCC__)
            backend_.check(enqueue_mini_hip_dense_rows(input, kHiddenWidth, rows, project, stream));
#else
            backend_.check(enqueue_mini_cuda_dense_rows(input, kHiddenWidth, rows, project, stream));
#endif
            return;
        }
#if defined(__HIPCC__)
        backend_.check(enqueue_mini_hip_head(input, 1U, matrix, row_scores, stream));
#else
        backend_.check(enqueue_mini_cuda_logits(matrix.data(), input, row_scores.data(), 1U, stream));
#endif
    }

    void reserve_staged_logits(std::size_t elements) {
        if (elements <= staged_logits_size_) return;
        void* address{};
#if defined(__HIPCC__)
        mini_gpu::check(hipHostMalloc(&address, elements * sizeof(float)), "allocate Mini logits reply");
#else
        mini_gpu::check(cudaMallocHost(&address, elements * sizeof(float)), "allocate Mini logits reply");
#endif
        staged_logits_.reset(static_cast<float*>(address));
        staged_logits_size_ = elements;
    }

    [[nodiscard]] unsigned append(MiniResidentSessionState& state, std::span<const unsigned> ids, bool pick_next,
                                  std::span<float> host_logits, MiniAppendKind kind = MiniAppendKind::prefill,
                                  std::span<unsigned> verify_top = {}, const session::EmbedSplice* splice = nullptr) {
        static_assert(std::is_same_v<unsigned, std::uint32_t>);
        const auto rows = static_cast<unsigned>(ids.size());
        const auto elements = ids.size() * kHiddenWidth;
        const auto embeddings = embeddings_.span().first(elements);
        reserve_staged_logits(host_logits.size());
        pending_scores_ = {};
        const bool multi_pick = verify_top.size() > 1U;
        if (multi_pick && !verify_greedy_) {
            if (!verify_logits_) verify_logits_.emplace(std::size_t{max_verify_rows()} * kLogitPitch);
            verify_greedy_.emplace(max_verify_rows(), kOutputVocabulary);
        }
        auto& greedy = multi_pick ? *verify_greedy_ : greedy_;
        auto token = head_argmax::kNone;
        // The one-row decode step of the single session replays from a device step descriptor. DSpark
        // taps and the other slots bind position- or slot-dependent addresses, so they stay eager.
        const bool decode_step = kind == MiniAppendKind::decode && rows == 1U &&
                                 &state == &slots_.front()->state() && draft_taps_.span().empty();
#if defined(__HIPCC__)
        // Only a one-row greedy pick without host logits may skip the full logits row. A captured step graph replays
        // the head it captured, so a change of head between decode steps drops the graph (it is captured again).
        const bool screened = screen_.has_value() && pick_next && !multi_pick && host_logits.empty();
        if (decode_step && screened != screened_graph_) {
            backend_.abandon_capture();
            screened_graph_ = screened;
        }
#endif
        try {
            backend_.set_step_token(decode_step ? std::optional<unsigned>{ids.front()} : std::nullopt);
            MiniResidentForward forward(backend_, state, workspace(&state));
            forward.append({state.committed(), rows}, embeddings, hidden_.span().first(elements),
                [&](MiniResidentChunk, MiniConstRows hidden) {
                    if (!pick_next && host_logits.empty()) return;
                    const auto head_rows = !verify_top.empty() ? verify_top.size() :
                        (pick_next ? std::size_t{1} : host_logits.size() / kOutputVocabulary);
                    const auto tail = hidden.last(head_rows * kHiddenWidth);
                    const auto scores = multi_pick ? verify_logits_->span() : logits_.span();
                    // Use the single-row head for every score row. A wide dense
                    // dispatch can select different arithmetic on the same device.
                    for (std::size_t row = 0; row < head_rows; ++row) {
                        const auto* input = tail.data() + row * kHiddenWidth;
#if defined(__HIPCC__)
                        if (screened) {
                            greedy.enqueue_produced([&](HipGreedyRowsReply* reply) {
                                backend_.check(screen_->enqueue(input, reply, stream_.get()));
                            });
                            continue;
                        }
#endif
                        const auto row_scores = scores.subspan(
                            ::strix::head::logit_row_offset(multi_pick ? static_cast<unsigned>(row) : 0U, kLogitPitch), kOutputVocabulary);
                        project_head(input, row_scores, true);
                        if (pick_next && !multi_pick) greedy.enqueue(row_scores, stream_.get());
                        if (!decode_step && !host_logits.empty())
                            mini_gpu::copy(staged_logits_.get() + row * kOutputVocabulary, row_scores.data(),
                                           row_scores.size_bytes(), mini_gpu::device_to_host, stream_.get());
                    }
                    if (multi_pick) {
                        const auto picked = static_cast<unsigned>(head_rows);
                        greedy.enqueue_rows(scores.first(::strix::head::logit_rows_elements(picked, kOutputVocabulary, kLogitPitch)), picked,
                                            kLogitPitch, stream_.get());
                    }
                },
                [&] {
                    if (!pick_next) return;
                    if (verify_top.empty()) token = checked_token(greedy, 0U);
                    else for (std::size_t row = 0; row < verify_top.size(); ++row)
                        verify_top[row] = checked_token(greedy, row);
                }, [&] {
                    embed(ids, embeddings, rows, kind);
                    if (splice && !splice->empty()) splice_rows(*splice, embeddings);
                }, [&] {
                    // The optional host reply buffer can grow between calls.
                    if (decode_step && !host_logits.empty())
                        mini_gpu::copy(staged_logits_.get(), logits_.data(),
                                       std::size_t{kOutputVocabulary} * sizeof(float),
                                       mini_gpu::device_to_host, stream_.get());
                }, !verify_top.empty());
        } catch (...) {
            backend_.abandon_capture();
            state.drain_or_poison([&] { mini_gpu::synchronize(stream_.get()); });
            throw;
        }
        if (!host_logits.empty()) std::copy_n(staged_logits_.get(), host_logits.size(), host_logits.data());
#if defined(__HIPCC__)
        // The screened pick skipped the score row; a later score read recomputes it from this head input.
        if (screened) pending_scores_ = {hidden_.data() + elements - kHiddenWidth, logits_.data()};
#endif
        return token;
    }

    // The score row a screened pick skipped: recomputed on the first score read (read_slot_logits, e.g. a sampled
    // served lane) from the head input that pick used (still in hidden_: no forward ran since), through the same
    // project_head the unscreened pick runs, so the reader gets the scores of this step, never those of an earlier one.
    void materialize_scores() {
        if (pending_scores_.input == nullptr) return;
        const auto pending = pending_scores_;
        pending_scores_ = {};
        project_head(pending.input, std::span<float>(pending.scores, kOutputVocabulary), true);
    }

    static unsigned admitted_tail_rows(unsigned chunk_rows, unsigned tail_rows) noexcept {
#if defined(__HIPCC__)
        if (tail_rows == 0U) return 0U;
        return mini_prefill_tail_rows(chunk_rows, tail_rows, MiniResidentBackend::admits_chunk_rows(chunk_rows + tail_rows));
#else
        (void)chunk_rows;
        (void)tail_rows;
        return 0U;
#endif
    }

    unsigned prefill_rows_, tail_rows_, decode_rows_, max_rows_;
    MiniSessionStatePlan layout_;
    MiniResidentWeightPlan weight_plan_;
    mini_gpu::StreamOwner stream_;
    mini_gpu::Buffer<std::uint8_t> weight_storage_;
    mini_gpu::Buffer<float> activations_, history_backup_;
    mini_gpu::Buffer<std::uint16_t> kv_backup_;
    mini_gpu::Buffer<float> embeddings_, hidden_, logits_;
    std::optional<mini_gpu::Buffer<std::uint16_t>> splice_staging_;  // allocated by the first provided-rows prefill
    mini_gpu::Buffer<std::uint16_t> draft_taps_;
#if !defined(__HIPCC__)
    mini_gpu::Buffer<std::uint32_t> token_ids_;
#endif
    MiniResidentWeights weights_;
    MiniResidentBackend backend_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<MiniResidentAppendGroup> groups_;
    core::ResidentGreedy greedy_;
    std::optional<mini_gpu::Buffer<float>> verify_logits_;
    std::optional<core::ResidentGreedy> verify_greedy_;
    // Head input and score row of the last pick that skipped its scores (materialize_scores).
    struct PendingScores {
        const float* input = nullptr;
        float* scores = nullptr;
    } pending_scores_;
#if defined(__HIPCC__)
    std::optional<core::hip_screened_head::ResidentScreen> screen_;
    bool screened_graph_{false};
#endif
    std::array<unsigned, kMaxVerifyRows> verify_ids_{}, verify_top_{};
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits_;
    std::size_t staged_logits_size_{};
    unsigned last_logit_rows_{};
#if !defined(__HIPCC__)
    exec::CohortProgramPolicy cohort_policy_{exec::resolve_cohort_programs()};
    unsigned cohort_context_{};
    std::vector<state::SlotEntry> slot_directory_host_;
    mini_gpu::Buffer<state::SlotEntry> slot_directory_{std::size_t{kMiniLaneStates} * decode_rows_};
    mini_gpu::Buffer<std::uint32_t> lane_ids_{exec::kStepMaxRows};
    mini_gpu::Buffer<std::uint32_t> lane_anchor_{1U};
    std::vector<exec::CohortLane> cohort_lanes_;
    std::vector<MiniResidentAppendGroup> padded_groups_;
    std::bitset<exec::kStepMaxRows> reported_cohort_replays_;
    std::bitset<exec::kStepMaxRows> reported_padded_cohort_replays_;
#endif
};

}  // namespace strix::inkling2

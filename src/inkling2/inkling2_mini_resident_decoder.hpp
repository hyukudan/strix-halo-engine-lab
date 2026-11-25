#pragma once

#include "core/resident_greedy.hpp"
#include "core/weight_slice_loader.hpp"
#include "inkling2_mini_resident_backend.hpp"
#include "inkling2_mini_schedule.hpp"
#include "inkling2_mini_weight_options.hpp"
#include "strix/profiles/inkling2_mini.hpp"

#if defined(__HIPCC__)
#include "core/gemv_rows.hpp"
#include "core/hip_screened_head.hpp"
#include "core/hip_weight_uploader.hpp"
#else
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/head_screened.cuh"
#include "cuda/core/weight_uploader.hpp"
#endif

#include "strix/exec/cohort_programs.hpp"
#include "strix/head_argmax.hpp"
#include "strix/logit_rows.hpp"
#include "strix/serve/cohort_passes.hpp"
#include "strix/session/embed_splice_device.hpp"

#if !defined(__HIPCC__)
#include "cuda/core/step_inputs.cuh"
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <iostream>
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

// The resident upload of the Mini text weights into `storage` (plan.bytes() long) and the views bound over it: the one
// the decoder runs, shared with the operand-class fixture so that nothing in it is a second copy. On CUDA the staged
// attention tensors are then certified (the declaration the attention linears read) through the shared bounded scan:
// in chunks, no host copy of a whole tensor; no scan scratch leaves the certificate empty (three planes) and the load
// complete. The HIP load path is unchanged.
[[nodiscard]] inline MiniResidentWeights upload_mini_resident_weights(const MiniResidentWeightPlan& plan,
                                                                      std::span<std::uint8_t> storage,
                                                                      [[maybe_unused]] mini_gpu::Stream stream) {
#if defined(__HIPCC__)
    core::HipWeightUploader uploader;
#else
    strix::cuda::core::WeightUploader uploader;
#endif
    core::WeightSliceLoader loader(uploader, {});
    auto weights = MiniResidentWeights::bind(plan, storage);
    loader.load_resident(plan.slices(), storage);
#if !defined(__HIPCC__)
    core::Bf16StagedScan scan;
    weights.certify_attention(scan, [&](std::span<const std::uint16_t> staged, std::span<std::uint16_t> host) {
        mini_gpu::copy(host.data(), staged.data(), staged.size_bytes(), mini_gpu::device_to_host, stream);
        mini_gpu::synchronize(stream);
    });
#endif
    return weights;
}

// One unpadded token stream, bound to the selected device and stream for its
// lifetime. All checkpoint tensors and live cache slots stay on the device.
// The owner is serial. Calls return after the stream and state commit finish.
class MiniResidentDecoder {
public:
    static constexpr unsigned kMaxVerifyRows = kMiniMaxVerifyRows;
#if defined(__HIPCC__)
    // Rows of one packed cohort decode pass (served route lever, decode_slots_served): the HIP dense rows family keeps
    // its one-row body up to gemv_rows::max_rows rows and runs the tile GEMM past it.
    static constexpr unsigned kCohortPassRows = core::gemv_rows::max_rows;
#endif
    // Multi-row score buffers (cohort lanes, verify rows) start every row on the shared head pitch.
    static constexpr std::size_t kLogitPitch = ::strix::head::logit_row_pitch(kOutputVocabulary);
    static_assert(kLogitPitch >= kOutputVocabulary &&
                  kLogitPitch * sizeof(float) % ::strix::head::kLogitRowAlignmentBytes == 0U,
                  "every score row starts on the shared head alignment");
    using Rows = std::span<const unsigned>;
    // all: the window's picks; scores: some reader (EngineVerifyTarget's observer, sampled verification) also reads its
    // rows' final scores. Only the verify rows lever's level 2 tells them apart (a lanes screened pick writes no scores).
    enum class LogitRows { all, scores };
    // The resident EXACT screened greedy head of this backend (same plan, core/screened_head_plan.hpp).
#if defined(__HIPCC__)
    using Screen = core::hip_screened_head::ResidentScreen;
#else
    using Screen = ::strix::cuda::core::head_screened::ResidentScreen;
#endif

    // tail_rows: prefill() lets its final chunk take up to tail_rows rows beyond max_chunk_rows when that keeps the
    // bits of the greedy schedule (mini_prefill_tail_rows: the tail takes the same route forms in both schedules and
    // the enlarged chunk passes the device admission); otherwise the tail is dropped and the greedy chunks run. The
    // device chunk capacity is max_chunk_rows + the admitted tail. Only the HIP routes are qualified; CUDA keeps
    // greedy chunks.
    // served_route (the served route lever, kMiniServedCliRoute; HIP admits it, CUDA declines): the decoder of a
    // server that runs the command line's route. Only then tail_rows is a capacity (the largest admitted tail up to
    // it, admitted_tail_capacity) from which each served or command-line call absorbs its actual remainder
    // (call_tail_rows), and cohorts run packed passes (decode_slots_served). Without it every path is the incumbent one.
    MiniResidentDecoder(const std::string& checkpoint, unsigned max_positions, unsigned max_chunk_rows,
                        bool packed_decode = false, bool step_graphs = MiniResidentBackend::default_step_graphs,
                        weight::RoleStorage storage = {}, unsigned verify_rows = 0U,
                        bool capture_draft_taps = false, unsigned decode_rows = 1U, unsigned tail_rows = 0U,
                        bool narrow_experts = false, bool controller_one_plane = false, bool served_route = false,
                        MiniLowBitRows lowbit = {})
        : prefill_rows_(MiniResidentBackend::checked_chunk_rows(max_chunk_rows)),
          tail_rows_((admits_served_route(served_route) ? admitted_tail_capacity : admitted_tail_rows)(
              prefill_rows_, tail_rows <= std::numeric_limits<unsigned>::max() - prefill_rows_
                  ? tail_rows : throw std::invalid_argument("Inkling 2 Mini: invalid prefill tail"))),
          decode_rows_(MiniResidentBackend::checked_chunk_rows(decode_rows)),
          max_rows_(std::max({MiniResidentBackend::checked_chunk_rows(prefill_rows_ + tail_rows_),
                              std::min(kMaxVerifyRows, verify_rows), decode_rows_})),
          layout_(max_positions, max_rows_), weight_plan_(read_plan(checkpoint, storage, lowbit, narrow_experts)),
          weight_storage_(weight_plan_.bytes()), narrow_storage_(weight_plan_.narrow_bytes()),
          activations_(mini_resident_activation_elements(max_rows_)),
          history_backup_(session::cache_buffer_elements<float>(layout_.convolution_elements())),
          kv_backup_(mini_resident_max_kv_checkpoint_elements()), embeddings_(std::size_t{max_rows_} * kHiddenWidth),
          hidden_(embeddings_.span().size()), logits_(std::size_t{decode_rows_} * kLogitPitch),
          draft_taps_(capture_draft_taps ? MiniDSparkTapPlan::elements(max_positions) : 0U),
#if !defined(__HIPCC__)
          token_ids_(max_rows_),
#endif
          weights_(upload_weights(controller_one_plane)), backend_(weights_, max_rows_, stream_.get(), packed_decode, step_graphs,
                                                     admits_served_route(served_route)),
          served_cli_route_(admits_served_route(served_route)),
          calibrated_decode_(lowbit.admitted != nullptr && lowbit.decode_only), greedy_(decode_rows_, kOutputVocabulary) {
#if defined(__HIPCC__)
        // EXACT screened greedy head (core/hip_screened_head.hpp) over the BF16 head: same token and status as the
        // full head. A compact head keeps its own full head.
        if ((!weights_.compact_head_bound() || weights_.bf16_bound()) && core::hip_screened_head::enabled())
            emplace_screen(1U);
        head_ = {screen_.has_value(), MiniHeadReason::hip_binding};
#endif
#if !defined(__HIPCC__)
        cohort_context_ = max_positions;
#endif
        allocate_slots(1U);
        reset_gate_session();
    }

    ~MiniResidentDecoder() {
        backend_.abandon_capture();
        // The run's witness of the lever's screened route (0: no greedy pick ran screened).
        if (head_.reason == MiniHeadReason::lever) std::cerr << "inkling2 head screened picks " << screened_picks_ << '\n';
    }

    // Screened-head lever (mini_screened_head), set once after construction through configure_mini_decoder (or the
    // CUDA step program's load). CUDA: selected -> the EXACT screened head (cuda/core/head_screened.cuh) over the BF16
    // head is built now; a compact (Q8_0) head or a plane that cannot be built keeps the full head. Greedy picks that
    // return no score row then run it; host logits, score reads (materialized on the first read), verify rows and
    // cohort programs keep the full head. HIP keeps its own binding (built with the decoder unless
    // STRIX_SCREENED_HEAD=0) and ignores the lever. Returns the effective head; prints the "inkling2 head" line.
    MiniHead set_screened_head(bool selected) {
#if defined(__HIPCC__)
        static_cast<void>(selected);
#else
        backend_.abandon_capture();
        screened_graph_ = false;
        screen_.reset();
        head_ = mini_select_head(selected, [&] { return build_screen(); });
#endif
        std::cerr << "inkling2 head " << mini_head_field(head_);
        if (head_.screened && screen_) std::cerr << " plane_mib " << (screen_->device_bytes() >> 20U);
        std::cerr << '\n';
        return head_;
    }
    [[nodiscard]] MiniHead head() const noexcept { return head_; }
    // Device bytes of the plane the lever built (0: none; HIP's own binding is not the lever's and reports 0), for the
    // served slot budget (mini_serve_kv_slots): the plane stays only when it costs no KV slot.
    [[nodiscard]] std::uint64_t screen_device_bytes() const noexcept {
        return head_.reason == MiniHeadReason::lever && screen_ ? screen_->device_bytes() : 0U;
    }
    // Frees the lever's plane before the first served pick (the full head picks the same tokens).
    void release_screen() {
        if (head_.reason != MiniHeadReason::lever) return;
        backend_.abandon_capture();
        screened_graph_ = false;
        screen_.reset();
        head_ = {false, MiniHeadReason::kv_slot};
        std::cerr << "inkling2 head " << mini_head_field(head_) << '\n';
    }
    // Greedy picks that ran screened so far (both routes; the CUDA step program counts its screened steps here).
    [[nodiscard]] std::uint64_t screened_picks() const noexcept { return screened_picks_; }
    // Verify rows level 2: multi-row picks that ran through the lanes screened head (cohorts and verify windows), and the
    // most rows one such pick screens (0 when the lanes pick is not admitted); wider ones keep the full head.
    [[nodiscard]] std::uint64_t lanes_picks() const noexcept { return lanes_picks_; }
    [[nodiscard]] unsigned lanes_pick_capacity() const noexcept {
#if defined(__HIPCC__)
        return verify_lanes_pick_ && screen_ ? screen_->max_lanes() : 0U;
#else
        return 0U;
#endif
    }

    // Return the token after the last prompt input. Only that row goes through
    // the output projection. If a later chunk fails after an earlier one
    // committed, retire the session so a retry cannot append the prefix twice.
    [[nodiscard]] unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits = {}) {
        backend_.abandon_capture();
        auto& state = slots_.front()->state();
        validate_inputs(state, prompt, logits);
        return state.append_sequence(prompt, [&](std::span<const unsigned> chunk, bool last) {
            return append(state, chunk, last, last ? logits : std::span<float>{});
        }, prefill_rows_, served_cli_route_ ? call_tail_rows(prompt.size()) : tail_rows_);
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
        }, prefill_rows_, served_cli_route_ ? call_tail_rows(prompt.size()) : tail_rows_);
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
    [[nodiscard]] std::uint64_t slot_bytes() const {
        return layout_.kv_bytes() + layout_.convolution_bytes() + slot_tape_elements() * sizeof(float);
    }

    void allocate_slots(unsigned slots) {
        if (slots == 0U || slots > decode_rows_) throw std::invalid_argument("invalid Mini slot count");
        slots_.reserve(slots);
        while (slots_.size() < slots) slots_.push_back(std::make_unique<Slot>(layout_));
        bind_slot_tapes();
#if !defined(__HIPCC__)
        publish_slot_directory();
#endif
    }

    // Served verify windows (serve::FixedKvCohort): set once after construction and configure_mini_decoder, before
    // the slots are allocated. On, a served lane's group may carry drafts: the window is staged on the lane's slot with
    // the command line's verify arithmetic (append, MiniAppendKind::verify: every row on its one-row decode body), and
    // under the verify rows lever every slot binds its own convolution tape (a partial commit is copies, no replay).
    // One window is staged at a time (the decoder's one checkpoint workspace). Off: one row per lane, as before.
    void set_slot_windows(bool enabled) {
        if (slots_.size() > 1U) throw std::logic_error("set slot windows before allocating slots");
        slot_windows_ = enabled && max_verify_rows() > 1U;
    }
    [[nodiscard]] std::uint32_t max_slot_window() const { return slot_windows_ ? max_verify_rows() : 1U; }
    [[nodiscard]] bool slot_window_alone() const noexcept { return true; }

    void stage_slot_window(std::uint32_t slot, std::span<const std::uint32_t> ids, std::span<std::uint32_t> top) {
        if (!slot_windows_ || ids.size() < 2U || ids.size() > max_verify_rows() || top.size() != ids.size())
            throw std::invalid_argument("invalid Mini served verify window");
        // An idle slot holds no state to inspect.
        if (staged_window_) throw std::logic_error("Mini served verify: one staged window at a time");
        auto& state = slots_.at(slot)->state();
        validate_inputs(state, ids, {});
        last_logit_rows_ = 0U;  // the window's score rows are not a decode_slots window
        const auto picks = std::span(verify_top_).first(ids.size());
        (void)append(state, ids, true, {}, MiniAppendKind::verify, picks);
        staged_window_ = slot;
        std::ranges::copy(picks, top.begin());
    }

    // Keeps the window's leading `keep` rows (0 discards it); `ids` are the staged window's tokens.
    void commit_slot_window(std::uint32_t slot, std::span<const std::uint32_t> ids, std::uint32_t keep) {
        auto& state = slots_.at(slot)->state();
        if (staged_window_ != slot || !state.staged() || keep > ids.size())
            throw std::logic_error("invalid Mini served verify commit");
        staged_window_.reset();
        state.commit_staged(backend_, keep, [&](unsigned kept) {
            (void)append(state, ids.first(kept), kept == 1U, {}, MiniAppendKind::verify);
        });
    }

    // Served prompt route (serve::FixedKvCohort reads it once, at its construction): emits-first (prefill_slot_first)
    // where the decoder was built with the served route lever and the backend admits it (HIP), or while a W4A4 expert
    // class is bound (the final prompt token is then a prefill row on both entry points, so a served lane quantises
    // exactly the prompt rows of the command line); else the incumbent held-back route (prefill_slot).
    // A calibrated decode-only image (the TURBO low-bit row) takes the same form: the final prompt token is a BF16
    // prefill row on both entry points, never a compact decode row on the server alone (mini_served_prompt_emits_first).
    [[nodiscard]] bool prefill_emits_first() const noexcept {
        return mini_served_prompt_emits_first(served_cli_route_, backend_.moe_w4a4_bound(), calibrated_decode_);
    }
    // The decode expert form this decoder was built with (packed or grouped), for receipts and fixtures.
    [[nodiscard]] bool packed_decode() const noexcept { return backend_.packed_decode(); }

    // Held-back served prefill (the CUDA served route): a lane's prompt rows by greedy chunks with no score row; the
    // cohort holds back the last prompt token and decodes it as the lane's first decode row.
    void prefill_slot(unsigned slot, std::span<const unsigned> ids, bool first) {
        if (first) reset_slot(slot);
        auto& state = slots_.at(slot)->state();
        validate_inputs(state, ids, {});
        (void)state.append_sequence(ids, [&](std::span<const unsigned> chunk, bool) {
            return append(state, chunk, false, {});
        }, prefill_rows_);
    }

    // Emits-first served prefill (the HIP served route): a lane's prompt rows by the same chunks and per-call tail rule
    // as prefill() (the command line), the first id from the final row of the call's last chunk; its score row is
    // read_slot_logits(0) (materialized on the first read when the pick was screened). The cohort issues a whole prompt
    // as calls of whole chunks, or, beside a decoding lane, as pieces cut inside those chunks by prefill_piece_rule()
    // only, so a lane's prompt bits never depend on other lanes.
    [[nodiscard]] unsigned prefill_slot_first(unsigned slot, std::span<const unsigned> ids, bool first) {
        if (first) reset_slot(slot);
        auto& state = slots_.at(slot)->state();
        validate_inputs(state, ids, {});
        last_logit_rows_ = 0U;
        const auto token = state.append_sequence(ids, [&](std::span<const unsigned> chunk, bool last) {
            return append(state, chunk, last, {});
        }, prefill_rows_, served_cli_route_ ? call_tail_rows(ids.size()) : tail_rows_);
        last_logit_rows_ = 1U;
        return token;
    }
    // Admitted tail capacity (serve::CohortBackend::prompt_tail_rows), the cut rule of served pieces and the first
    // chunk of one prefill call (prompt_piece_rule / prompt_call_chunk).
    [[nodiscard]] unsigned prefill_tail_rows() const noexcept { return tail_rows_; }
    // Pieces cut only on the command line's route (the held-back route runs its incumbent whole slices).
    [[nodiscard]] serve::PromptPieceRule prefill_piece_rule() const noexcept {
        return mini_prefill_piece_rule(served_cli_route_);
    }
    [[nodiscard]] std::uint64_t prefill_call_chunk_rows(std::uint64_t rows) const noexcept {
        if (!served_cli_route_) return rows;  // the incumbent route: one call is one slice
        return mini_call_first_chunk(rows, prefill_rows_, call_tail_rows(static_cast<std::size_t>(rows)));
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
    // The lever's screened head (nullptr: the full head) and the count of a step the program picked through it.
    [[nodiscard]] const Screen* step_program_screen() const noexcept {
        return head_.reason == MiniHeadReason::lever && screen_ ? &*screen_ : nullptr;
    }
    void step_program_screened_pick() { note_screened_pick(); }
#endif

    // One forward for all lanes. Every row keeps single-row decode arithmetic,
    // so a lane matches its own decode() stream. With DSpark tap capture on,
    // lanes step one at a time so each committed position records its taps.
    // reads_scores (CohortGroup's contract, default true = read): some lane may read its final scores (read_slot_logits)
    // after this call. Only the verify rows lever's level 2 uses it: lanes whose scores nobody reads pick through the
    // lanes form of the screened head (verify_lanes_pick_), an announced read keeps the full head.
    void decode_slots(std::span<const unsigned> slots, std::span<const unsigned> ids, std::span<unsigned> top,
                      bool reads_scores = true) {
#if defined(__HIPCC__)
        // Served route lever (the decoder was built with it): balanced packed passes, decode_slots_served below. Without
        // the lever every cohort takes the incumbent body that follows.
        if (served_cli_route_) {
            decode_slots_served(slots, ids, top);
            return;
        }
#endif
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
                top[row] = append(*groups_[row].state, ids.subspan(row, 1U), true, {}, MiniAppendKind::decode, {}, nullptr,
                                  reads_scores);
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
        // Verify rows level 2: lanes whose scores nobody reads pick through the lanes form of the EXACT screened head
        // (one plane read for the cohort, each lane with its one-row token and status). Their score rows are recomputed
        // on a read all the same (materialize_scores): the lanes' head inputs stay in hidden_.
#if defined(__HIPCC__)
        // Cohort weights follow decode admission even when the packed kernels use verify geometry.
        const bool lanes_screened = !backend_.lowbit_applies(MiniAppendKind::decode, rows) &&
                                    verify_lanes_pick_ && !reads_scores && rows > 1U && rows <= screen_->max_lanes();
#endif
        MiniResidentForward forward(backend_, workspace());
        forward.append_batch(groups_, rows, embeddings, hidden, [&] {
            embed(ids, embeddings, rows, kind, rows);
        }, [&](MiniConstRows output) {
#if defined(__HIPCC__)
            if (lanes_screened) {
                greedy_.enqueue_produced_rows(rows, [&](HipGreedyRowsReply* replies) {
                    return screen_->enqueue_lanes(output.data(), rows, kHiddenWidth, replies, stream_.get());
                });
                return;
            }
#endif
            for (unsigned row = 0; row < rows; ++row)
                project_head(output.data() + std::size_t{row} * kHiddenWidth,
                             logits.subspan(::strix::head::logit_row_offset(row, kLogitPitch), kOutputVocabulary), true);
            greedy_.enqueue_rows(logits, rows, kLogitPitch, stream_.get());
        }, [&] {
            for (std::size_t row = 0; row < top.size(); ++row) (void)checked_token(greedy_, row);
        });
        for (std::size_t row = 0; row < top.size(); ++row) top[row] = greedy_.token(row);
#if defined(__HIPCC__)
        if (lanes_screened) {
            pending_scores_ = {hidden_.data(), rows, logits.data()};
            ++lanes_picks_;
        }
#endif
        last_logit_rows_ = rows;
    }

#if defined(__HIPCC__)
    // Served route lever only (decoder built with served_route; HIP): packed decode in balanced passes of at most
    // kCohortPassRows lanes, every dense row on its one-row body and the pitched rows head (project_head_rows), so a
    // lane's bits do not depend on the cohort width. The whole cohort is checked before the first pass writes;
    // serve::run_cohort_passes copies the input ids before any pass (ids and top may overlap) and publishes the picks
    // only after every pass finished: a pass that throws retires every lane of the cohort and top stays untouched.
    void decode_slots_served(std::span<const unsigned> slots, std::span<const unsigned> ids, std::span<unsigned> top) {
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
        pass_ids_.assign(ids.begin(), ids.end());  // ids and top may overlap: read the inputs from the copy
        if (groups_.size() == 1U || draft_taps_.span().size() != 0U) {
            for (std::size_t row = 0; row < slots.size(); ++row)
                top[row] = append(*groups_[row].state, std::span<const unsigned>(pass_ids_).subspan(row, 1U), true, {},
                                 MiniAppendKind::decode);
            // The head scores of a lone lane stay in row 0. Several lanes stepped one at a time overwrite it.
            last_logit_rows_ = slots.size() == 1U ? 1U : 0U;
            return;
        }

        // Lane r's head scores land at logit row r for any pass split.
        const auto lanes = static_cast<unsigned>(ids.size());
        MiniResidentSessionState::require_batch(groups_);
        ::strix::serve::run_cohort_passes(ids, top, mini_cohort_pass_cap(backend_.packed_decode(), kCohortPassRows),
                                          pass_ids_, pass_picks_,
                                          [&](unsigned first, unsigned rows, std::span<const unsigned> pass_ids,
                                              std::span<unsigned> picks) {
            const auto embeddings = embeddings_.span().first(std::size_t{rows} * kHiddenWidth);
            const auto hidden = hidden_.span().first(embeddings.size());
            const auto scores = logits_.span().subspan(::strix::head::logit_row_offset(first, kLogitPitch));
            const auto logits = scores.first(::strix::head::logit_rows_elements(rows, kOutputVocabulary, kLogitPitch));
            // Verify routing gives each packed row the decode MoE and attention arithmetic.
            const auto kind = rows <= kMaxVerifyRows ? MiniAppendKind::verify : MiniAppendKind::decode;
            MiniResidentForward forward(backend_, workspace());
            forward.append_batch(std::span<const MiniResidentAppendGroup>(groups_).subspan(first, rows), rows,
                                 embeddings, hidden, [&] {
                embed(pass_ids, embeddings, rows, kind, lanes);
            }, [&](MiniConstRows output) {
                project_head_rows(output.data(), rows, scores.first(std::size_t{rows} * kLogitPitch));
                greedy_.enqueue_rows(logits, rows, kLogitPitch, stream_.get());
            }, [&] {
                for (unsigned row = 0; row < rows; ++row) (void)checked_token(greedy_, row);
            });
            for (unsigned row = 0; row < rows; ++row) picks[row] = greedy_.token(row);
        }, [&] {
            for (const auto& group : groups_) group.state->poison();
            last_logit_rows_ = 0U;
        });
        last_logit_rows_ = lanes;
    }
#endif

#if !defined(__HIPCC__)
    // Effective cohort program policy and the programs installed so far (receipts / effective config).
    [[nodiscard]] const exec::CohortProgramPolicy& cohort_programs() const noexcept { return cohort_policy_; }
    [[nodiscard]] const exec::WidthStepGraphs& cohort_graphs() const noexcept { return backend_.cohort_graphs(); }
    // A cohort program of this width (padded or exact rows) has completed at least one replay (the stderr receipt's bit).
    [[nodiscard]] bool cohort_replay_completed(unsigned width) const noexcept {
        return width != 0U && width <= exec::kStepMaxRows &&
               (reported_cohort_replays_.test(width - 1U) || reported_padded_cohort_replays_.test(width - 1U));
    }
    // Completed cohort program replays so far, padded (fewer lanes than the program width) or full-width: a caller reads
    // it before and after a run to witness that THIS run replayed, in the mode its lane count implies.
    [[nodiscard]] std::uint64_t cohort_replay_count(bool padded) const noexcept {
        return padded ? padded_cohort_replays_ : full_cohort_replays_;
    }
#endif

    // Final-row scores of the last decode_slots() call, exactly as the greedy pick saw them (after the
    // output softcap). Lane `row` follows the slot order of that call. Valid until the next decode_slots().
    [[nodiscard]] bool final_logits_supported() const { return draft_taps_.span().empty(); }
    [[nodiscard]] std::uint32_t logit_width() const { return kOutputVocabulary; }
    // The last pick was screened and its score row waits for the first score read (materialize_scores): a greedy
    // caller that never reads scores never pays the full head projection.
    [[nodiscard]] bool scores_deferred() const noexcept { return pending_scores_.rows != 0U; }
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
    Rows forward_rows(const unsigned* ids, unsigned rows, LogitRows read, std::span<float> logits = {}) {
        if (rows == 0U || rows > max_verify_rows())
            throw std::invalid_argument("invalid Mini resident verify width");
        auto& state = slots_.front()->state();
        const Rows input{ids, rows};
        validate_inputs(state, input, {});
        if (!logits.empty() && logits.size() != std::size_t{rows} * kOutputVocabulary)
            throw std::invalid_argument("invalid Mini verify logits shape");
        auto top = std::span(verify_top_).first(rows);
        (void)append(state, input, true, logits, MiniAppendKind::verify, top, nullptr, read == LogitRows::scores);
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
    void set_calibration_moments(weight::calib::MomentSet* moments) { backend_.set_calibration_moments(moments); }
    void set_calibration_window(std::optional<std::size_t> window) { backend_.set_calibration_window(window); }
    void set_gate_order_null(bool enabled) { backend_.set_gate_order_null(enabled); }
    // Resolved prepared-attention round fold (mini_prepared_round_fold); set once after construction. True when the fold
    // runs (admitted on the qualified HIP target; declined elsewhere, CUDA included).
    bool set_prepared_round_fold(bool selected) { return backend_.set_prepared_round_fold(selected); }
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off); set once after
    // construction. No bool form: a bool would drop the class and run nearest@1 for mse / mse-gate-up.
    void set_moe_w4a4(moe_core::grouped_w4a4::Activations activations) { backend_.set_moe_w4a4(activations); }
    // Resolved exact down-image lever (mini_combine_image); set once after construction. True when the image route is
    // active (HIP, qualified target); every entry point that resolves the profile passes the resolved state.
    bool set_combine_image(bool selected) { return backend_.set_combine_image(selected); }
    // Resolved projected-bias attention lever (mini_attn_projected_bias); set once after construction.
    void set_attention_projected_bias(bool enabled) { backend_.set_attention_projected_bias(enabled); }
    // Resolved verify rows lever (mini_verify_rows, kMiniVerifyRows); set once after construction, before any append.
    // Admitted on HIP when verify windows of two or more rows exist: a window's score rows take the pitched rows head
    // (project_head_rows), its attention one lanes launch set, and slot 0 records the convolution inputs of a staged
    // window (the shared session tape), so a partial commit rebuilds the histories by copies instead of replaying the
    // kept rows. Declined on CUDA (0): every window keeps the per-row incumbent. Level 2 (kMiniVerifyLanesPick) adds the
    // lanes pick (verify_lanes_pick_): admitted on HIP when the BF16 screened head exists and its lanes form screens two
    // or more rows on this device; otherwise the effective level stays 1. Returns the effective level.
    unsigned set_verify_rows(unsigned level) {
#if defined(__HIPCC__)
        const bool effective = level != 0U && max_verify_rows() > 1U;
        const bool lanes_pick = effective && level >= 2U && admit_lanes_pick();
        verify_lanes_pick_ = lanes_pick;
#else
        static_cast<void>(level);
        const bool effective = false;
        const bool lanes_pick = false;
#endif
        const auto admitted = effective ? (lanes_pick ? 2U : 1U) : 0U;
        if (effective == verify_rows_) return admitted;
        auto& slot = *slots_.front();
        if (effective) {
            if (!conv_tape_)
                conv_tape_.emplace(static_cast<std::size_t>(max_verify_rows() * slot.convolution_row_elements()));
            slot.bind_convolution_tape(conv_tape_->span(), max_verify_rows());
        } else {
            slot.bind_convolution_tape({}, 0U);
        }
        backend_.set_verify_lanes(effective);
        verify_rows_ = effective;
        return admitted;
    }
    [[nodiscard]] std::uint64_t projected_attention_dispatches() const { return backend_.projected_attention_dispatches(); }
    [[nodiscard]] std::uint64_t gate_null_applications() const { return backend_.gate_null_applications(); }

    void prefill_gate_rows(std::span<const unsigned> prompt, std::span<float> logits) {
        backend_.abandon_capture();
        settle_pending_scores(false);
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
        pending_scores_ = {};  // the cohort rewrites hidden_ and every score row it describes (last_logit_rows_)
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
            ++(rows < width ? padded_cohort_replays_ : full_cohort_replays_);
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

    void embed(std::span<const unsigned> ids, MiniMutableRows output, unsigned rows, MiniAppendKind kind,
               unsigned decode_cohort_width = 0U) {
        const auto table = weights_.bf16("embed.weight");
        const auto gain = weights_.bf16("embed_norm.weight");
        // During a step-graph capture the token comes from the device descriptor.
        const auto* step = backend_.capture_descriptor();
        backend_.start_chunk(rows, kind, decode_cohort_width);
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

    [[nodiscard]] static MiniResidentWeightPlan read_plan(const std::string& path, weight::RoleStorage storage,
                                                          const MiniLowBitRows& lowbit, bool narrow_experts) {
        MiniTextWeights source;
        const auto status = source.open(path);
        if (!status) throw std::runtime_error(status.detail);
        const auto* pack = lowbit.admitted.get();
        auto plan = MiniResidentWeightPlan::make(source, storage, MiniResidentBackend::resident_row_pitch_rule(),
                                                 narrow_experts, pack, lowbit.decode_only);
        if (pack) {
            std::cerr << "strix-lowbit-pack/v2 calibrated identity " << hex_prefix(pack->identity(), 32U)
                      << " matrices " << plan.prepacked() << '/' << pack->entries().size();
            if (lowbit.decode_only) std::cerr << " image bf16+lowbit decode widths 1,8,16 (prefill bf16)";
            std::cerr << '\n';
        }
        return plan;
    }

    [[nodiscard]] static std::string hex_prefix(const weight::Digest256& digest, std::size_t bytes = 8U) {
        std::string out;
        for (std::size_t i = 0; i < bytes && i < digest.bytes.size(); ++i) {
            out += "0123456789abcdef"[digest.bytes[i] >> 4U];
            out += "0123456789abcdef"[digest.bytes[i] & 15U];
        }
        return out;
    }

    [[nodiscard]] MiniResidentWeights upload_weights(bool controller_one_plane) {
        auto weights = upload_mini_resident_weights(weight_plan_, weight_storage_.span(), stream_.get());
#if defined(__HIPCC__)
        static_cast<void>(controller_one_plane);
#else
        // CUDA certifies the staged controller weights rather than re-reading the checkpoint.
        if (controller_one_plane)
            weights.certify_controller([&](std::span<const std::uint16_t> staged, std::span<std::uint16_t> host) {
                mini_gpu::copy(host.data(), staged.data(), staged.size_bytes(), mini_gpu::device_to_host, stream_.get());
                mini_gpu::synchronize(stream_.get());
            });
#endif
        if (weight_plan_.narrow_experts()) upload_narrow_experts(weights);
        return weights;
    }

    // Narrow expert image: each MXFP8 slab of the checkpoint is transcoded to NVFP4 on the host (shared
    // weight/formats/microscaled_encode.hpp), copied into narrow_storage_ and bound; the receipt line states the
    // transcode error against the checkpoint values and its cost.
    void upload_narrow_experts(MiniResidentWeights& weights) {
        namespace ms = weight::microscaled;
        const auto start = std::chrono::steady_clock::now();
        ms::PackError total;
        unsigned slabs = 0;
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
            if (!MiniResidentWeightPlan::narrowed(layer)) continue;
            for (const auto projection : {MiniExpertProjection::gate_up, MiniExpertProjection::down}) {
                ms::Nvfp4Slab slab;
                if (const auto status = ms::transcode_slab_to_nvfp4(weight_plan_.expert_slab(layer, projection), slab);
                    !status)
                    throw std::runtime_error("Inkling 2 Mini: narrow expert transcode: " + status.detail);
                const auto range = weight_plan_.narrow_range(layer, projection);
                auto* codes = narrow_storage_.data() + range.codes;
                auto* scales = narrow_storage_.data() + range.scales;
                mini_gpu::copy(codes, slab.codes.data(), slab.codes.size(), mini_gpu::host_to_device, stream_.get());
                mini_gpu::copy(scales, slab.scales.data(), slab.scales.size(), mini_gpu::host_to_device, stream_.get());
                mini_gpu::synchronize(stream_.get());  // the host bytes end with this scope
                weights.attach_narrow(layer, projection, slab, {codes, slab.codes.size()}, {scales, slab.scales.size()});
                total.sse += slab.error.sse;
                total.energy += slab.error.energy;
                total.max_abs = std::max(total.max_abs, slab.error.max_abs);
                ++slabs;
            }
        }
        const auto ms_elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cerr << "strix-experts-nvfp4 slabs " << slabs << " relative_rms " << total.relative() << " max_abs "
                  << total.max_abs << " bytes " << weight_plan_.narrow_bytes() << " transcode_ms " << ms_elapsed << '\n';
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

#if defined(__HIPCC__)
    // Softcapped score rows of `rows` consecutive cohort lanes (decode_slots_served, served route lever only) at the
    // decoder's one row pitch: row r lands at logit_row_offset(r, kLogitPitch) for any row count. A BF16 head with more
    // than one row goes through the shared pitched rows head (core::enqueue_bf16_softcap_head_rows): up to
    // gemv_rows::max_rows rows a launch read the head once and keep every row on its one-row body (same chunk order and
    // wave sum), so the bits of the per-row launch; the pad outputs of a row are stored vocabulary rows nobody picks or
    // reads. A compact Q8_0 head takes the shared Q8_0 form of the same contract
    // (core::enqueue_resident_q8_softcap_head_rows). A single row takes the one-row head.
    void project_head_rows(const float* input, unsigned rows, std::span<float> scores) {
        if (rows == 0U || scores.size() / kLogitPitch < rows)
            throw std::invalid_argument("Inkling 2 Mini: head rows exceed the score rows");
        static_assert(kLogitPitch <= kVocabulary);
        if (weights_.compact_head_bound() && backend_.lowbit_chunk() && rows > 1U) {
            backend_.check(core::enqueue_resident_q8_softcap_head_rows(
                weights_.compact_head(), kOutputVocabulary, static_cast<unsigned>(kLogitPitch), input, rows,
                scores.first(std::size_t{rows} * kLogitPitch), kLogitSoftcap, stream_.get(), kMiniLowBitArithmetic));
            return;
        }
        if (rows > 1U) {
            const auto matrix = weights_.bf16("unembed.weight");
            backend_.check(core::enqueue_bf16_softcap_head_rows(
                rows, core::gemv_rows::max_rows, {kHiddenWidth, kVocabulary, kOutputVocabulary, kLogitSoftcap},
                static_cast<unsigned>(kLogitPitch), matrix, scores, stream_.get(),
                [&](unsigned first, unsigned launch, std::span<const MiniHipDenseProjection> projections) {
                    return enqueue_mini_hip_dense_rows(input + std::size_t{first} * kHiddenWidth, kHiddenWidth, launch,
                                                       projections, stream_.get());
                }));
            return;
        }
        project_head(input, scores.first(kOutputVocabulary), true);
    }
#endif

    // `rows` consecutive input rows (rows > 1 only without the softcap) into rows x vocabulary scores.
    void project_head(const float* input, std::span<float> row_scores, bool softcap, unsigned rows = 1U) {
        const auto stream = stream_.get();
        if (rows == 0U || (softcap && rows != 1U) || row_scores.size() / kOutputVocabulary < rows)
            throw std::invalid_argument("invalid head projection rows");
        if (weights_.compact_head_bound() && backend_.lowbit_chunk()) {
            const auto& head = weights_.compact_head();
            if (softcap)
                backend_.check(core::enqueue_resident_q8_softcap_head(head, kOutputVocabulary, input, 1U,
                                                                     row_scores, kLogitSoftcap, stream, kMiniLowBitArithmetic));
            else
                backend_.check(core::enqueue_resident_q8_rows(head, kOutputVocabulary, input, rows,
                                                             row_scores.data(), stream
#if !defined(__HIPCC__)
                                                             , nullptr
#endif
                                                             , kMiniLowBitArithmetic));
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
                                  std::span<unsigned> verify_top = {}, const session::EmbedSplice* splice = nullptr,
                                  bool scores_read = false) {
        static_assert(std::is_same_v<unsigned, std::uint32_t>);
        const auto rows = static_cast<unsigned>(ids.size());
        const auto elements = ids.size() * kHiddenWidth;
        const auto embeddings = embeddings_.span().first(elements);
        reserve_staged_logits(host_logits.size());
        settle_pending_scores(pick_next && verify_top.size() <= 1U);
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
        // A calibrated decode uses its admitted head for both single and lanes picks.
        const bool lowbit = backend_.lowbit_applies(kind, rows);
        // The single-pick screened head is HIP's own binding or the CUDA lever (CUDA verify rows stay full).
        // A captured step graph replays the head it captured, so a change of head drops the graph for recapture.
        bool need_scores = !host_logits.empty();
#if defined(__HIPCC__)
        // Level 2 keeps the full single-row head for announced score reads and screens unread verify windows in lanes.
        need_scores = need_scores || (verify_lanes_pick_ && scores_read);
        const bool lanes_screened = !lowbit && verify_lanes_pick_ && pick_next && multi_pick && !need_scores &&
                                    verify_top.size() <= screen_->max_lanes();
#else
        static_cast<void>(scores_read);  // only the HIP verify rows level 2 tells reads apart
#endif
        const bool screened = !lowbit &&
            mini_screened_pick(head_.reason, screen_.has_value(), kind, pick_next, multi_pick, need_scores);
        if (decode_step && screened != screened_graph_) {
            backend_.abandon_capture();
            screened_graph_ = screened;
        }
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
                    // Verify rows lever (HIP): the decoder's one pitched rows head projects every row of the window in
                    // one launch set, each row on its one-row body (project_head_rows; served cohort lanes bind the
                    // same head). A one-row window and the single pick stay in the loop below.
                    bool rows_head = false;
#if defined(__HIPCC__)
                    if (lanes_screened) {
                        greedy.enqueue_produced_rows(static_cast<unsigned>(head_rows), [&](HipGreedyRowsReply* replies) {
                            return screen_->enqueue_lanes(tail.data(), static_cast<unsigned>(head_rows), kHiddenWidth,
                                                          replies, stream_.get());
                        });
                        return;
                    }
                    if (verify_rows_ && multi_pick) {
                        project_head_rows(tail.data(), static_cast<unsigned>(head_rows), scores);
                        rows_head = true;
                    }
#endif
                    for (std::size_t row = 0; row < head_rows; ++row) {
                        const auto* input = tail.data() + row * kHiddenWidth;
                        if (screened) {
#if defined(__HIPCC__)
                            greedy.enqueue_produced([&](HipGreedyRowsReply* reply) {
                                backend_.check(screen_->enqueue(input, reply, stream_.get()));
                            });
#else
                            greedy.enqueue_produced([&](std::uint32_t* device_token) {
                                return screen_->enqueue(input,
                                    ::strix::cuda::core::head_argmax::EmitIndexOrNone{device_token}, stream_.get());
                            }, stream_.get());
#endif
                            continue;
                        }
                        const auto row_scores = scores.subspan(
                            ::strix::head::logit_row_offset(multi_pick ? static_cast<unsigned>(row) : 0U, kLogitPitch), kOutputVocabulary);
                        if (!rows_head) project_head(input, row_scores, true);
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
        if (screened) {
            // The screened pick skipped the score row; a later score read recomputes it from this head input.
            pending_scores_ = {hidden_.data() + elements - kHiddenWidth, 1U, logits_.data()};
            note_screened_pick();
        }
#if defined(__HIPCC__)
        if (lanes_screened) ++lanes_picks_;
#endif
        return token;
    }

    // Counts a greedy pick that ran screened; the lever's first one is reported once (the route ran, not only built).
    void note_screened_pick() {
        if (screened_picks_++ == 0U && head_.reason == MiniHeadReason::lever)
            std::cerr << "inkling2 head screened first pick\n";
    }

#if !defined(__HIPCC__)
    // The lever's plane over the BF16 head (true: built and usable). A compact (Q8_0) head has its own full head; a
    // plane that cannot be built (e.g. no device memory left for it) keeps the full head and says why.
    bool build_screen() {
        if (weights_.compact()) return false;
        try {
            screen_.emplace(weights_.bf16("unembed.weight").data(), kOutputVocabulary, kHiddenWidth, kLogitSoftcap,
                            stream_.get());
        } catch (const std::exception& error) {
            screen_.reset();
            static_cast<void>(cudaGetLastError());
            std::cerr << "inkling2 head screened plane not built: " << error.what() << '\n';
            return false;
        }
        if (screen_->usable()) return true;
        screen_.reset();
        return false;
    }
#endif

    // The score rows a screened pick skipped (the one-row pick, or a level-2 lanes pick of a cohort): recomputed on the
    // first score read (read_slot_logits, e.g. a sampled served lane) from the head inputs that pick used (still in
    // hidden_: no forward ran since), through the same per-row project_head the unscreened pick runs at the same pitched
    // rows, so the reader gets the scores of this step, never those of an earlier one.
    // The descriptor stays until every row is projected: a projection that throws leaves it pending, so a retried read
    // recomputes the rows from the same inputs instead of reading the score rows of an earlier step.
    void materialize_scores() {
        if (pending_scores_.rows == 0U) return;
        const auto pending = pending_scores_;
        for (unsigned row = 0; row < pending.rows; ++row)
            project_head(pending.input + std::size_t{row} * kHiddenWidth,
                         std::span<float>(pending.scores + ::strix::head::logit_row_offset(row, kLogitPitch),
                                          kOutputVocabulary), true);
        pending_scores_ = {};
    }

    // Before an operation overwrites hidden_ (the head inputs of a deferred pick): score rows that stay readable after it
    // (last_logit_rows_ still describes them: a multi-row verify writes its scores to verify_logits_, a prompt chunk
    // writes none) are materialized first, so a late read_slot_logits gets this step's bits exactly as with eager scores;
    // rows the operation itself rewrites (one deferred row and a one-row pick into row 0) or no longer describes
    // (last_logit_rows_ == 0) are dropped. The one-row screened pick followed by the next one-row pick never projects.
    void settle_pending_scores(bool rewrites_row0) {
        if (pending_scores_.rows == 0U) return;
        if (last_logit_rows_ == 0U || (rewrites_row0 && pending_scores_.rows == 1U)) {
            pending_scores_ = {};
            return;
        }
        materialize_scores();
    }

#if defined(__HIPCC__)
    // The BF16 EXACT screened head, its scratch sized for `lanes` rows per enqueue_lanes call (1: the one-row pick).
    void emplace_screen(unsigned lanes) {
        screen_.reset();
        screen_.emplace(weights_.bf16("unembed.weight").data(), kOutputVocabulary, kHiddenWidth, kLogitSoftcap,
                        stream_.get(), 8U, lanes);
        if (!screen_->usable()) screen_.reset();
        head_.screened = screen_.has_value();
    }

    // Verify rows level 2: rebuild the screened head with as many lanes as one pass-1 group holds on this device (the
    // plane is read once for all of them, screened_head_plan::pass1_lanes against the shared memory of the device the
    // screen launches on). Declined (the one-row screen kept) without a screened head or with fewer than two lanes.
    bool admit_lanes_pick() {
        if (!screen_) return false;
        int device = 0;
        mini_gpu::check(hipGetDevice(&device), "query the Mini head device");
        const auto lanes = core::hip_screened_head::pass1_lanes(kHiddenWidth, kMaxVerifyRows, device);
        if (lanes < 2U) return false;
        emplace_screen(lanes);
        if (screen_) return true;
        emplace_screen(1U);
        return false;
    }
#endif

    // Tail rows one prefill call of `rows` rows absorbs (mini_call_tail_rows within the admitted capacity).
    [[nodiscard]] unsigned call_tail_rows(std::size_t rows) const noexcept {
#if defined(__HIPCC__)
        return mini_call_tail_rows(rows, prefill_rows_, tail_rows_,
                                   [](unsigned enlarged) { return MiniResidentBackend::admits_chunk_rows(enlarged); });
#else
        (void)rows;
        return 0U;
#endif
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

    // Served route lever: the tail capacity a reusable served decoder reserves (the largest admitted tail up to
    // tail_rows); each prefill call then absorbs its actual remainder within it (call_tail_rows).
    static unsigned admitted_tail_capacity(unsigned chunk_rows, unsigned tail_rows) noexcept {
#if defined(__HIPCC__)
        return mini_admitted_tail_capacity(chunk_rows, tail_rows,
                                           [](unsigned enlarged) { return MiniResidentBackend::admits_chunk_rows(enlarged); });
#else
        (void)chunk_rows;
        (void)tail_rows;
        return 0U;
#endif
    }

    // The served route lever is admitted on HIP only.
    static constexpr bool admits_served_route(bool requested) noexcept {
#if defined(__HIPCC__)
        return mini_served_prefill_emits_first(true, requested);
#else
        return mini_served_prefill_emits_first(false, requested);
#endif
    }

    unsigned prefill_rows_, tail_rows_, decode_rows_, max_rows_;
    MiniSessionStatePlan layout_;
    MiniResidentWeightPlan weight_plan_;
    mini_gpu::StreamOwner stream_;
    mini_gpu::Buffer<std::uint8_t> weight_storage_;
    mini_gpu::Buffer<std::uint8_t> narrow_storage_;  // narrow expert image (empty without the lever)
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
    std::vector<unsigned> pass_ids_;    // decode_slots input snapshot
    std::vector<unsigned> pass_picks_;  // decode_slots picks, published after the last pass
    bool served_cli_route_;             // served route lever, fixed at construction (admits_served_route)
    bool calibrated_decode_;            // calibrated decode-only image bound (BF16 prefill rows, compact decode rows)
    core::ResidentGreedy greedy_;
    std::optional<mini_gpu::Buffer<float>> verify_logits_;
    std::optional<core::ResidentGreedy> verify_greedy_;
    // Verify rows lever (set_verify_rows): effective state and slot 0's convolution tape.
    bool verify_rows_{false};
    std::optional<mini_gpu::Buffer<float>> conv_tape_;
    // Served verify windows (set_slot_windows) and the convolution tapes of slots 1.. (slot 0 binds conv_tape_).
    bool slot_windows_{false};
    std::optional<std::uint32_t> staged_window_;  // the slot holding the one staged served window
    std::optional<mini_gpu::Buffer<float>> slot_tapes_;

    [[nodiscard]] std::uint64_t slot_tape_elements() const {
        return slot_windows_ && verify_rows_ ? std::uint64_t{max_verify_rows()} * layout_.convolution_row_elements() : 0U;
    }

    // Under served windows and the verify rows lever, every slot past slot 0 binds its own tape (slot 0: set_verify_rows).
    void bind_slot_tapes() {
        const auto per_slot = static_cast<std::size_t>(slot_tape_elements());
        if (per_slot == 0U || slots_.size() < 2U) return;
        const auto needed = per_slot * (slots_.size() - 1U);
        if (!slot_tapes_ || slot_tapes_->span().size() < needed) slot_tapes_.emplace(needed);
        for (std::size_t slot = 1; slot < slots_.size(); ++slot)
            slots_[slot]->bind_convolution_tape(slot_tapes_->span().subspan((slot - 1U) * per_slot, per_slot),
                                                max_verify_rows());
    }
    // Head inputs and pitched score rows of the last pick that skipped its scores (materialize_scores).
    struct PendingScores {
        const float* input = nullptr;
        unsigned rows = 0U;
        float* scores = nullptr;
    } pending_scores_;
    std::optional<Screen> screen_;
#if defined(__HIPCC__)
    bool verify_lanes_pick_{false};  // verify rows level 2 admitted (set_verify_rows): lanes screened picks
#endif
    bool screened_graph_{false};
    MiniHead head_{};                  // the effective greedy head (set_screened_head / release_screen)
    std::uint64_t screened_picks_{};  // greedy picks that ran screened (note_screened_pick)
    std::uint64_t lanes_picks_{};     // multi-row picks through the lanes screened head (lanes_picks)
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
    std::uint64_t full_cohort_replays_ = 0;
    std::uint64_t padded_cohort_replays_ = 0;
#endif
};

}  // namespace strix::inkling2

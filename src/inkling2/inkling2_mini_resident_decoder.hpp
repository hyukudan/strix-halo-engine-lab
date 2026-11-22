#pragma once

#include "core/resident_greedy.hpp"
#include "core/weight_slice_loader.hpp"
#include "inkling2_mini_resident_backend.hpp"

#if defined(__HIPCC__)
#include "core/hip_weight_uploader.hpp"
#else
#include "cuda/core/head_argmax.cuh"
#include "cuda/core/weight_uploader.hpp"
#endif

#include "strix/head_argmax.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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
    using Rows = std::span<const unsigned>;
    enum class LogitRows { all };

    MiniResidentDecoder(const std::string& checkpoint, unsigned max_positions, unsigned max_chunk_rows,
                        bool packed_decode = false, bool step_graphs = MiniResidentBackend::default_step_graphs,
                        weight::FormatId storage = weight::kBf16WeightStorage, unsigned verify_rows = 0U,
                        bool capture_draft_taps = false, unsigned decode_rows = 1U)
        : prefill_rows_(MiniResidentBackend::checked_chunk_rows(max_chunk_rows)),
          decode_rows_(MiniResidentBackend::checked_chunk_rows(decode_rows)),
          max_rows_(std::max({prefill_rows_, std::min(kMaxVerifyRows, verify_rows), decode_rows_})),
          layout_(max_positions, max_rows_), weight_plan_(read_plan(checkpoint, storage)),
          weight_storage_(weight_plan_.bytes()),
          activations_(mini_resident_activation_elements(max_rows_)),
          history_backup_(session::cache_buffer_elements<float>(layout_.convolution_elements())),
          kv_backup_(mini_resident_max_kv_checkpoint_elements()), embeddings_(std::size_t{max_rows_} * kHiddenWidth),
          hidden_(embeddings_.span().size()), logits_(std::size_t{decode_rows_} * kOutputVocabulary),
          draft_taps_(capture_draft_taps ? MiniDSparkTapPlan::elements(max_positions) : 0U),
#if !defined(__HIPCC__)
          token_ids_(max_rows_),
#endif
          weights_(upload_weights()), backend_(weights_, max_rows_, stream_.get(), packed_decode, step_graphs),
          greedy_(decode_rows_, kOutputVocabulary) {
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
        }, prefill_rows_);
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
        groups_.clear();
        for (std::size_t row = 0; row < slots.size(); ++row) {
            auto& state = slots_.at(slots[row])->state();
            validate_inputs(state, ids.subspan(row, 1U), {});
            groups_.push_back({&state, {state.committed(), 1U}});
        }
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
        const auto logits = logits_.span().first(ids.size() * kOutputVocabulary);
        // Verify routing gives each packed row the decode MoE and attention arithmetic.
        const auto kind = rows <= kMaxVerifyRows ? MiniAppendKind::verify : MiniAppendKind::decode;
        MiniResidentForward forward(backend_, workspace());
        forward.append_batch(groups_, rows, embeddings, hidden, [&] {
            embed(ids, embeddings, rows, kind);
        }, [&](MiniConstRows output) {
            for (unsigned row = 0; row < rows; ++row)
                project_head(output.data() + std::size_t{row} * kHiddenWidth,
                             logits.subspan(std::size_t{row} * kOutputVocabulary, kOutputVocabulary), true);
            greedy_.enqueue(logits, stream_.get());
        }, [&] {
            for (std::size_t row = 0; row < top.size(); ++row) (void)checked_token(greedy_, row);
        });
        for (std::size_t row = 0; row < top.size(); ++row) top[row] = greedy_.token(row);
        last_logit_rows_ = rows;
    }

    // Final-row scores of the last decode_slots() call, exactly as the greedy pick saw them (after the
    // output softcap). Lane `row` follows the slot order of that call. Valid until the next decode_slots().
    [[nodiscard]] bool final_logits_supported() const { return draft_taps_.span().empty(); }
    [[nodiscard]] std::uint32_t logit_width() const { return kOutputVocabulary; }
    void read_slot_logits(std::uint32_t row, std::span<float> output) {
        if (row >= last_logit_rows_ || output.size() != kOutputVocabulary)
            throw std::logic_error("Inkling 2 Mini: no final logits for that decode lane");
        reserve_staged_logits(kOutputVocabulary);
        mini_gpu::copy(staged_logits_.get(), logits_.data() + std::size_t{row} * kOutputVocabulary,
                       std::size_t{kOutputVocabulary} * sizeof(float), mini_gpu::device_to_host, stream_.get());
        mini_gpu::synchronize(stream_.get());
        std::copy_n(staged_logits_.get(), kOutputVocabulary, output.data());
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
    // Resolved FAST dense prefill lever (mini_dense_prefill_mma1); set once after construction.
    void set_prefill_dense_mma1(bool enabled) { backend_.set_prefill_dense_mma1(enabled); }
    [[nodiscard]] std::uint64_t gate_null_applications() const { return backend_.gate_null_applications(); }

    void prefill_gate_rows(std::span<const unsigned> prompt, std::span<float> logits) {
        backend_.abandon_capture();
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
                                  std::span<unsigned> verify_top = {}) {
        static_assert(std::is_same_v<unsigned, std::uint32_t>);
        const auto rows = static_cast<unsigned>(ids.size());
        const auto elements = ids.size() * kHiddenWidth;
        const auto embeddings = embeddings_.span().first(elements);
        reserve_staged_logits(host_logits.size());
        const bool multi_pick = verify_top.size() > 1U;
        if (multi_pick && !verify_greedy_) {
            if (!verify_logits_) verify_logits_.emplace(std::size_t{max_verify_rows()} * kOutputVocabulary);
            verify_greedy_.emplace(max_verify_rows(), kOutputVocabulary);
        }
        auto& greedy = multi_pick ? *verify_greedy_ : greedy_;
        auto token = head_argmax::kNone;
        // The one-row decode step of the single session replays from a device step descriptor. DSpark
        // taps and the other slots bind position- or slot-dependent addresses, so they stay eager.
        const bool decode_step = kind == MiniAppendKind::decode && rows == 1U &&
                                 &state == &slots_.front()->state() && draft_taps_.span().empty();
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
                        const auto row_scores = scores.subspan((multi_pick ? row : 0U) * kOutputVocabulary,
                                                               kOutputVocabulary);
                        project_head(input, row_scores, true);
                        if (pick_next && !multi_pick) greedy.enqueue(row_scores, stream_.get());
                        if (!decode_step && !host_logits.empty())
                            mini_gpu::copy(staged_logits_.get() + row * kOutputVocabulary, row_scores.data(),
                                           row_scores.size_bytes(), mini_gpu::device_to_host, stream_.get());
                    }
                    if (multi_pick) greedy.enqueue(scores.first(head_rows * kOutputVocabulary), stream_.get());
                },
                [&] {
                    if (!pick_next) return;
                    if (verify_top.empty()) token = checked_token(greedy, 0U);
                    else for (std::size_t row = 0; row < verify_top.size(); ++row)
                        verify_top[row] = checked_token(greedy, row);
                }, [&] { embed(ids, embeddings, rows, kind); }, [&] {
                    // The optional host reply buffer can grow between calls.
                    if (decode_step && !host_logits.empty())
                        mini_gpu::copy(staged_logits_.get(), logits_.data(), logits_.span().size_bytes(),
                                       mini_gpu::device_to_host, stream_.get());
                }, !verify_top.empty());
        } catch (...) {
            backend_.abandon_capture();
            state.drain_or_poison([&] { mini_gpu::synchronize(stream_.get()); });
            throw;
        }
        if (!host_logits.empty()) std::copy_n(staged_logits_.get(), host_logits.size(), host_logits.data());
        return token;
    }

    unsigned prefill_rows_, decode_rows_, max_rows_;
    MiniSessionStatePlan layout_;
    MiniResidentWeightPlan weight_plan_;
    mini_gpu::StreamOwner stream_;
    mini_gpu::Buffer<std::uint8_t> weight_storage_;
    mini_gpu::Buffer<float> activations_, history_backup_;
    mini_gpu::Buffer<std::uint16_t> kv_backup_;
    mini_gpu::Buffer<float> embeddings_, hidden_, logits_;
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
    std::array<unsigned, kMaxVerifyRows> verify_ids_{}, verify_top_{};
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits_;
    std::size_t staged_logits_size_{};
    unsigned last_logit_rows_{};
};

}  // namespace strix::inkling2

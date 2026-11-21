#pragma once

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
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

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

// Both device paths use their shared finite first-max reduction and return
// only a token to the host unless the caller also asked for the score row.
class MiniResidentGreedy {
public:
#if defined(__HIPCC__)
    MiniResidentGreedy() {
        void* address{};
        mini_gpu::check(hipHostMalloc(&address, sizeof(HipGreedyRowsReply),
                                     hipHostMallocMapped | hipHostMallocCoherent), "allocate Mini greedy reply");
        reply_ = static_cast<HipGreedyRowsReply*>(address);
    }
    ~MiniResidentGreedy() { (void)hipHostFree(reply_); }

    void enqueue(std::span<const float> logits, mini_gpu::Stream stream) {
        const ExecutionBufferAddress row{{}, reinterpret_cast<std::uintptr_t>(logits.data()), logits.size_bytes(), 0U};
        const ExecutionBufferAddress scratch{{}, reinterpret_cast<std::uintptr_t>(partials_.data()),
                                             partials_.span().size_bytes(), 0U};
        enqueue_mini_hip_greedy({&row, 1U}, scratch, reply_, stream);
    }
    [[nodiscard]] unsigned token() const { return reply_->status != 0U ? head_argmax::kNone : reply_->token; }

private:
    mini_gpu::Buffer<std::uint8_t> partials_{static_cast<std::size_t>(
        hip_greedy_rows_scratch_bytes(1U, kOutputVocabulary))};
    HipGreedyRowsReply* reply_{};
#else
    MiniResidentGreedy() {
        void* address{};
        mini_gpu::check(cudaMallocHost(&address, sizeof(std::uint32_t)), "allocate Mini greedy reply");
        reply_ = static_cast<std::uint32_t*>(address);
    }
    ~MiniResidentGreedy() { (void)cudaFreeHost(reply_); }

    void enqueue(std::span<const float> logits, mini_gpu::Stream stream) {
        namespace argmax = strix::cuda::core::head_argmax;
        mini_gpu::check(argmax::pick_rows(
            {logits.data(), 0U, kOutputVocabulary, 1U, 0U, kChunks, partials_.data()},
            argmax::EmitIndexOrNone{device_token_.data()}, stream), "enqueue Mini greedy pick");
        mini_gpu::copy(reply_, device_token_.data(), sizeof(*reply_), mini_gpu::device_to_host, stream);
    }
    [[nodiscard]] unsigned token() const { return *reply_; }

private:
    static constexpr auto kChunks = head_argmax::default_chunks(kOutputVocabulary);
    mini_gpu::Buffer<head_argmax::Pick> partials_{kChunks};
    mini_gpu::Buffer<std::uint32_t> device_token_{1U};
    std::uint32_t* reply_{};
#endif
};

}  // namespace detail

// One unpadded token stream, bound to the selected device and stream for its
// lifetime. All checkpoint tensors and live cache slots stay on the device.
// The owner is serial. Calls return after the stream and state commit finish.
class MiniResidentDecoder {
public:
    MiniResidentDecoder(const std::string& checkpoint, unsigned max_positions, unsigned max_chunk_rows,
                        bool packed_decode = false, bool step_graphs = MiniResidentBackend::default_step_graphs)
        : max_rows_(MiniResidentBackend::checked_chunk_rows(max_chunk_rows)),
          layout_(max_positions, max_rows_), weight_plan_(read_plan(checkpoint)),
          weight_storage_(weight_plan_.bytes()),
          keys_(session::cache_buffer_elements<std::uint16_t>(layout_.kv_elements_per_plane())),
          values_(keys_.span().size()),
          histories_(session::cache_buffer_elements<float>(layout_.convolution_elements())),
          activations_(mini_resident_activation_elements(max_rows_)), history_backup_(histories_.span().size()),
          kv_backup_(mini_resident_max_kv_checkpoint_elements()), embeddings_(std::size_t{max_rows_} * kHiddenWidth),
          hidden_(embeddings_.span().size()), logits_(kOutputVocabulary),
#if !defined(__HIPCC__)
          token_ids_(max_rows_),
#endif
          weights_(upload_weights()), backend_(weights_, max_rows_, stream_.get(), packed_decode, step_graphs) {
        reset_gate_session();
    }

    ~MiniResidentDecoder() { backend_.abandon_capture(); }

    // Return the token after the last prompt input. Only that row goes through
    // the output projection. If a later chunk fails after an earlier one
    // committed, retire the session so a retry cannot append the prefix twice.
    [[nodiscard]] unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits = {}) {
        backend_.abandon_capture();
        validate_inputs(prompt, logits);
        return state_->append_sequence(prompt, [&](std::span<const unsigned> chunk, bool last) {
            return append(chunk, last, last ? logits : std::span<float>{});
        });
    }

    [[nodiscard]] unsigned decode(unsigned token, std::span<float> logits = {}) try {
        const std::span<const unsigned> ids{&token, 1U};
        validate_inputs(ids, logits);
        return append(ids, true, logits, true);
    } catch (...) {
        backend_.abandon_capture();
        throw;
    }

    void reset_gate_session() {
        backend_.reset_step_graph();
        mini_gpu::clear(histories_.data(), histories_.span().size_bytes(), stream_.get());
        mini_gpu::synchronize(stream_.get());
        // KV visibility is determined by the fresh frontier. No old row is read.
        forward_.reset();
        state_.emplace(layout_, session::KvRowPlanes{keys_.span(), values_.span()}, histories_.span());
        forward_.emplace(backend_, *state_, MiniResidentForwardWorkspace{
            activations_.span(), history_backup_.span(), kv_backup_.span()});
    }

    void set_gate_route_tap(moe::RouteTap* tap) { backend_.set_gate_route_tap(tap); }

    void prefill_gate_rows(std::span<const unsigned> prompt, std::span<float> logits) {
        backend_.abandon_capture();
        validate_inputs(prompt, {});
        const auto rows = logits.size() / kOutputVocabulary;
        if (rows == 0U || rows > prompt.size() || logits.size() % kOutputVocabulary != 0U)
            throw std::invalid_argument("invalid Mini prefill gate head shape");
        const auto start = state_->committed();
        const auto first_tail_row = prompt.size() - rows;
        (void)state_->append_sequence(prompt, [&](std::span<const unsigned> chunk, bool /*last*/) {
            const auto first = std::size_t{state_->committed() - start};
            const auto end = first + chunk.size();
            if (end <= first_tail_row) return append(chunk, false, {});
            const auto offset = std::max(first, first_tail_row) - first_tail_row;
            const auto count = end - std::max(first, first_tail_row);
            return append(chunk, false, logits.subspan(offset * kOutputVocabulary, count * kOutputVocabulary));
        });
    }

private:
    [[nodiscard]] static MiniResidentWeightPlan read_plan(const std::string& path) {
        MiniTextWeights source;
        const auto status = source.open(path);
        if (!status) throw std::runtime_error(status.detail);
        MiniResidentBackend::require_device();
        return MiniResidentWeightPlan::make(source);
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

    void validate_inputs(std::span<const unsigned> tokens, std::span<float> logits) const {
        if (state_->poisoned()) throw std::runtime_error("Inkling 2 Mini: device session is invalid");
        if (!layout_.admits_sequence(state_->committed(), tokens.size()))
            throw std::invalid_argument("Inkling 2 Mini: input tokens exceed the session context");
        if (!logits.empty() && logits.size() != kOutputVocabulary)
            throw std::invalid_argument("Inkling 2 Mini: logits output must hold one vocabulary row");
        if (std::ranges::any_of(tokens, [](unsigned id) { return id >= kVocabulary; }))
            throw std::out_of_range("token id exceeds the Mini vocabulary");
    }

    [[nodiscard]] unsigned append(std::span<const unsigned> ids, bool pick_next, std::span<float> host_logits,
                                  bool decode_step = false) {
        static_assert(std::is_same_v<unsigned, std::uint32_t>);
        const auto rows = static_cast<unsigned>(ids.size());
        const auto elements = ids.size() * kHiddenWidth;
        const auto embeddings = embeddings_.span().first(elements);
        const auto table = weights_.bf16("embed.weight");
        const auto gain = weights_.bf16("embed_norm.weight");
        auto token = head_argmax::kNone;
        try {
            if (host_logits.size() > staged_logits_size_) {
                void* address{};
#if defined(__HIPCC__)
                mini_gpu::check(hipHostMalloc(&address, host_logits.size_bytes()), "allocate Mini logits reply");
#else
                mini_gpu::check(cudaMallocHost(&address, host_logits.size_bytes()), "allocate Mini logits reply");
#endif
                staged_logits_.reset(static_cast<float*>(address));
                staged_logits_size_ = host_logits.size();
            }
            backend_.set_step_token(decode_step ? std::optional{ids.front()} : std::nullopt);
            const auto enqueue_embeddings = [&] {
                const auto* step = backend_.capture_descriptor();
#if defined(__HIPCC__)
                if (step) {
                    core::enqueue_hip_bf16_embedding_device_row(table, kVocabulary, kHiddenWidth, &step->token,
                                                                embeddings, stream_.get());
                } else {
                    enqueue_mini_hip_embeddings(table, ids, embeddings, stream_.get());
                }
                backend_.rms(embeddings.data(), gain.data(), embeddings.data(), rows);
#else
                if (!step) {
                    // The caller's token span can be pageable.
                    mini_gpu::copy(token_ids_.data(), ids.data(), ids.size_bytes(), mini_gpu::host_to_device,
                                   stream_.get());
                }
                backend_.check(enqueue_mini_cuda_embeddings(table.data(), gain.data(),
                    step ? &step->token : token_ids_.data(), embeddings.data(), rows, stream_.get()));
#endif
            };
            forward_->append({state_->committed(), rows}, embeddings, hidden_.span().first(elements),
                [&](MiniResidentChunk, MiniConstRows hidden) {
                    if (!pick_next && host_logits.empty()) return;
                    const auto head_rows = pick_next ? std::size_t{1} : host_logits.size() / kOutputVocabulary;
                    const auto matrix = weights_.bf16("unembed.weight");
                    const auto tail = hidden.last(head_rows * kHiddenWidth);
                    // Use the single-row head for every score row. A wide dense
                    // dispatch can select different arithmetic on the same device.
                    for (std::size_t row = 0; row < head_rows; ++row) {
                        const auto* input = tail.data() + row * kHiddenWidth;
#if defined(__HIPCC__)
                        backend_.check(enqueue_mini_hip_head(input, 1U, matrix, logits_.span(), stream_.get()));
#else
                        backend_.check(enqueue_mini_cuda_logits(matrix.data(), input, logits_.data(), 1U, stream_.get()));
#endif
                        if (pick_next) greedy_.enqueue(logits_.span(), stream_.get());
                        if (!decode_step && !host_logits.empty())
                            mini_gpu::copy(staged_logits_.get() + row * kOutputVocabulary, logits_.data(),
                                           logits_.span().size_bytes(), mini_gpu::device_to_host, stream_.get());
                    }
                },
                [&] {
                    if (!pick_next) return;
                    token = greedy_.token();
                    if (token >= kOutputVocabulary)
                        throw std::runtime_error("Inkling 2 Mini: non-finite output logits");
                }, enqueue_embeddings, [&] {
                    // The optional host reply buffer can grow between calls.
                    if (decode_step && !host_logits.empty())
                        mini_gpu::copy(staged_logits_.get(), logits_.data(), logits_.span().size_bytes(),
                                       mini_gpu::device_to_host, stream_.get());
                });
        } catch (...) {
            backend_.abandon_capture();
            state_->drain_or_poison([&] { mini_gpu::synchronize(stream_.get()); });
            throw;
        }
        if (!host_logits.empty()) std::copy_n(staged_logits_.get(), host_logits.size(), host_logits.data());
        return token;
    }

    unsigned max_rows_;
    MiniSessionStatePlan layout_;
    MiniResidentWeightPlan weight_plan_;
    mini_gpu::StreamOwner stream_;
    mini_gpu::Buffer<std::uint8_t> weight_storage_;
    mini_gpu::Buffer<std::uint16_t> keys_, values_;
    mini_gpu::Buffer<float> histories_, activations_, history_backup_;
    mini_gpu::Buffer<std::uint16_t> kv_backup_;
    mini_gpu::Buffer<float> embeddings_, hidden_, logits_;
#if !defined(__HIPCC__)
    mini_gpu::Buffer<std::uint32_t> token_ids_;
#endif
    MiniResidentWeights weights_;
    MiniResidentBackend backend_;
    std::optional<MiniResidentSessionState> state_;
    std::optional<MiniResidentForward<MiniResidentBackend>> forward_;
    detail::MiniResidentGreedy greedy_;
    std::unique_ptr<float, detail::MiniPinnedLogitsDeleter> staged_logits_;
    std::size_t staged_logits_size_{};
};

}  // namespace strix::inkling2

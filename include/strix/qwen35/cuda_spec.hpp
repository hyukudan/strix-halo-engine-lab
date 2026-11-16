#pragma once
// Binds the qwen35 CUDA engine to the shared block drafting layer (strix/spec/block_drafter.hpp).
// Home: specific (engine API binding: CudaEngine rows / commit / MTP); the drafting logic is shared (strix::spec).
// Header-only (CUDA runtime calls); weight-format agnostic: everything goes through CudaEngine, so BF16 / Q8_0 / TURBO
// packs need no change here.
//
// SpecEngine owns the glue state of one stream:
//   verify(tokens)  -> forward_rows(R, all logits) + host argmax of every row           (strix::spec::VerifyTarget)
//   commit(k)       -> engine commit(k) (+ host sync so the step wall is complete)
//   MtpChain        -> native MTP drafter (strix::spec::ChainDrafter) with LAZY catch-up: the hidden rows of every
//                      committed position are kept in a device ring and written into the MTP KV only when a draft
//                      needs them (copy-only stretches pay no MTP cost). Chain steps use the MTP layer's own hidden.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strix/qwen35/cuda_engine.hpp"
#include "strix/spec/block_drafter.hpp"

namespace strix::qwen35 {

class SpecEngine final : public strix::spec::VerifyTarget {
public:
    explicit SpecEngine(CudaEngine& e, bool chain_pre_hidden = false) : e_(e), chain_pre_(chain_pre_hidden) {
        hid_ = e.config().hidden;
        if (e.has_mtp()) {
            alloc(ring_, kRing * hid_);
            alloc(carry_, hid_);
            ring_next_.reserve(kRing);
        }
    }
    ~SpecEngine() override {
        if (ring_) cudaFree(ring_);
        if (carry_) cudaFree(carry_);
    }
    SpecEngine(const SpecEngine&) = delete;
    SpecEngine& operator=(const SpecEngine&) = delete;

    CudaEngine& engine() { return e_; }
    [[nodiscard]] bool has_chain() const { return ring_ != nullptr; }

    // ---- VerifyTarget
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        last_rows_ = static_cast<std::uint32_t>(tokens.size());
        last_ = e_.forward_rows(tokens.data(), last_rows_, LogitRows::all);
        greedy.resize(last_rows_);
        e_.read_top(last_, last_rows_, greedy.data());  // synchronizes
    }
    void commit(std::uint32_t k) override {
        e_.commit(k);
        e_.sync();
    }
    // Prefill chunk (not a drafting step): rows at positions committed().. ; next_first = the token after the chunk when
    // it is not the last one. The last chunk returns the greedy token after the prompt (the anchor) in *top_out.
    // Feeds the MTP ring (row i's successor = tokens[i+1], the last row's = next_first / the anchor).
    void prefill(const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t next_first, bool last, std::uint32_t* top_out) {
        const auto o = e_.forward_rows(tokens, rows, last ? LogitRows::last : LogitRows::none);
        if (last) e_.read_top(o, 1, top_out);
        if (ring_) {
            std::uint32_t nx[32];
            for (std::uint32_t i = 0; i + 1 < rows; ++i) nx[i] = tokens[i + 1];
            nx[rows - 1] = last ? *top_out : next_first;
            push_hidden(o, rows, nx);
        }
        e_.commit(rows);
        e_.sync();
    }
    // ---- chain
    class MtpChain final : public strix::spec::ChainDrafter {
    public:
        explicit MtpChain(SpecEngine& s) : s_(s) {}
        void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
            s_.chain_draft(prefix, k, out);
        }
        void accept(std::span<const std::uint32_t> emitted) override { s_.chain_accept(emitted); }

    private:
        SpecEngine& s_;
    };
    MtpChain make_chain() { return MtpChain(*this); }
    std::uint64_t chain_calls = 0, chain_rows = 0;

private:
    static constexpr std::uint32_t kRing = 512;
    static void alloc(float*& p, std::size_t n) {
        if (cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float)) != cudaSuccess) throw std::runtime_error("spec: cudaMalloc");
    }
    // Append the hidden rows of the last forward + their successor tokens; flush when the ring is full.
    void push_hidden(const RowsOut& o, std::uint32_t rows, const std::uint32_t* next_tokens) {
        const float* src = chain_pre_ ? o.hidden_pre : o.hidden_post;
        std::uint32_t done = 0;
        while (done < rows) {
            if (ring_next_.size() == kRing) flush();
            const std::uint32_t n = std::min<std::uint32_t>(rows - done, kRing - static_cast<std::uint32_t>(ring_next_.size()));
            cudaMemcpyAsync(ring_ + static_cast<std::size_t>(ring_next_.size()) * hid_, src + static_cast<std::size_t>(done) * hid_,
                            static_cast<std::size_t>(n) * hid_ * sizeof(float), cudaMemcpyDeviceToDevice, e_.stream());
            ring_next_.insert(ring_next_.end(), next_tokens + done, next_tokens + done + n);
            done += n;
        }
    }
    void chain_accept(std::span<const std::uint32_t> emitted) {
        if (!ring_ || last_rows_ == 0) return;
        // rows 0..emitted-1 of the last verify are committed; emitted[i] follows row i
        push_hidden(last_, static_cast<std::uint32_t>(emitted.size()), emitted.data());
    }
    // Write the ring into the MTP KV (rows of 32, causal inside each call). Leaves the last row's top / hidden in
    // d1_ / carry_ (the draft of the token after the last committed row's successor).
    void flush() {
        std::uint32_t off = 0;
        const auto total = static_cast<std::uint32_t>(ring_next_.size());
        while (off < total) {
            const std::uint32_t n = std::min<std::uint32_t>(32, total - off);
            const bool final_chunk = off + n == total;  // only the last row's draft is needed: earlier chunks skip lm_head
            const auto m = e_.mtp_forward(ring_next_.data() + off, ring_ + static_cast<std::size_t>(off) * hid_, n, mtp_pos_, final_chunk);
            ++chain_calls;
            chain_rows += n;
            mtp_pos_ += n;
            off += n;
            if (final_chunk) {
                read_last(m, n);
            }
        }
        ring_next_.clear();
    }
    void read_last(const CudaEngine::MtpOut& m, std::uint32_t n) {
        RowsOut ro;
        ro.top = m.top;
        e_.read_top(ro, n, top_tmp_);
        d1_ = top_tmp_[n - 1];
        const float* h = chain_pre_ ? m.hidden_pre : m.hidden_post;
        cudaMemcpyAsync(carry_, h + static_cast<std::size_t>(n - 1) * hid_, hid_ * sizeof(float), cudaMemcpyDeviceToDevice, e_.stream());
    }
    void chain_draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) {
        out.clear();
        if (!ring_) return;
        if (!ring_next_.empty()) flush();
        // carry_ = MTP hidden after the last committed row; d1_ = its prediction (the token after the anchor's successor)
        std::uint32_t pos = mtp_pos_;  // speculative rows from here on are overwritten by the next flush
        std::uint32_t tok = d1_;       // chain token for the next step when there is no prefix
        for (std::size_t i = 0; i < prefix.size(); ++i)  // feed the copy tokens: only the last step's prediction is used
            tok = step(prefix[i], pos++, i + 1 == prefix.size());
        for (std::uint32_t j = 0; j < k; ++j) {
            out.push_back(tok);
            if (j + 1 < k) tok = step(tok, pos++);
        }
    }
    std::uint32_t step(std::uint32_t input_token, std::uint32_t pos, bool want_top = true) {
        const auto m = e_.mtp_forward(&input_token, carry_, 1, pos, want_top);
        ++chain_calls;
        ++chain_rows;
        std::uint32_t t = 0;
        if (want_top) {
            RowsOut ro;
            ro.top = m.top;
            e_.read_top(ro, 1, &t);
        }
        const float* h = chain_pre_ ? m.hidden_pre : m.hidden_post;
        cudaMemcpyAsync(carry_, h, hid_ * sizeof(float), cudaMemcpyDeviceToDevice, e_.stream());
        return t;
    }

    CudaEngine& e_;
    bool chain_pre_;
    std::uint32_t hid_ = 0;
    RowsOut last_{};
    std::uint32_t last_rows_ = 0;
    float* ring_ = nullptr;
    float* carry_ = nullptr;
    std::vector<std::uint32_t> ring_next_;
    std::uint32_t mtp_pos_ = 0, d1_ = 0;
    std::uint32_t top_tmp_[32]{};

public:
    // New sequence: the engine is reset by the caller; forget the MTP positions.
    void reset() {
        ring_next_.clear();
        mtp_pos_ = 0;
        d1_ = 0;
        last_rows_ = 0;
    }
};

}  // namespace strix::qwen35

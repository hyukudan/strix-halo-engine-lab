#pragma once
// Binds the qwen35 CUDA engine to the shared block drafting layer (strix/spec/block_drafter.hpp).
// Home: specific (engine API binding: CudaEngine rows / commit / MTP); the drafting logic and the MTP chain glue are shared
// (strix::spec::MtpChain in strix/spec/mtp_chain.hpp; this file supplies only the device calls).
// Header-only (CUDA runtime calls); weight-format agnostic: everything goes through CudaEngine, so BF16 / Q8_0 / TURBO
// packs need no change here.
//
// SpecEngine owns the glue state of one stream:
//   verify(tokens)  -> forward_rows(R, all logits) + host argmax of every row           (strix::spec::VerifyTarget)
//   verify_launch_fed(host rows, feed) / verify_collect(tokens)  -> verify pre-launch: forward_rows_fed behind the draft
//                      (drafted ids gathered from the chain's feed slots on the device), then the same read as verify()
//   commit(k)       -> engine commit(k) (+ host sync so the step wall is complete)
//   make_chain()    -> native MTP drafter (strix::spec::ChainDrafter = shared MtpChain)
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "strix/qwen35/cuda_engine.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/mtp_chain.hpp"

namespace strix::qwen35 {

struct CudaChainDev {
    static float* alloc(CudaEngine&, std::size_t n) {
        float* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float)) != cudaSuccess) throw std::runtime_error("spec: cudaMalloc");
        return p;
    }
    static void release(CudaEngine&, float* p) { cudaFree(p); }
    static void copy(CudaEngine& e, float* dst, const float* src, std::size_t n) {
        cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, e.stream());
    }
};

class SpecEngine final : public strix::spec::VerifyTarget {
public:
    using Chain = strix::spec::MtpChain<CudaEngine, CudaChainDev>;
    explicit SpecEngine(CudaEngine& e, bool chain_pre_hidden = false) : e_(e), pre_(chain_pre_hidden) {
        if (e.has_mtp()) chain_ = std::make_unique<Chain>(e, chain_pre_hidden, 32U);
    }
    ~SpecEngine() override {
        if (rows_dev_) cudaFree(rows_dev_);
    }
    SpecEngine(const SpecEngine&) = delete;
    SpecEngine& operator=(const SpecEngine&) = delete;

    CudaEngine& engine() { return e_; }
    [[nodiscard]] bool has_chain() const { return chain_ != nullptr; }

    // ---- VerifyTarget
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto o = e_.forward_rows(tokens.data(), rows, LogitRows::all);
        greedy.resize(rows);
        e_.read_top(o, rows, greedy.data());  // synchronizes
        if (chain_) chain_->observe_verify(o, rows);
    }
    // Verify pre-launch (strix/spec/block_drafter.hpp): the verify graph is enqueued behind the draft with the drafts'
    // ids gathered from the chain's feed slots on the device; verify_collect() reads the result like verify().
    bool verify_launch_fed(std::span<const std::uint32_t> host_rows, const strix::spec::DraftFeed& feed) override {
        if (!chain_) return false;
        pending_ = e_.forward_rows_fed(host_rows.data(), static_cast<std::uint32_t>(host_rows.size()), feed.slot.data(),
                                       feed.host_id.data(), feed.k, LogitRows::all);
        pending_rows_ = static_cast<std::uint32_t>(host_rows.size()) + feed.k;
        return true;
    }
    void verify_collect(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        if (pending_rows_ == 0 || rows != pending_rows_) throw std::runtime_error("spec: verify_collect does not match the pre-launched verify");
        pending_rows_ = 0;
        greedy.resize(rows);
        e_.read_top(pending_, rows, greedy.data());  // synchronizes
        if (chain_) chain_->observe_verify(pending_, rows);
    }
    void commit(std::uint32_t k) override {
        e_.commit(k);
        e_.sync();
    }
    // Token tree (strix/spec/tree_window.hpp): sibling rows verified in the same window, EXACT (each row = the linear window of
    // its path). Opt-in until measured on the device (set_tree); off = tree_rows() 0, every block linear (bitwise as before)
    // and the draft head computes no runner-ups (plain argmax launches, the pre-tree draft path).
    void set_tree(bool on) {
        tree_ = on;
        e_.set_runner_ups(on);
    }
    // decode_attn8 (FAST e4m3 shadow) has no leaf K/V index (CudaEngine::forward_tree refuses it): linear windows only.
    [[nodiscard]] std::uint32_t tree_rows() const override {
        return tree_ && !e_.options().decode_attn8 ? strix::cuda_core::kMaxRows : 0U;
    }
    void verify_tree(const strix::spec::TreeRows& rows, std::vector<std::uint32_t>& greedy) override {
        tw_ = strix::spec::tree_window(rows);
        const auto o = e_.forward_tree(rows.tokens.data(), tw_, LogitRows::all);
        greedy.resize(tw_.rows);
        e_.read_top(o, tw_.rows, greedy.data());  // synchronizes
        if (chain_) chain_->observe_verify(o, tw_.rows);
    }
    void commit_path(std::span<const std::uint32_t> path) override {
        e_.commit_path(strix::spec::path_commit(tw_, path));
        e_.sync();
    }
    // Prefill chunk (not a drafting step): rows at positions committed().. ; next_first = the token after the chunk when
    // it is not the last one. The last chunk returns the greedy token after the prompt (the anchor) in *top_out.
    // Feeds the MTP ring (row i's successor = tokens[i+1], the last row's = next_first / the anchor).
    void prefill(const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t next_first, bool last, std::uint32_t* top_out) {
        const auto o = e_.forward_rows(tokens, rows, last ? LogitRows::last : LogitRows::none);
        if (last) e_.read_top(o, 1, top_out);
        if (chain_) chain_->observe_prefill(o, rows, tokens, last ? *top_out : next_first);
        e_.commit(rows);
        e_.sync();
    }
    // Whole prompt (positions committed().. ) through the engine's large-M prefill (exact or FAST by its options) with every
    // row's hidden exported (CudaEngine::prefill rows_out) and fed to the MTP chain: the chain sees the same rows, successors
    // and positions as from prefill() windows of any size (it rings and chunks them itself); under the exact route the
    // committed trunk state is that of the windows (cuda_engine.hpp prefill contract). *top_out = the anchor. Long prompts
    // only cost hidden x 4 B per row of device memory (kept for reuse, freed with this object).
    // Returns the trunk wall (engine prefill + anchor read, ms); the rest of the call is the MTP feed (stage split).
    double prefill_big(const std::uint32_t* tokens, std::uint32_t n, std::uint32_t* top_out) {
        if (n == 0) throw std::invalid_argument("prefill_big: empty prompt");
        const auto t0 = std::chrono::steady_clock::now();
        float* rows = nullptr;
        if (chain_) {
            const std::size_t need = std::size_t(n) * e_.config().hidden;
            if (need > rows_cap_) {
                if (rows_dev_) cudaFree(rows_dev_);
                rows_dev_ = nullptr;
                rows_cap_ = 0;
                if (cudaMalloc(reinterpret_cast<void**>(&rows_dev_), need * sizeof(float)) != cudaSuccess)
                    throw std::runtime_error("prefill_big: cudaMalloc of the hidden rows");
                rows_cap_ = need;
            }
            rows = rows_dev_;
        }
        const auto o = e_.prefill(tokens, n, LogitRows::last, rows, pre_);
        e_.read_top(o, 1, top_out);  // synchronizes
        const double trunk_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (chain_) {
            RowsOut h{};
            h.hidden_post = rows;
            h.hidden_pre = rows;
            chain_->observe_prefill(h, n, tokens, *top_out);
        }
        e_.sync();
        return trunk_ms;
    }
    Chain& make_chain() { return *chain_; }
    [[nodiscard]] std::uint64_t chain_calls() const { return chain_ ? chain_->calls() : 0; }
    [[nodiscard]] std::uint64_t chain_rows() const { return chain_ ? chain_->rows() : 0; }
    // K/V-only catch-up chunks (MtpChain::kv_calls, not in chain_calls): full catch-up calls == chain_calls + chain_kv_calls
    [[nodiscard]] std::uint64_t chain_kv_calls() const { return chain_ ? chain_->kv_calls() : 0; }
    // New sequence: the engine is reset by the caller; forget the MTP positions.
    void reset() {
        if (chain_) chain_->reset();
    }

private:
    CudaEngine& e_;
    bool pre_ = false;  // chain hidden convention (pre = residual, post = final-norm output)
    std::unique_ptr<Chain> chain_;
    RowsOut pending_{};              // pre-launched verify (verify_launch_fed) until verify_collect
    std::uint32_t pending_rows_ = 0;
    bool tree_ = false;
    strix::spec::TreeWindow tw_{};
    float* rows_dev_ = nullptr;  // prefill_big hidden rows [rows_cap_ floats]
    std::size_t rows_cap_ = 0;
};

}  // namespace strix::qwen35

#pragma once

#include "strix/deepseek41_plan.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Host reference of a deepseek41 decode step over a trimmed model (target layers [0, N) plus the embedding and the
// output head), executing a decode plan one token at a time. Every operation is deterministic for a given host
// binary: a run reproduces its logits bit for bit, and a device run of the same plan is compared op by op.
//
// Pieces: the layer sites (embedding, Engram, hyper-connections, norms, RoPE, Q8_0/F16 projections with FP32
// activations, head and split argmax) are the layer piece's declared references; the router, shared expert, routed
// experts and the two-half combine (two ranks emulated on one node) are the formats/MoE piece's (A8 activations).
// Attention (window ring, compressor, indexer, D512 attention) is the attention piece's reference
// (ds41_attention.hpp) through AttentionHost; `stand_in_ops()` lists operations still on an integration stand-in.
namespace strix::ds41 {
class EngramHasher;
class WeightOverlay;
}

namespace strix::deepseek41 {

class MappedFile;
class AttentionHost;

struct HostForwardOptions {
    std::uint32_t layers = 4;          // target layers [0, layers)
    std::uint32_t max_positions = 4096;
    std::filesystem::path weights_overlay;  // optional requantization candidate (ds41::WeightOverlay)
    // Called with every op output buffer ("op@layer/site:buffer", values) of every step (activation calibration).
    std::function<void(const std::string&, const std::vector<float>&)> buffer_tap;
};

class HostForward {
public:
    HostForward(const std::filesystem::path& gguf, const HostForwardOptions& options);
    ~HostForward();
    HostForward(const HostForward&) = delete;
    HostForward& operator=(const HostForward&) = delete;

    // Feed one token at the next position; returns the FP32 logits of the next token.
    const std::vector<float>& step(std::uint32_t token);
    [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
    // Greedy next token of the last step (split-vocabulary argmax, lowest index on ties).
    [[nodiscard]] std::uint32_t next_token() const noexcept { return next_token_; }
    [[nodiscard]] const Plan& plan() const noexcept { return plan_; }
    [[nodiscard]] const TargetModel& model() const noexcept { return model_; }
    // Per-op FNV-1a hashes of the outputs of the last step, in plan order (for host/device and run/run diffs).
    [[nodiscard]] const std::vector<std::pair<std::string, std::uint64_t>>& last_step_hashes() const noexcept { return hashes_; }
    [[nodiscard]] static std::vector<Op> stand_in_ops();
    // The residual stream (hc_mult x hidden) after each layer of the last step, in layer order.
    [[nodiscard]] const std::vector<std::vector<float>>& last_step_streams() const noexcept { return streams_; }
    // Diagnostic: keep a copy of the first output buffer of one step ("op@layer" or "op@layer/site") every token.
    void set_dump(std::string label) { dump_label_ = std::move(label); }
    // The weight overlay's description line (empty without one).
    [[nodiscard]] std::string overlay_description() const;
    [[nodiscard]] const std::vector<float>& dumped() const noexcept { return dumped_; }

private:
    std::span<const std::uint8_t> tensor(const std::string& name) const;
    void run_step(const Step& s, std::uint32_t token);

    std::unique_ptr<MappedFile> file_;
    std::unique_ptr<ds41::WeightOverlay> overlay_;
    TargetModel model_;
    Plan plan_;
    HostForwardOptions options_;
    std::map<std::string, const ManifestTensor*> tensors_;
    std::unique_ptr<AttentionHost> attention_;
    std::unique_ptr<ds41::EngramHasher> hasher_;
    std::vector<std::uint64_t> engram_rows_;  // rows of the current token, [engram layer][column]
    std::uint32_t position_ = 0;
    // per-step activation buffers (FP32), indexed by Buffer
    std::map<Buffer, std::vector<float>> buf_;
    std::vector<std::uint32_t> route_ids_;
    std::vector<float> logits_;
    std::uint32_t next_token_ = 0;
    std::vector<std::pair<std::string, std::uint64_t>> hashes_;
    std::vector<std::vector<float>> streams_;
    std::string dump_label_;
    std::vector<float> dumped_;
};

}  // namespace strix::deepseek41

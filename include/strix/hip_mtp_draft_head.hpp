#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix {
// Draft-only MTP output head. Drafts are guesses verified by the target, so the
// head may cover a vocabulary subset: a static base set plus distinct context
// tokens (prompt and output, capped). Rows are gathered from the Q8_0 head into
// a compact copy at setup/update; each active row computes exactly the logit of
// the full head (same row kernel), and argmax maps back to real token ids with
// the full selector's finite first-max order (ties pick the lowest token id).
// Full mode keeps the original head and identity ids (acceptance A/B baseline).
struct HipMtpDraftHeadConfig {
    std::uint32_t hidden_size=0,vocab_size=0,max_depth=0;
    bool full=true;
    std::vector<std::uint32_t> base_tokens;
    std::uint32_t context_capacity=0;
    // Subset rows requantized Q8_0 -> Q4_0 at gather (about half the head bytes);
    // changes draft logits only, so acceptance must be checked A/B.
    bool q4=false;
};
struct HipMtpDraftHeadStats {
    std::uint32_t rows=0,base_rows=0,context_rows=0,context_dropped=0;
};
// Serial owner bound to one stream. Results land in host-mapped coherent memory:
// read() is valid only after the caller has completed the stream.
class HipMtpDraftHead final {
public:
    HipMtpDraftHead(const void* q8_head,HipMtpDraftHeadConfig,void* stream);
    ~HipMtpDraftHead();
    HipMtpDraftHead(const HipMtpDraftHead&)=delete;
    HipMtpDraftHead& operator=(const HipMtpDraftHead&)=delete;
    // Admits unseen context ids (up to the cap) and enqueues their row gather.
    void observe(std::span<const std::uint32_t> tokens);
    // Clears mapped status/tokens; the previous chain must have completed.
    void begin_chain();
    // Head over the active rows for hidden BF16[hidden], then argmax into the
    // device token slot(step) and mapped token[step].
    void launch(const std::uint16_t* hidden,std::uint32_t step,void* stream);
    [[nodiscard]] const std::uint32_t* device_token(std::uint32_t step) const;
    // Throws on a non-finite selected row; the caller already synchronized.
    [[nodiscard]] std::vector<std::uint32_t> read(std::uint32_t count) const;
    [[nodiscard]] bool full() const noexcept;
    [[nodiscard]] HipMtpDraftHeadStats stats() const noexcept;
    // Device views for tests: compact logits [rows] and real ids [rows] (null in full mode).
    [[nodiscard]] const float* logits() const noexcept;
    [[nodiscard]] const std::uint32_t* ids() const noexcept;
    // Little-endian uint32 token ids; ids >= vocabulary are rejected.
    [[nodiscard]] static std::vector<std::uint32_t> read_token_file(const std::string& path,std::uint32_t vocabulary);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix

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
    // Also reduce the runner-up logit and publish its softmax probability (window controller
    // features and diagnostics); the selected token and its probability are unchanged.
    bool runner_up=false;
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
    // Multi-session draft batch. shared_rows: the number of leading compact rows holding the same ids in every
    // head (same Q8 head, Q4 subset heads with the same hidden size and runner-up setting, no runner-up
    // diagnostic), UINT32_MAX when the heads cannot be batched. launch_rows: launch() of every head at `step`
    // as one batch on `stream` (rows [0, shared) read once for all heads); each head's logits, slot and reply
    // equal its own launch() bit for bit. Each head's previous work must be ordered before `stream`.
    [[nodiscard]] static std::uint32_t shared_rows(std::span<HipMtpDraftHead* const> heads);
    static void launch_rows(std::span<HipMtpDraftHead* const> heads,std::span<const std::uint16_t* const> hidden,
        std::uint32_t step,std::uint32_t shared,void* stream);
    [[nodiscard]] const std::uint32_t* device_token(std::uint32_t step) const;
    // Throws on a non-finite selected row; the caller already synchronized.
    [[nodiscard]] std::vector<std::uint32_t> read(std::uint32_t count) const;
    // Max softmax probability of step's selected row over the active head rows
    // (1/sum exp(l - max), finite rows only; 0 for a failed step). Same mapped
    // reply and completion rule as read(); never affects the selected token.
    [[nodiscard]] float probability(std::uint32_t step) const;
    // Softmax probability of the best row after the selected one (same rows and completion rule);
    // 0 when the head was built without runner_up or the step failed.
    [[nodiscard]] float runner_up_probability(std::uint32_t step) const;
    // Waits until step's selection is published in the mapped reply (its last kernel stores a
    // completion count after the reply fields), polling instead of a stream synchronisation;
    // falls back to synchronising `stream` (and reports its errors) if the count does not
    // arrive. Equivalent to synchronising before reading probability(step). `stream` is the stream carrying
    // the step: the head's own, or the multi-session draft batch stream (launch_rows).
    void wait_step(std::uint32_t step,void* stream) const;
    // Sampled drafts: after wait_step(step), copy the active rows' logits to the host,
    // and replace the step's selection (device slot and mapped reply) before the
    // next step reads it. host_ids() maps rows to token ids (empty in full mode).
    void read_logits(std::vector<float>& out) const;
    [[nodiscard]] std::span<const std::uint32_t> host_ids() const noexcept;
    void set_token(std::uint32_t step,std::uint32_t token,void* stream);
    // Drafts sampled on the device: the step's device slot (written by the sampler's kernels after launch()),
    // the entry capacity, and the host record of a token the device wrote (for read(), after completion).
    [[nodiscard]] std::uint32_t* device_token_slot(std::uint32_t step);
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    void record_token(std::uint32_t step,std::uint32_t token);
    // Diagnostic (STRIX_MTP_DRAFT_TOP2=1 at construction): every step's logits are also copied to pinned host
    // memory, so the runner-up row of a completed step can be read. Off by default; never affects selection.
    [[nodiscard]] bool records_runner_up() const noexcept;
    // Runner-up of step (second in the selector's order) and its softmax probability over the active rows;
    // false when not recorded or fewer than two finite rows. Valid after the caller completed the stream.
    bool runner_up(std::uint32_t step,std::uint32_t& token,float& probability) const;
    [[nodiscard]] bool full() const noexcept;
    [[nodiscard]] HipMtpDraftHeadStats stats() const noexcept;
    // Device views for tests: compact logits [rows] and real ids [rows] (null in full mode).
    [[nodiscard]] const float* logits() const noexcept;
    [[nodiscard]] const std::uint32_t* ids() const noexcept;
    // Little-endian uint32 token ids; ids >= vocabulary are rejected.
    [[nodiscard]] static std::vector<std::uint32_t> read_token_file(const std::string& path,std::uint32_t vocabulary);
    // "auto:DIR": DIR/core32k.bin plus the DIR/pack-<lang>16k.bin holding the most prompt tokens, when that is
    // at least 2 tokens and 5 % of the prompt (otherwise the core alone). Any other spec is returned unchanged.
    // `chosen` receives a short description of the selection. STRIX_MTP_DRAFT_VOCAB=2 uses pack2-<lang>.bin for the
    // chosen pack and core64k.bin when no pack is chosen; STRIX_MTP_DRAFT_CORE=<N>k forces core<N>k.bin.
    [[nodiscard]] static std::string resolve_vocabulary_spec(const std::string& spec,
        std::span<const std::uint32_t> prompt,std::uint32_t vocabulary,std::string& chosen);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix

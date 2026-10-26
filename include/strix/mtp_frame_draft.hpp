#pragma once
// Strict tool-call frame proposals for the MTP draft chain (host only, opt-in).
//
// Inside a tool call (after the open token, before its close token) the output follows the chat template's frame:
// "<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>". The frames
// already in the request's context (earlier tool calls, the template's example) are the grammar: the anchor is the
// longest suffix (up to max_anchor tokens) of the current frame, the occurrences of that anchor inside earlier frames
// give its continuations, and only their common prefix is proposed (strict: every earlier frame agrees, so a
// function or parameter name is proposed only once its typed prefix identifies it). A disagreement right after the
// anchor proposes nothing. Acceptance is learned per request by shadow scoring as in the n-gram extension; the target
// verifies every row, so emitted tokens never depend on these proposals.
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace strix {

struct MtpFrameDraftConfig {
    std::uint32_t open = 0, close = 0;  // the tool-call open/close token ids (both required)
    std::uint32_t max_anchor = 16;     // longest frame suffix used as the anchor
    std::uint32_t max_rows = 7;
    double prior = 0.8, prior_weight = 2.0;  // frames confirmed by 2+ earlier frames: a grammar, optimistic prior
    // Prior of a continuation seen in a single earlier frame (e.g. that call's arguments). Measured (ABA x3): 0.8 gave
    // tp-smith-0 +3.4 % (its accepted rows were single-frame ones) and tp-smith-2 -0.09 % (9 rows rejected); 0.4
    // removed both. Default 0.8; the batch runner reads STRIX_TOOL_DRAFT_SINGLE_PRIOR.
    double single_prior = 0.8;
    double floor = 0.5;
};

struct MtpFrameDraftStats {
    std::uint64_t frame_cycles = 0;     // cycles that started inside a tool-call frame
    std::uint64_t cycles = 0;           // shadow proposals (strict continuation found)
    std::uint64_t scored_cycles = 0, shadow_right_rows = 0;
    std::uint64_t extended_cycles = 0, extension_rows = 0, accepted_rows = 0;
    std::uint64_t ambiguous = 0;        // an anchor whose earlier frames disagree on the next token
};

class MtpFrameDraft {
public:
    explicit MtpFrameDraft(MtpFrameDraftConfig config);
    void reset(std::span<const std::uint32_t> prompt);
    // As MtpNgramDraft::begin_cycle: `pending` (appended the first time), the MTP chain `drafts` of which
    // `confident` are kept, `room` the candidate capacity. Returns the proposal (possibly empty) and its learned
    // acceptance per row.
    std::vector<std::uint32_t> begin_cycle(std::uint32_t pending, std::span<const std::uint32_t> drafts,
        std::size_t confident, std::size_t room, std::vector<double>& acceptance);
    void used(std::size_t rows);
    void finish_cycle(std::span<const std::uint32_t> emitted);
    [[nodiscard]] const MtpFrameDraftStats& stats() const noexcept { return stats_; }
    [[nodiscard]] double acceptance(std::size_t row) const;
    // The strict continuation of (context + extra) inside the current frame (exposed for tests); `support` is the
    // number of earlier frame positions that agree, `ambiguous` an anchor whose continuations disagree at once.
    std::vector<std::uint32_t> lookup(std::span<const std::uint32_t> extra, std::size_t length, bool& ambiguous,
        std::size_t& support) const;
private:
    void push(std::uint32_t token);
    MtpFrameDraftConfig config_;
    std::vector<std::uint32_t> context_;
    std::vector<std::pair<std::size_t, std::size_t>> frames_;  // closed frames [open, close] in context_
    std::size_t open_at_ = SIZE_MAX;                            // the open frame's open token, SIZE_MAX: none
    std::vector<double> right_, tried_;
    bool started_ = false;
    struct Shadow {
        bool valid = false;
        std::vector<std::uint32_t> prefix, tokens;
        std::size_t used = 0;
        bool supported = false;  // two or more earlier frames agreed (its own acceptance cells)
    } shadow_;
    MtpFrameDraftStats stats_;
};

}  // namespace strix

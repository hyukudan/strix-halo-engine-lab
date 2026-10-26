#pragma once
// deepseek41 serve, armed instrumentation: per-request draft accounting of one batch step, from host data the step
// already holds (no device work). A decoding sequence that steps n rows (a window: its last token and n - 1 drafts)
// verifies n - 1 drafts and keeps accepted + 1 rows, so its generated tokens grow by accepted + 1 (exactly what
// BatchSpeculator::finish adds to drafts_verified / drafts_accepted). Prompt rows carry no drafts; a plain row or a
// one-row window proposes nothing.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::deepseek41::serve_observe {

struct Drafts {
    std::uint64_t proposed = 0, accepted = 0;
};

// The sequences of one step a tally covers: at least the batch forward's slot limit (64), so that no configuration it
// accepts can have a step with more owners. The serve also refuses --slots above it at start-up.
inline constexpr std::size_t kMaxStepOwners = 64;
static_assert(kMaxStepOwners >= 64, "a step tally must cover every slot the batch forward accepts");

// Prompt accounting of one request, the serve layer's disjoint categories (prompt = cached + imported + computed):
// cached = rows the device prefill resumed from this rank's prefix cache, imported = rows continued from an imported
// (parked) state. Kept for every request (the status's prefill rate uses them); the client usage fields report their
// union (rows not prefilled for the request).
struct PromptAccount {
    std::uint32_t cached = 0, imported = 0;
    void resumed(std::uint32_t rows) noexcept { cached = rows; imported = 0; }          // the device prefill's cache hit
    void continued(std::uint32_t position) noexcept { cached = 0; imported = position; }  // a parked imported state
    // Stats: GenerationStats (cached_tokens, imported_tokens)
    template <class Stats>
    void report(Stats& st) const noexcept {
        st.cached_tokens = cached;
        st.imported_tokens = imported;
    }
};

// Seq: has reading_prompt() and generated.size() (BatchSequence).
template <class Seq>
class StepDraftTally {
public:
    static constexpr std::size_t max_owners = kMaxStepOwners;

    // Before the step's tokens are applied: rows per owner (owner_of_row: index into seqs, per row), which owners
    // decode, and their generated sizes.
    void begin(std::span<Seq* const> seqs, std::span<const std::uint32_t> owner_of_row) noexcept {
        owners_ = std::min(seqs.size(), max_owners);
        std::fill_n(rows_.begin(), owners_, 0U);
        for (const auto o : owner_of_row)
            if (o < owners_) ++rows_[o];
        for (std::size_t i = 0; i < owners_; ++i) {
            decoding_[i] = !seqs[i]->reading_prompt();
            before_[i] = seqs[i]->generated.size();
        }
    }
    // After the step was applied (before anything trims the generated tokens): owner i's drafts of this step.
    [[nodiscard]] Drafts after(std::size_t i, const Seq& s) const noexcept {
        if (i >= owners_ || !decoding_[i] || rows_[i] < 2U || s.generated.size() <= before_[i]) return {};
        const std::uint64_t proposed = rows_[i] - 1U;
        return {proposed, std::min<std::uint64_t>(s.generated.size() - before_[i] - 1U, proposed)};
    }

private:
    std::size_t owners_ = 0;
    std::array<std::uint32_t, max_owners> rows_{};
    std::array<std::size_t, max_owners> before_{};
    std::array<bool, max_owners> decoding_{};
};

}  // namespace strix::deepseek41::serve_observe

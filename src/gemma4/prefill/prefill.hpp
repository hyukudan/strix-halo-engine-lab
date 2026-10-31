#pragma once

// g4-wmma-pf-v1 is a distinct prefill identity. Decode remains g4-v1 relative
// to its committed state. Admission requires native arithmetic and fidelity gates.
#include "gemma4/contracts/g4_types.hpp"
#include <string_view>

namespace strix::gemma4::prefill {
inline constexpr char identity[] = "g4-wmma-pf-v1";
inline constexpr unsigned tile = 16, chunk_rows = 512;
enum class Mode { exact, wmma };
// Empty/auto stays exact unless the coordinator supplies complete qualification
// admission. Numerical-band helpers alone must never set wmma_admitted.
[[nodiscard]] Mode resolve(std::string_view value, bool wmma_admitted = false);

// Pure chunk policy: scheduler fragments never determine numerical boundaries.
struct Chunk {
    std::uint64_t position = 0;
    unsigned rows = 0, padded_rows = 0;
};
[[nodiscard]] std::vector<Chunk> plan(std::uint64_t frontier, std::size_t rows,
                                      std::uint32_t admitted_context = max_positions);
struct ChunkResult {
    unsigned rows = 0, vocab = 0, tap_width = 0;
    std::vector<float> last_logits, taps;
    std::vector<float> all_logits;
    std::vector<float> hidden; // rows x hidden final-norm rows, only when requested
};
struct Result {
    std::uint64_t frontier = 0;
    std::vector<float> last_logits, tail_taps;
    std::vector<std::uint64_t> tail_positions;
};
// Append a retained committed tail once, in groups admitted by device draft
// context (at most eight rows). Validate the complete tail before publication.
void append_tail(DraftContext& context, const Result& result, unsigned hidden_width);

// FP32 -> FP16, RNE, gradual underflow, signed zero. False means nonfinite or
// rounding overflow; callers must fail the request, never silently clip.
STRIX_G4_HD inline bool half_bits(float x, std::uint16_t& out) {
    const auto u = bit_cast<std::uint32_t>(x), a = u & 0x7fffffffU;
    const auto sign = std::uint16_t((u >> 16U) & 0x8000U);
    if (a >= 0x7f800000U) return false;
    if (a < 0x33000000U) { out = sign; return true; }
    if (a < 0x38800000U) {
        const unsigned shift = 126U - (a >> 23U);
        const auto m = (a & 0x7fffffU) | 0x800000U;
        const auto q = m >> shift, remainder = m & ((1U << shift) - 1U);
        const auto midpoint = 1U << (shift - 1U);
        out = std::uint16_t(sign | (q + unsigned(remainder > midpoint || (remainder == midpoint && (q & 1U)))));
        return true;
    }
    const auto q = ((a - 0x38000000U) + 0xfffU + ((a >> 13U) & 1U)) >> 13U;
    if (q >= 0x7c00U) return false;
    out = std::uint16_t(sign | q);
    return true;
}

struct Fidelity {
    // These are numerical bands only. Corpus coverage, provenance, serial,
    // speculation, overflow and native opcode gates are separate admission gates.
    std::size_t positions = 0, top1_matches = 0, values = 0;
    double sum_kl = 0.0, squared_error = 0.0, maximum_error = 0.0;
    std::vector<double> kl;
    // Full vocabulary, capped logits, identical teacher-forced prefix. KL(ref||candidate), FP64.
    void observe(std::span<const float> reference, std::span<const float> candidate);
    [[nodiscard]] double mean_kl() const;
    [[nodiscard]] double p99_kl() const;
    [[nodiscard]] double top1() const;
    [[nodiscard]] double rmse() const;
    [[nodiscard]] bool same_weight_bands_pass() const;
    [[nodiscard]] bool bf16_bands_pass() const;
};
} // namespace strix::gemma4::prefill

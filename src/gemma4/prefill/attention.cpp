#include "attention.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strix::gemma4::prefill {
AttentionMode resolve_attention(std::string_view value) {
    if (value.empty() || value == "auto" || value == "canonical") return AttentionMode::canonical;
    if (value == "wmma" || value == attention_identity) return AttentionMode::wmma;
    throw std::invalid_argument("unknown prefill attention identity");
}
AttentionMode attention_mode() {
    static const auto mode = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_ATTN"); return resolve_attention(p ? p : ""); }();
    return mode;
}
bool attention_pack_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_ATTN_PACK"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
namespace {
std::uint16_t half(float value) {
    std::uint16_t result = 0;
    if (!half_bits(value, result)) throw std::runtime_error("WMMA attention FP16 overflow/nonfinite operand");
    return result;
}
void finite(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite WMMA attention arithmetic");
}
} // namespace
std::vector<float> emulate_attention(std::span<const float> q, std::uint64_t position, unsigned window_size,
                                      const AttentionRead& read) {
    if (q.empty() || q.size() > 512 || !read || position >= max_positions)
        throw std::invalid_argument("WMMA attention emulator geometry");
    const auto first = window_size && position >= window_size ? position - window_size + 1 : 0;
    std::vector<std::uint16_t> query((q.size() + 15) / 16 * 16);
    for (std::size_t d = 0; d < q.size(); ++d) query[d] = half(q[d]);
    std::vector<float> numerator(q.size(), 0.0F);
    float maximum = -std::numeric_limits<float>::infinity(), denominator = 0.0F;
    for (auto base = first / 16 * 16; base <= position; base += 16) {
        std::array<float, 16> scores;
        scores.fill(-std::numeric_limits<float>::infinity());
        float next_maximum = maximum;
        for (unsigned j = 0; j < 16; ++j) if (base + j >= first && base + j <= position) {
            const auto* k = read(base + j, false);
            float acc = 0.0F;
            for (std::size_t d = 0; d < query.size(); d += 16) {
                std::array<std::uint16_t, 16> a{}, b{};
                for (unsigned i = 0; i < 16; ++i) { a[i] = query[d + i]; b[i] = d + i < q.size() ? half(k[d + i]) : 0; }
                acc = instruction(a, b, acc);
            }
            finite(acc); scores[j] = acc; next_maximum = std::max(next_maximum, acc);
        }
        const float alpha = denominator == 0.0F ? 0.0F : declared_exp(maximum - next_maximum);
        denominator = denominator * alpha;
        std::array<std::uint16_t, 16> probabilities{};
        for (unsigned j = 0; j < 16; ++j) if (base + j >= first && base + j <= position) {
            const float p = declared_exp(scores[j] - next_maximum);
            denominator = denominator + p; probabilities[j] = half(p);
        }
        for (unsigned d = 0; d < q.size(); ++d) {
            std::array<std::uint16_t, 16> positive{}, negative{};
            for (unsigned j = 0; j < 16; ++j) if (base + j >= first && base + j <= position) {
                const auto value = half(read(base + j, true)[d]);
                if (value & 0x8000U) negative[j] = value & 0x7fffU;
                else positive[j] = value;
            }
            // Nonnegative PV operands make masked products +0 regardless of
            // other queries' visible V. Opposite-sign zero products otherwise
            // perturb the hardware accumulator and break row invariance.
            const float part = instruction(probabilities, positive, 0.0F) - instruction(probabilities, negative, 0.0F);
            numerator[d] = numerator[d] * alpha + part;
            finite(numerator[d]);
        }
        finite(denominator); maximum = next_maximum;
    }
    for (auto& value : numerator) { value = value / denominator; finite(value); }
    return numerator;
}
} // namespace strix::gemma4::prefill

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace strix::core::residual_mix {

inline constexpr unsigned kMaxBanks = 8;
inline constexpr unsigned kMaxQueries = 8;

struct NormalizeArgs {
    const float* input{};
    float* output{};
    unsigned rows{};
    unsigned width{};
    float epsilon{};
    std::size_t input_stride{};   // Zero selects width.
    std::size_t output_stride{};
    bool round_bf16{};
};

struct NormalizePairArgs {
    NormalizeArgs first, second;
};

struct MixArgs {
    const float* banks[kMaxBanks]{};   // Each bank is [rows][width].
    const float* queries{};           // [query_count][width], shared by tokens.
    float* output[kMaxQueries]{};     // Each output is [rows][width], disjoint from banks.
    unsigned rows{};
    unsigned width{};
    unsigned bank_count{};
    unsigned query_count{};
    float epsilon{};
    bool normalized{};               // Banks are already normalized as FP32.
};

struct AddArgs {
    const float* lhs{};
    const float* rhs{};
    float* output{};
    std::size_t elements{};
    bool round_bf16{};
};

struct RoundArgs {
    const float* input{};
    float* output{};
    std::size_t elements{};
};

struct ControllerArgs {
    const float* current{};
    const float* message{};
    const float* forget{};
    const float* gain{};
    float* output{};
    std::size_t elements{};
};

namespace detail {

inline constexpr std::size_t kMaxElements = std::numeric_limits<std::size_t>::max() / sizeof(float);
inline constexpr unsigned kThreads = 256;

inline bool valid_rows(unsigned rows, unsigned width, std::size_t stride) {
    return width != 0U && stride >= width && rows <= kMaxElements / stride;
}

}  // namespace detail

inline bool fits(NormalizeArgs args) {
    if (args.rows == 0U) return true;
    return args.input && args.output && args.epsilon > 0.0F && std::isfinite(args.epsilon) &&
           detail::valid_rows(args.rows, args.width, args.input_stride ? args.input_stride : args.width) &&
           detail::valid_rows(args.rows, args.width, args.output_stride ? args.output_stride : args.width);
}

inline bool fits(const MixArgs& args) {
    if (args.rows == 0U) return true;
    if (!args.queries || args.bank_count == 0U || args.bank_count > kMaxBanks ||
        args.query_count == 0U || args.query_count > kMaxQueries ||
        !detail::valid_rows(args.rows, args.width, args.width) ||
        !detail::valid_rows(args.query_count, args.width, args.width) ||
        (!args.normalized && (!(args.epsilon > 0.0F) || !std::isfinite(args.epsilon)))) return false;
    for (unsigned bank = 0; bank < args.bank_count; ++bank)
        if (!args.banks[bank]) return false;
    for (unsigned query = 0; query < args.query_count; ++query) {
        if (!args.output[query] || args.output[query] == args.queries) return false;
        for (unsigned bank = 0; bank < args.bank_count; ++bank)
            if (args.banks[bank] == args.output[query]) return false;
        for (unsigned other = 0; other < query; ++other)
            if (args.output[other] == args.output[query]) return false;
    }
    return true;
}

}  // namespace strix::core::residual_mix

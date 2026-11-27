#pragma once
// Plan of the block-per-row low-rank controller (decode rows): one launch, one 256-thread block per row, runs the
// controller's readout, the two operand norms, the down projection, the [forget, gain] gate pair, the controller and the
// final norm that enqueue_low_rank_controller issues as six launches. Host-side shapes only (no device header): the
// device body is in core/controller_block_rows.hpp.
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix::core::controller_block {

// Rows of one launch (decode, verify and cohort rows); taller calls keep the tiled launches.
inline constexpr unsigned kMaxRows = 16U;
// Fewest rows the decode route hands to the block form. One block per row streams the controller's 1.18 MB of weights
// from one compute unit (gfx1151 graph bench, 96 controllers with their own weights: 42-48 us per controller at any row
// count), while the served launches spread them (23 us at 1 row, 39 us at 8, 81-99 us at 12-16 rows): the block form
// is the faster one only past 8 rows.
inline constexpr unsigned kMinRows = 9U;
constexpr bool preferred(unsigned rows) noexcept { return rows >= kMinRows && rows <= kMaxRows; }
inline constexpr unsigned kThreads = 256U;
// The gate pair's K: one 16-value chunk per lane of a four-lane group (the chunk order of the wave-sum form).
inline constexpr unsigned kRank = 64U;

// Values per thread of a row (width / kThreads) for the widths the device body instantiates; 0 = not instantiated.
constexpr unsigned row_regs(unsigned width) noexcept { return width == 2304U ? 9U : 0U; }

struct Args {
    const float* raw{};       // [rows][width] residual rows (may be the output rows: read before they are stored)
    const float* bank0{};     // [rows][width] normalized readout banks; both null where `message` is given
    const float* bank1{};
    const float* message{};   // [rows][width] readout already produced (used when the banks are null)
    const float* query{};     // [width] learned readout row (banks only)
    const std::uint16_t* down{};    // [rank][2 * width] BF16
    const std::uint16_t* forget{};  // [width][rank] BF16
    const std::uint16_t* gain{};    // [width][rank] BF16
    const std::uint16_t* scale{};   // [width] BF16 final norm scale
    float* output{};          // [rows][width]
    unsigned rows{};
    unsigned width{};
    unsigned rank{};
    float operand_epsilon{};  // epsilon of the two operand norms
    float norm_epsilon{};     // epsilon of the final norm
    bool round_operands{};    // the down operands are BF16-rounded (the only form the device body runs)
    bool round_latent{};      // the latent is BF16-rounded (likewise)
};

namespace detail {
inline bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a), b0 = reinterpret_cast<std::uintptr_t>(b);
    return !(a0 + a_bytes <= b0 || b0 + b_bytes <= a0);
}
}  // namespace detail

// Admission: the shape the device body is built for, 16-byte aligned weight rows, and output apart from everything the
// blocks read, except that the raw rows may be the output rows (every block reads its raw row before it stores it).
inline bool admits(const Args& a) noexcept {
    const bool banks = a.bank0 != nullptr && a.bank1 != nullptr;
    const bool given = a.message != nullptr && a.bank0 == nullptr && a.bank1 == nullptr;
    if (!(banks != given) || !a.raw || !a.down || !a.forget || !a.gain || !a.scale || !a.output || (banks && !a.query))
        return false;
    if (a.rows == 0U || a.rows > kMaxRows || a.rank != kRank || row_regs(a.width) == 0U || !a.round_operands ||
        !a.round_latent || !(a.operand_epsilon > 0.0F) || !(a.norm_epsilon > 0.0F) ||
        a.operand_epsilon == std::numeric_limits<float>::infinity() ||
        a.norm_epsilon == std::numeric_limits<float>::infinity())
        return false;
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16U == 0U; };
    if (!aligned(a.down) || !aligned(a.forget) || !aligned(a.gain)) return false;
    const std::size_t rows_bytes = std::size_t{a.rows} * a.width * sizeof(float);
    const std::size_t row_bytes = std::size_t{a.width} * sizeof(float);
    const std::size_t weight_bytes[] = {std::size_t{a.rank} * 2U * a.width * 2U, std::size_t{a.width} * a.rank * 2U,
                                        std::size_t{a.width} * a.rank * 2U, std::size_t{a.width} * 2U};
    const void* const weights[] = {a.down, a.forget, a.gain, a.scale};
    for (unsigned i = 0; i < 4U; ++i)
        if (detail::overlaps(a.output, rows_bytes, weights[i], weight_bytes[i])) return false;
    if (a.raw != a.output && detail::overlaps(a.output, rows_bytes, a.raw, rows_bytes)) return false;
    const float* const operands[] = {a.bank0, a.bank1, a.message};
    for (const float* p : operands)
        if (p && detail::overlaps(a.output, rows_bytes, p, rows_bytes)) return false;
    if (a.query && detail::overlaps(a.output, rows_bytes, a.query, row_bytes)) return false;
    return true;
}

}  // namespace strix::core::controller_block

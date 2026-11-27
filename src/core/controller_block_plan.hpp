#pragma once
// Plan of the fused low-rank controller (decode rows): the controller's readout, the two operand norms, the down
// projection, the [forget, gain] gate pair, the controller and the final norm that enqueue_low_rank_controller issues as
// six launches, as two launches with the bits of those launches. Host-side shapes only (no device header):
// the device bodies are in core/controller_block_rows.hpp.
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix::core::controller_block {

// Rows of one launch (decode, verify and cohort rows); taller calls keep the tiled launches.
inline constexpr unsigned kMaxRows = 16U;
inline constexpr unsigned kThreads = 256U;
// The gate pair's K: one 16-value chunk per lane of a four-lane group (the chunk order of the wave-sum form).
inline constexpr unsigned kRank = 64U;
// The down launch: one latent per wave of a 256-thread block, so kRank / 8 blocks per row.
inline constexpr unsigned kDownBlocks = kRank / (kThreads / 32U);
// The gate launch: outputs per block (two steps of eight waves x eight outputs).
inline constexpr unsigned kGateOutputs = 128U;

// Device form: two launches. (kDownBlocks, rows) blocks each recompute the row's readout and operand norms and produce
// eight latents; (width / kGateOutputs, rows) blocks each produce kGateOutputs gate sums and their controller values, and
// the last of a row's gate blocks to finish (device-scope ticket) runs the final norm over the whole row. The weight
// bytes spread over many compute units. gfx1151 bench (96 controllers with their own weights per graph, us per
// controller, rows 1 / 2 / 4 / 8 / 9 / 12 / 16): fused 15.5 / 15.9 / 17.7 / 22.3 / 22.2 / 30.8 / 32.1 against the four
// served launches 22.4 / 25.5 / 29.8 / 39.4 / 71.3 / 72.8 / 75.6 (and a one-block-per-row launch, 41.6-42.2 at every row
// count: one compute unit streams the 1.18 MB of weights). The decode route takes it at every row count of the band.
constexpr bool preferred(unsigned rows) noexcept { return rows >= 1U && rows <= kMaxRows; }

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
    float* scratch{};         // rows the launches write and read (scratch_elements_for); apart from every row above
    std::size_t scratch_elements{};
};

// Scratch (floats): [rows][width] controller rows, [rows][width] readout rows (banks form), [rows][kRank] latents, then
// one 32-bit ticket per row (the down launch zeroes it, the gate launch counts its blocks on it).
constexpr std::size_t scratch_elements_for(unsigned rows, unsigned width) noexcept {
    return std::size_t{rows} * (2U * std::size_t{width} + kRank + 1U);
}

namespace detail {
inline bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a), b0 = reinterpret_cast<std::uintptr_t>(b);
    return !(a0 + a_bytes <= b0 || b0 + b_bytes <= a0);
}
}  // namespace detail

// Admission: the shape the device body is built for, 16-byte aligned weight rows, output apart from everything the
// blocks read, except that the raw rows may be the output rows (the final norm stores a row after every block of the row
// has read it), and a 16-byte aligned scratch of scratch_elements_for apart from every row and weight.
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
    if (a.width % kGateOutputs != 0U || !a.scratch) return false;
    const std::size_t need = scratch_elements_for(a.rows, a.width);
    if (a.scratch_elements < need || !aligned(a.scratch)) return false;
    const std::size_t scratch_bytes = need * sizeof(float);
    const float* const rows[] = {a.raw, a.bank0, a.bank1, a.message, a.output};
    for (const float* p : rows)
        if (p && detail::overlaps(a.scratch, scratch_bytes, p, rows_bytes)) return false;
    if (a.query && detail::overlaps(a.scratch, scratch_bytes, a.query, row_bytes)) return false;
    for (unsigned i = 0; i < 4U; ++i)
        if (detail::overlaps(a.scratch, scratch_bytes, weights[i], weight_bytes[i])) return false;
    return true;
}

}  // namespace strix::core::controller_block

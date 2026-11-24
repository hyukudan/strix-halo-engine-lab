// Host check of the microscaled R1 rows chunking (launch_rows): for every batch size, each routed row lands in exactly
// one chunk of at most kMaxRows rows, at the chunk-local position whose input, ids, weights, failure word and output
// are the batch row's own (row_offsets of the batch row), and every chunk restarts the activation scratch at its base.
// The kernels then run the one-row body per row, so the row's bits are those of a one-row launch for any batch size.
#include "moe_core/microscaled_r1.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;

constexpr unsigned kTopK = 8U, kHidden = 48U, kIntermediate = 16U;

bool check_rows(unsigned rows, bool with_failure) {
    std::vector<unsigned> ids(std::size_t{rows} * kTopK), failure(rows);
    std::vector<float> weights(ids.size()), input(std::size_t{rows} * kHidden), output(input.size());
    std::vector<std::uint16_t> activation(std::size_t{r1::kMaxRows} * kTopK * kIntermediate);
    const r1::Args batch{nullptr, ids.data(), weights.data(), with_failure ? failure.data() : nullptr, input.data(),
                         activation.data(), output.data(), 128U, kHidden, kIntermediate, 4.0F, 25.0F};
    std::vector<unsigned> seen(rows, 0U);
    unsigned chunks = 0;
    // The launchers' own iteration (for_each_rows_chunk): `true` = launched, `false` = refused.
    const bool ok = r1::for_each_rows_chunk(rows, true, [&](unsigned first, unsigned count) {
        ++chunks;
        if (count == 0U || count > r1::kMaxRows) return false;
        const auto chunk = r1::rows_chunk<kTopK>(batch, first);
        if (chunk.activation != activation.data() || chunk.matrices != batch.matrices || chunk.hidden != kHidden ||
            chunk.intermediate != kIntermediate || chunk.experts != batch.experts)
            return false;
        for (unsigned local = 0; local < count; ++local) {
            const unsigned row = first + local;
            const auto mine = r1::row_offsets<kTopK>(row, kHidden, kIntermediate);
            const auto at = r1::row_offsets<kTopK>(local, kHidden, kIntermediate);
            if (chunk.input + at.input != input.data() + mine.input ||
                chunk.output + at.output != output.data() + mine.output ||
                chunk.ids + at.route != ids.data() + mine.route ||
                chunk.weights + at.route != weights.data() + mine.route)
                return false;
            if (with_failure ? chunk.failure + local != failure.data() + row : chunk.failure != nullptr) return false;
            // The chunk's activation rows stay inside the kMaxRows scratch.
            if (at.activation + std::size_t{kTopK} * kIntermediate > activation.size()) return false;
            ++seen[row];
        }
        return true;
    });
    if (!ok || chunks != r1::rows_chunk_count(rows)) return false;
    for (unsigned count : seen)
        if (count != 1U) return false;
    return true;
}

// Near UINT_MAX (no batch allocated): the count does not wrap, the last chunk ends exactly at rows, every row is
// covered once (sizes sum to rows) and the iteration stops; a refused chunk stops the launches at that chunk.
bool check_wrap_edge() {
    constexpr unsigned kMax = std::numeric_limits<unsigned>::max();
    static_assert(r1::rows_chunk_count(kMax) == kMax / r1::kMaxRows + 1U);
    static_assert(r1::rows_chunk_count(kMax - kMax % r1::kMaxRows) == kMax / r1::kMaxRows);
    static_assert(r1::rows_chunk_size(kMax, kMax - kMax % r1::kMaxRows) == kMax % r1::kMaxRows);
    for (unsigned rows : {kMax, kMax - kMax % r1::kMaxRows}) {
        unsigned calls = 0, last_first = 0, last_count = 0;
        std::uint64_t covered = 0, expected_first = 0;
        bool ordered = true;
        const int status = r1::for_each_rows_chunk(rows, 0, [&](unsigned first, unsigned count) {
            ordered = ordered && first == expected_first && count >= 1U && count <= r1::kMaxRows;
            expected_first = std::uint64_t{first} + count;
            covered += count;
            last_first = first;
            last_count = count;
            ++calls;
            return 0;
        });
        const unsigned tail = rows % r1::kMaxRows == 0U ? r1::kMaxRows : rows % r1::kMaxRows;
        if (status != 0 || !ordered || calls != r1::rows_chunk_count(rows) || covered != rows ||
            last_count != tail || last_first != rows - tail) {
            std::printf("FAIL wrap edge rows %u: calls %u last %u+%u\n", rows, calls, last_first, last_count);
            return false;
        }
    }
    unsigned calls = 0;
    const int refused = r1::for_each_rows_chunk(kMax, 0, [&](unsigned first, unsigned) {
        ++calls;
        return first == 2U * r1::kMaxRows ? 7 : 0;
    });
    return refused == 7 && calls == 3U;
}

// rows_cta: every CTA of a rows launch maps to a distinct (unit, block) and every pair is reached, for any tile; a
// tile's CTAs of one unit are consecutive blocks; tile = blocks is the (block, unit) order of the one-row grids.
bool check_rows_order(unsigned units, unsigned blocks, unsigned tile) {
    const r1::RowsOrder order{units, blocks, tile};
    std::vector<unsigned char> seen(std::size_t{units} * blocks, 0U);
    for (unsigned linear = 0; linear < units * blocks; ++linear) {
        const auto cta = r1::rows_cta(order, linear);
        if (cta.unit >= units || cta.block >= blocks) return false;
        auto& hit = seen[std::size_t{cta.unit} * blocks + cta.block];
        if (hit != 0U) return false;
        hit = 1U;
        if (tile == blocks && (cta.unit != linear / blocks || cta.block != linear % blocks)) return false;
        if (linear > 0U) {
            const auto before = r1::rows_cta(order, linear - 1U);
            const bool same_tile = before.block / tile == cta.block / tile;
            if (same_tile && before.unit == cta.unit && before.block + 1U != cta.block) return false;
        }
    }
    return true;
}

} // namespace

int main() {
    static_assert(r1::rows_chunk_count(1U) == 1U && r1::rows_chunk_count(r1::kMaxRows) == 1U);
    static_assert(r1::rows_chunk_count(r1::kMaxRows + 1U) == 2U && r1::rows_chunk_count(16U) == 2U);
    // Tile policy on the Inkling-shaped gate_up (intermediate 1152, hidden 2304, top-8, 2 columns per wave): the
    // largest tile whose span fits the measured budget; no budget keeps the (block, unit) order.
    using packed = strix::weight::microscaled::MatrixFormat;
    namespace tiles = strix::r1_rows_tiles;
    // A plan holds only for the workload it was measured on (strix/r1_rows_tiles.hpp): gfx1151 / 20 WGPs, 128 experts,
    // 2304 x 1152, top-8, NVFP4 or MXFP8, rows 4 and 8. Every other key or row count resolves to the incumbent order.
    constexpr auto budget = [](std::string_view arch, unsigned mp, packed format, unsigned experts, unsigned hidden,
                               unsigned intermediate, unsigned top_k, unsigned rows) {
        return tiles::gate_up_cache_bytes(arch, mp, format, experts, hidden, intermediate, top_k, rows);
    };
    constexpr std::uint64_t mib24 = std::uint64_t{24U} << 20U;
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 4U) == mib24);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == mib24);
    static_assert(budget("gfx1151:sramecc-:xnack-", 20U, packed::mxfp8, 128U, 2304U, 1152U, 8U, 8U) == mib24);
    // Row counts: 1 / 2 measured without gain, 3 / 5 / 6 / 7 not measured, 9+ past the table.
    for (unsigned rows : {0U, 1U, 2U, 3U, 5U, 6U, 7U, 9U, 16U}) {
        for (packed format : {packed::nvfp4, packed::mxfp8}) {
            if (budget("gfx1151", 20U, format, 128U, 2304U, 1152U, 8U, rows) != 0U) {
                std::printf("FAIL rows %u has a tile plan\n", rows);
                return 1;
            }
        }
    }
    // Every other key field: architecture, multiprocessors, bank size (the review's 256-expert bank), hidden,
    // intermediate, top_k.
    static_assert(budget("gfx1201", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 16U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 256U, 2304U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 64U, 2304U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 128U, 4096U, 1152U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1536U, 8U, 8U) == 0U);
    static_assert(budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 4U, 8U) == 0U);
    // Uncovered -> the incumbent order (tile == blocks: CTA i = (i / blocks, i % blocks)).
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 5U, 8U, 2U,
                                                         budget("gfx1151", 20U, packed::nvfp4, 128U, 2304U, 1152U, 8U, 5U))
                      .tile == 72U);
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 8U, 8U, 2U,
                                                         budget("gfx1151", 20U, packed::nvfp4, 256U, 2304U, 1152U, 8U, 8U))
                      .tile == 72U);
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 2U, 8U, 2U, 0U).tile == 72U);
    // Covered -> the measured tiles.
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 4U, 8U, 2U, mib24).tile == 18U);
    static_assert(r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, 8U, 8U, 2U, mib24).tile == 9U);
    static_assert(r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, 8U, 8U, 2U, mib24).tile == 5U);
    static_assert(r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, 4U, 8U, 2U, mib24).tile == 10U);
    // A single block's span above the budget keeps the launch order.
    static_assert(r1::rows_order(64U, 72U, (std::uint64_t{1U} << 20U), mib24).tile == 72U);
    static_assert(r1::rows_order(64U, 72U, mib24 / 64U, mib24).tile == 1U);
    // Launch grids, every form, 64-bit: x <= 0x7FFFFFFF CTAs, y <= 0xFFFF CTAs, <= 0xFFFFFFFF work-items per axis. With
    // 256 work-items per CTA the last legal x is 0xFFFFFF CTAs (one CTA more is exactly 2^32 work-items).
    constexpr unsigned threads = r1::kBlockThreads;
    static_assert(threads == 256U);
    static_assert(r1::grid_launchable(0xFFFFFFU, 1U, threads) && !r1::grid_launchable(0x1000000U, 1U, threads));
    static_assert(r1::grid_launchable(0x7FFFFFFFU, 1U, 1U) && !r1::grid_launchable(0x80000000U, 1U, 1U));
    static_assert(r1::grid_launchable(1U, 0xFFFFU, threads) && !r1::grid_launchable(1U, 0x10000U, threads));
    static_assert(!r1::grid_launchable(0U, 1U, threads) && !r1::grid_launchable(1U, 0U, threads));
    // The flat rows grid (units x blocks on x) falls back to the 2D form past these limits.
    static_assert(r1::rows_order_launchable({64U, 72U, 9U}, threads));
    static_assert(r1::rows_order_launchable({1U, 0xFFFFFFU, 1U}, threads));
    static_assert(!r1::rows_order_launchable({1U, 0x1000000U, 1U}, threads));
    static_assert(r1::rows_order_launchable({256U, 65535U, 65535U}, threads));
    static_assert(!r1::rows_order_launchable({256U, 65536U, 65536U}, threads));
    static_assert(!r1::rows_order_launchable({256U, 16777217U, 16777217U}, threads));
    // The 2D forms (gate (gate_blocks, rows x top_k), down (down_blocks, rows)): a legal fallback is admitted, an
    // unrepresentable one refused. gate_blocks(intermediate, 2) = intermediate / 16: 268,435,440 -> 0xFFFFFF blocks
    // (x = 2^32 - 256 work-items), 268,435,456 -> 2^24 blocks (exactly 2^32 work-items: the review's NVFP4 rows 2,
    // top-1, hidden 32 call).
    constexpr r1::Geometry g = r1::kGeometry;
    static_assert(r1::rows_grids_launchable(32U, 268435440U, 2U, 1U, g, threads));
    static_assert(!r1::rows_grids_launchable(32U, 268435456U, 2U, 1U, g, threads));
    // The r12 example (8 rows, top-32, intermediate 1,048,576): flat grid refused, the 2D fallback (65,536 x 256) legal.
    static_assert(!r1::rows_order_launchable(r1::gate_up_rows_order<packed::nvfp4>(1048576U, 2304U, 8U, 32U, 2U, 0U),
                                             threads));
    static_assert(r1::rows_grids_launchable(2304U, 1048576U, 8U, 32U, g, threads));
    // down_blocks(hidden, 4) = hidden / 4: hidden 67,108,800 -> 0xFFFFF0 blocks pass, 67,108,864 -> 2^24 refused.
    static_assert(r1::rows_grids_launchable(67108800U, 1152U, 1U, 8U, g, threads));
    static_assert(!r1::rows_grids_launchable(67108864U, 1152U, 1U, 8U, g, threads));
    unsigned failures = check_wrap_edge() ? 0U : 1U;
    for (unsigned units : {1U, 2U, 3U, 8U, 16U, 64U}) {
        for (unsigned blocks : {1U, 5U, 72U, 576U}) {
            for (unsigned tile : {1U, 2U, 7U, 8U, 64U, 72U, 576U}) {
                if (tile > blocks) continue;
                if (!check_rows_order(units, blocks, tile)) {
                    std::printf("FAIL rows order units %u blocks %u tile %u\n", units, blocks, tile);
                    ++failures;
                }
            }
        }
    }
    for (unsigned rows = 1; rows <= r1::kMaxRows; ++rows) {
        const auto nv = r1::gate_up_rows_order<packed::nvfp4>(1152U, 2304U, rows, 8U, 2U, mib24);
        const auto mx = r1::gate_up_rows_order<packed::mxfp8>(1152U, 2304U, rows, 8U, 2U, mib24);
        if (!check_rows_order(nv.units, nv.blocks, nv.tile) || !check_rows_order(mx.units, mx.blocks, mx.tile)) {
            std::printf("FAIL rows order policy rows %u\n", rows);
            ++failures;
        }
    }
    for (unsigned rows = 1; rows <= 4U * r1::kMaxRows + 3U; ++rows) {
        for (bool with_failure : {true, false}) {
            if (!check_rows(rows, with_failure)) {
                std::printf("FAIL rows %u failure words %d\n", rows, with_failure ? 1 : 0);
                ++failures;
            }
        }
    }
    if (failures != 0U) return 1;
    std::printf("microscaled_r1 rows chunking: PASS (rows 1..%u, UINT_MAX edge)\n", 4U * r1::kMaxRows + 3U);
    return 0;
}

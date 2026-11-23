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

} // namespace

int main() {
    static_assert(r1::rows_chunk_count(1U) == 1U && r1::rows_chunk_count(r1::kMaxRows) == 1U);
    static_assert(r1::rows_chunk_count(r1::kMaxRows + 1U) == 2U && r1::rows_chunk_count(16U) == 2U);
    unsigned failures = check_wrap_edge() ? 0U : 1U;
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

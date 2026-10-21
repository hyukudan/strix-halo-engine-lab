#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_formats.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <vector>

// ds41-expert-tp2-i1152-v1 host pieces: the local orders, the packed half rows (codes, scales, d/dmin of every group
// equal to the unsplit row's), the K-half product (equal to product_q2_k on whole rows) and the split's closeness to the
// unsplit expert in exact arithmetic.
namespace {

namespace tp = strix::ds41::expert_tp;
namespace ds41 = strix::ds41;

int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::vector<std::uint8_t> random_q2_row(std::mt19937& rng) {
    std::vector<std::uint8_t> row(tp::inter / 256U * ds41::q2_k_block_bytes);
    for (auto& b : row) b = static_cast<std::uint8_t>(rng());
    for (std::size_t blk = 0; blk < tp::inter / 256U; ++blk) {  // finite FP16 d and dmin (0x2000..0x2fff: small positives)
        auto* p = row.data() + blk * ds41::q2_k_block_bytes + 80U;
        for (unsigned k = 0; k < 2U; ++k) {
            const auto h = static_cast<std::uint16_t>(0x2000U + (rng() & 0x0fffU));
            p[2U * k] = static_cast<std::uint8_t>(h & 0xffU);
            p[2U * k + 1U] = static_cast<std::uint8_t>(h >> 8U);
        }
    }
    return row;
}

void test_orders() {
    std::set<std::uint32_t> inter, groups;
    bool group_aligned = true;
    for (unsigned r = 0; r < tp::ranks; ++r) {
        for (std::uint32_t j = 0; j < tp::half; ++j) inter.insert(tp::inter_index(r, j));
        for (std::uint32_t g = 0; g < tp::half_groups; ++g) {
            groups.insert(tp::group_index(r, g));
            for (std::uint32_t e = 0; e < 32U; ++e) group_aligned = group_aligned && tp::inter_index(r, 32U * g + e) == 32U * tp::group_index(r, g) + e;
            group_aligned = group_aligned && (r == 0U ? tp::group_index(r, g) < 36U : tp::group_index(r, g) >= 36U);
        }
    }
    expect(inter.size() == tp::inter, "the two ranks' intermediate rows cover 0..2303 once");
    expect(groups.size() == tp::inter / 32U, "the two ranks' down groups cover 0..71 once");
    expect(group_aligned, "local orders move whole 32-element groups; rank 0 holds [0, 1152), rank 1 [1152, 2304)");
}

void test_pack(std::mt19937& rng) {
    bool same = true;
    for (int t = 0; t < 64; ++t) {
        const auto row = random_q2_row(rng);
        for (unsigned r = 0; r < tp::ranks; ++r) {
            std::vector<std::uint8_t> packed(tp::down_row_bytes);
            tp::pack_down_row(row, r, packed);
            for (std::uint32_t g = 0; g < tp::half_groups; ++g) {
                const auto gg = tp::group_index(r, g);
                const auto a = ds41::q2_k_group(std::span<const std::uint8_t, ds41::q2_k_block_bytes>(packed.data() + g / 8U * ds41::q2_k_block_bytes, ds41::q2_k_block_bytes), g % 8U);
                const auto b = ds41::q2_k_group(std::span<const std::uint8_t, ds41::q2_k_block_bytes>(row.data() + gg / 8U * ds41::q2_k_block_bytes, ds41::q2_k_block_bytes), gg % 8U);
                same = same && a.codes == b.codes && a.scale[0] == b.scale[0] && a.scale[1] == b.scale[1] &&
                       std::bit_cast<std::uint32_t>(a.d) == std::bit_cast<std::uint32_t>(b.d) &&
                       std::bit_cast<std::uint32_t>(a.dmin) == std::bit_cast<std::uint32_t>(b.dmin);
            }
        }
    }
    expect(same, "every packed half-row group decodes to the unsplit row's group (codes, scales, d, dmin)");
    bool tail = true;
    for (int t = 0; t < 64; ++t) {
        const auto row = random_q2_row(rng);
        for (unsigned r = 0; r < tp::ranks; ++r) {
            std::vector<std::uint8_t> packed(tp::down_row_bytes_packed);
            tp::pack_down_row_packed(row, r, packed);
            for (std::uint32_t g = 0; g < tp::half_groups; ++g) {
                const auto gg = tp::group_index(r, g);
                const auto a = tp::packed_group(packed, g);
                const auto b = ds41::q2_k_group(std::span<const std::uint8_t, ds41::q2_k_block_bytes>(row.data() + gg / 8U * ds41::q2_k_block_bytes, ds41::q2_k_block_bytes), gg % 8U);
                tail = tail && a.codes == b.codes && a.scale[0] == b.scale[0] && a.scale[1] == b.scale[1] &&
                       std::bit_cast<std::uint32_t>(a.d) == std::bit_cast<std::uint32_t>(b.d) &&
                       std::bit_cast<std::uint32_t>(a.dmin) == std::bit_cast<std::uint32_t>(b.dmin);
            }
        }
    }
    expect(tail, "every group of the packed half-block row (380 B) decodes to the unsplit row's group");
    // gate rows
    const std::uint64_t rb = 66U;
    std::vector<std::uint8_t> expert(rb * tp::inter);
    for (auto& b : expert) b = static_cast<std::uint8_t>(rng());
    bool rows = true;
    for (unsigned r = 0; r < tp::ranks; ++r) {
        std::vector<std::uint8_t> out(rb * tp::half);
        tp::pack_gate_rows(expert, rb, r, out);
        for (std::uint32_t j = 0; j < tp::half; ++j)
            rows = rows && std::memcmp(out.data() + j * rb, expert.data() + std::uint64_t{tp::inter_index(r, j)} * rb, rb) == 0;
    }
    expect(rows, "packed gate/up rows are the unsplit rows in the local order");
}

void test_product(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-3.0F, 3.0F);
    bool whole = true;
    double worst = 0.0;
    for (int t = 0; t < 32; ++t) {
        const auto row = random_q2_row(rng);
        std::vector<float> act(tp::inter);
        for (auto& v : act) v = u(rng);
        const auto x = ds41::quantize_a8(act);
        // whole row: the K-half reader over 72 groups is product_q2_k
        const float a = tp::product_q2_k_groups(row, x, tp::inter / 32U), b = ds41::product_q2_k(row, x, tp::inter);
        whole = whole && std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
        // halves: local activations of each rank (the gate/up of the rank quantizes the same groups)
        double split = 0.0, exact = 0.0;
        std::vector<float> dq(tp::inter);
        for (std::uint32_t blk = 0; blk < tp::inter / 256U; ++blk)
            ds41::decode_q2_k(std::span<const std::uint8_t, ds41::q2_k_block_bytes>(row.data() + blk * ds41::q2_k_block_bytes, ds41::q2_k_block_bytes),
                              std::span<float, 256>(dq.data() + 256U * blk, 256));
        for (std::uint32_t i = 0; i < tp::inter; ++i) exact += static_cast<double>(dq[i]) * x.codes[i] * x.deltas[i / 32U];
        for (unsigned r = 0; r < tp::ranks; ++r) {
            std::vector<std::uint8_t> packed(tp::down_row_bytes);
            tp::pack_down_row(row, r, packed);
            ds41::A8Activation local;
            for (std::uint32_t g = 0; g < tp::half_groups; ++g) {
                const auto gg = tp::group_index(r, g);
                local.codes.insert(local.codes.end(), x.codes.begin() + 32 * gg, x.codes.begin() + 32 * gg + 32);
                local.deltas.push_back(x.deltas[gg]);
                local.sums.push_back(x.sums[gg]);
            }
            split += tp::product_q2_k_groups(packed, local, tp::half_groups);
        }
        worst = std::max(worst, std::fabs(split - exact) / (std::fabs(exact) + 1.0));
    }
    expect(whole, "K-half reader over the whole row == product_q2_k (bitwise)");
    expect(worst < 1e-4, "rank 0 part + rank 1 part == the exact product (rounding only)");
    std::cout << "worst relative difference of part0 + part1 vs exact: " << worst << '\n';
}

}  // namespace

int main() {
    std::mt19937 rng(1152);
    test_orders();
    test_pack(rng);
    test_product(rng);
    if (failures == 0) std::cout << "ds41 expert tp tests: PASS\n";
    return failures == 0 ? 0 : 1;
}

// Host model of qsa_index::rows_body (src/hip_qsa_prefill_index.inc) for the incumbent 4-row form (FP32 rows staged in
// LDS) and the 8-row form selected by STRIX_QPF2K_QSA_INDEX_SHAPE=4 (raw BF16 rows staged in LDS, widened at the read).
// Invariant: both forms give every (row, output) the same 256 contracted chains (chain t = ascending FMA over the
// features t, t + 256, ...) and the same tree (strides 128, 64, 32 in-lane, then lane l + 16 .. 1), so the outputs are
// bit-identical to each other and to a reference written from the chain definition. The model replays the kernels'
// workgroup / wave / lane mapping and their exact staging and widening bit operations, and checks that both LDS
// footprints are hidden x 16 bytes and that the 8-row form's 16-byte LDS accesses are free of bank conflicts.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using u16 = std::uint16_t;
using u32 = std::uint32_t;

float as_float(u32 bits) { float f; std::memcpy(&f, &bits, 4); return f; }
u32 as_bits(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }

struct Problem {
    unsigned rows = 0, width = 0;
    std::vector<unsigned> first;            // matrix m owns outputs first[m] .. first[m + 1] - 1
    std::vector<std::vector<u16>> weights;  // per matrix, row-major [output][width] BF16 bits
    std::vector<u16> input;                 // [row][width] BF16 bits
    unsigned outputs() const { return first.back(); }
    const u16* weight_row(unsigned g) const {
        unsigned m = 0;
        while (g >= first[m + 1]) ++m;
        return weights[m].data() + static_cast<std::size_t>(g - first[m]) * width;
    }
};

// Reference straight from the contract: 256 chains, then the 256-leaf tree.
std::vector<u32> reference(const Problem& p) {
    std::vector<u32> out(static_cast<std::size_t>(p.rows) * p.outputs());
    for (unsigned r = 0; r < p.rows; ++r)
        for (unsigned g = 0; g < p.outputs(); ++g) {
            const u16* w = p.weight_row(g);
            float leaf[256];
            for (unsigned t = 0; t < 256U; ++t) {
                float s = 0.0F;
                for (unsigned f = t; f < p.width; f += 256U)
                    s = std::fma(as_float(u32(p.input[std::size_t(r) * p.width + f]) << 16U), as_float(u32(w[f]) << 16U), s);
                leaf[t] = s;
            }
            for (unsigned s = 128U; s >= 1U; s /= 2U)
                for (unsigned t = 0; t < s; ++t) leaf[t] = leaf[t] + leaf[t + s];
            out[std::size_t(r) * p.outputs() + g] = as_bits(leaf[0]);
        }
    return out;
}

// One rows_batch_kernel launch with a single 64-row group per 64 rows of the problem (grid.y = rows / RB).
template <unsigned RB, unsigned OW, unsigned W>
std::vector<u32> kernel_model(const Problem& p, std::size_t* lds_bytes) {
    static_assert(RB == 4U || RB == 8U);
    const unsigned outputs = p.outputs(), gx = (outputs + OW * W - 1U) / (OW * W), gy = p.rows / RB;
    *lds_bytes = RB == 4U ? std::size_t(p.width) * RB * sizeof(float) : std::size_t(p.width) * RB * sizeof(u16);
    std::vector<u32> lds_words(*lds_bytes / 4U);
    std::vector<u32> out(std::size_t(p.rows) * outputs, 0xdeadbeefU);
    const auto in_word = [&](unsigned row, unsigned element) {  // the 4-byte word holding elements element, element + 1
        return u32(p.input[std::size_t(row) * p.width + element]) | (u32(p.input[std::size_t(row) * p.width + element + 1U]) << 16U);
    };
    for (unsigned by = 0; by < gy; ++by) {
        const unsigned r0 = by * RB;
        if constexpr (RB == 4U) {  // act[(f + 2j) * RB + r] = word << 16 / word & 0xffff0000 (FP32 rows)
            for (unsigned u = 0; u < RB * p.width / 8U; ++u) {
                const unsigned r = u % RB, f = u / RB * 8U;
                for (unsigned j = 0; j < 4U; ++j) {
                    const u32 word = in_word(r0 + r, f + 2U * j);
                    lds_words[(f + 2U * j) * RB + r] = word << 16U;
                    lds_words[(f + 2U * j + 1U) * RB + r] = word & 0xffff0000U;
                }
            }
        } else {  // act16[f * 8 + row]: word i of feature f = rows 2i (low) and 2i + 1 (high)
            for (unsigned u = 0; u < p.width / 2U; ++u) {
                u32 even[4], odd[4];
                for (unsigned i = 0; i < 4U; ++i) {
                    const u32 a = in_word(r0 + 2U * i, 2U * u), b = in_word(r0 + 2U * i + 1U, 2U * u);
                    even[i] = (a & 0xffffU) | (b << 16U);
                    odd[i] = (a >> 16U) | (b & 0xffff0000U);
                }
                for (unsigned i = 0; i < 4U; ++i) {
                    lds_words[(2U * u * 8U) / 2U + i] = even[i];
                    lds_words[((2U * u + 1U) * 8U) / 2U + i] = odd[i];
                }
            }
        }
        for (unsigned bx = 0; bx < gx; ++bx)
            for (unsigned wave = 0; wave < W; ++wave) {
                unsigned g[OW];
                const u16* weight[OW];
                for (unsigned o = 0; o < OW; ++o) {
                    g[o] = bx * (W * OW) + wave * OW + o;
                    weight[o] = p.weight_row(std::min(g[o], outputs - 1U));
                }
                float sum[32][OW][RB][8] = {};
                for (unsigned lane = 0; lane < 32U; ++lane)
                    for (unsigned k = 0; k < p.width / 256U; ++k)
                        for (unsigned j = 0; j < 8U; ++j) {
                            const unsigned f = k * 256U + 32U * j + lane;
                            float w[OW], a[RB];
                            for (unsigned o = 0; o < OW; ++o) w[o] = as_float(u32(weight[o][f]) << 16U);
                            if constexpr (RB == 4U) {
                                for (unsigned r = 0; r < RB; ++r) a[r] = as_float(lds_words[f * RB + r]);
                            } else {
                                for (unsigned i = 0; i < 4U; ++i) {
                                    const u32 word = lds_words[f * 8U / 2U + i];
                                    a[2U * i] = as_float(word << 16U);
                                    a[2U * i + 1U] = as_float(word & 0xffff0000U);
                                }
                            }
                            for (unsigned o = 0; o < OW; ++o)
                                for (unsigned r = 0; r < RB; ++r) sum[lane][o][r][j] = std::fma(a[r], w[o], sum[lane][o][r][j]);
                        }
                for (unsigned o = 0; o < OW; ++o)
                    for (unsigned r = 0; r < RB; ++r) {
                        float x[32];
                        for (unsigned lane = 0; lane < 32U; ++lane) {
                            float v[8];
                            std::memcpy(v, sum[lane][o][r], sizeof v);
                            for (unsigned j = 0; j < 4U; ++j) v[j] = v[j] + v[j + 4U];
                            v[0] = v[0] + v[2]; v[1] = v[1] + v[3];
                            x[lane] = v[0] + v[1];
                        }
                        for (unsigned d = 16U; d >= 1U; d /= 2U)  // lane_down<d>: lane l adds lane l + d
                            for (unsigned lane = 0; lane < d; ++lane) x[lane] = x[lane] + x[lane + d];
                        if (g[o] < outputs) out[std::size_t(r0 + r) * outputs + g[o]] = as_bits(x[0]);
                    }
            }
    }
    return out;
}

// 16-byte LDS accesses are served 8 lanes (128 B) per pass over 64 four-byte banks: a pass is conflict-free when the
// 8 lanes' 4-bank groups are disjoint.
bool b128_conflict_free(const std::vector<unsigned>& byte_offsets) {
    for (std::size_t pass = 0; pass + 8U <= byte_offsets.size(); pass += 8U) {
        bool used[64] = {};
        for (std::size_t l = pass; l < pass + 8U; ++l)
            for (unsigned b = 0; b < 4U; ++b) {
                const unsigned bank = (byte_offsets[l] / 4U + b) % 64U;
                if (used[bank]) return false;
                used[bank] = true;
            }
    }
    return true;
}

// special: 0 = none, 1 = with max finite, infinities and NaN payloads, 2 = finite only (signed zeros, subnormals,
// smallest normal, +-1, +-2^32: no product or sum overflows).
u16 bf16_bits(std::mt19937& rng, unsigned special) {
    if (special) {
        static const u16 values[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3f80, 0xbf80, 0x4f80, 0xcf80, 0x7f7f, 0x7f80, 0xff80, 0x7fc1, 0xffa3};
        const std::size_t count = special == 1U ? sizeof values / sizeof values[0] : 9U;
        if (rng() % (special == 1U ? 3U : 50U) == 0U) return values[rng() % count];
    }
    const float v = std::normal_distribution<float>(0.0F, 1.0F)(rng);
    return static_cast<u16>(as_bits(v) >> 16U);
}

Problem make(unsigned rows, unsigned width, std::vector<unsigned> first, std::uint32_t seed, unsigned special) {
    std::mt19937 rng(seed);
    Problem p;
    p.rows = rows; p.width = width; p.first = std::move(first);
    for (std::size_t m = 0; m + 1U < p.first.size(); ++m) {
        p.weights.emplace_back(std::size_t(p.first[m + 1U] - p.first[m]) * width);
        for (auto& w : p.weights.back()) w = bf16_bits(rng, special);
    }
    p.input.resize(std::size_t(rows) * width);
    for (auto& x : p.input) x = bf16_bits(rng, special);
    return p;
}

// nan_class: the host compiler may commute the operands of an add, which picks the other NaN payload; against the
// separately compiled contract reference a NaN therefore only has to be a NaN. The same holds between the two host
// models: they are separate instantiations and the compiler may commute an add in one and not the other (GCC 15 -O2
// does), so RB8 vs RB4 is bit for bit on every non-NaN output and NaN class on NaN outputs. The device forms are
// compared bit for bit by the HIP batch test.
unsigned compare(const char* name, const std::vector<u32>& got, const std::vector<u32>& want, bool nan_class) {
    require(got.size() == want.size(), std::string(name) + ": size");
    const auto nan = [](u32 b) { return (b & 0x7f800000U) == 0x7f800000U && (b & 0x007fffffU) != 0U; };
    unsigned diff = 0;
    for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != want[i] && !(nan_class && nan(got[i]) && nan(want[i]));
    if (diff) std::printf("FAIL %s: %u of %zu outputs differ\n", name, diff, got.size());
    return diff;
}

unsigned run_case(const char* label, const Problem& p) {
    std::size_t lds4 = 0, lds8 = 0;
    const auto ref = reference(p);
    const auto old = kernel_model<4U, 2U, 16U>(p, &lds4);
    const auto rb8 = kernel_model<8U, 2U, 16U>(p, &lds8);
    unsigned bad = compare((std::string(label) + " RB4 vs contract").c_str(), old, ref, true) +
        compare((std::string(label) + " RB8 vs contract").c_str(), rb8, ref, true) +
        compare((std::string(label) + " RB8 vs RB4").c_str(), rb8, old, true);
    unsigned nans = 0;
    for (const u32 b : rb8) nans += (b & 0x7f800000U) == 0x7f800000U && (b & 0x007fffffU) != 0U;
    if (nans) std::printf("note %s: %u NaN outputs (NaN class only: host models may commute adds)\n", label, nans);
    if (lds4 != lds8 || lds8 != std::size_t(p.width) * 16U) { std::printf("FAIL %s: LDS %zu / %zu\n", label, lds4, lds8); ++bad; }
    std::printf("%s %s rows=%u width=%u outputs=%u lds=%zu B\n", bad ? "FAIL" : "PASS", label, p.rows, p.width, p.outputs(), lds8);
    return bad;
}
} // namespace

int main() try {
    unsigned bad = 0;
    // Production geometry of one 64-row group (q 512 + k 128 outputs, hidden 2560), plus a partial last wave
    // (outputs not a multiple of W x OW) and special values (signed zeros, infinities, NaN payloads, subnormals).
    bad += run_case("qwen-q512-k128", make(64U, 2560U, {0U, 512U, 640U}, 1U, 0U));
    bad += run_case("two-groups-partial-wave", make(128U, 512U, {0U, 40U, 72U}, 2U, 0U));
    bad += run_case("finite-specials", make(64U, 2560U, {0U, 64U}, 4U, 2U));
    bad += run_case("inf-nan-specials", make(64U, 512U, {0U, 64U}, 3U, 1U));
    // Bank checks of the 8-row form for one wave: staging stores (unit u = tid: features 2u and 2u + 1, 16 B each)
    // and the main-loop read (feature f = base + lane, 16 B at f x 16).
    std::vector<unsigned> store_even, store_odd, read;
    for (unsigned lane = 0; lane < 32U; ++lane) {
        store_even.push_back(2U * lane * 16U);
        store_odd.push_back((2U * lane + 1U) * 16U);
        read.push_back((256U + 32U * 3U + lane) * 16U);
    }
    const bool banks = b128_conflict_free(store_even) && b128_conflict_free(store_odd) && b128_conflict_free(read);
    std::printf("%s RB8 16-byte LDS stores and reads conflict-free\n", banks ? "PASS" : "FAIL");
    bad += banks ? 0U : 1U;
    std::printf(bad ? "FAILED\n" : "ALL PASS\n");
    return bad ? 1 : 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }

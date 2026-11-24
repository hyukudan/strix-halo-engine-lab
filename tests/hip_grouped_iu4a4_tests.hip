// Device fixture of the grouped rotated A4W4 expert class (moe_core/grouped_iu4a4.hpp on the grouped trait of
// core/iu4a4_dense.hpp), bit for bit:
//   - the expert requantisation (NVFP4 and MXFP8 sources, plain and paired column order, contiguous and interleaved
//     row order) against ExpertSource on the host;
//   - every plain output of a routed gate/up and down launch against iu4a4_rot::dot of its gathered row (token image
//     read through the assignment -> token map; down rows in assignment order located through the gate/up tiles);
//   - the paired SwiGLU epilogue against the plain outputs of the same image put through the same device activation;
//   - rows outside the tiles stay untouched; a device tile count refuses the tiles past it;
//   - the validation launch: every bounds defect, a short / zero / oversized device count, overlapping tiles and a
//     planner failure set the verdict word; a clean plan leaves it 0;
//   - the whole pass (gi::launch) at Inkling 2 Mini shapes: the same inputs give the same token image, h image and
//     down rows bit for bit whatever the workspace held before (two poison patterns, three passes); a planner
//     failure or a short count stores nothing (h image and down rows keep the poison); a down tile output one past
//     the destination changes nothing (destinations derive from the admitted interval); the exact workspace is
//     admitted and one byte short is refused with nothing enqueued.
// --bench [tokens]: Inkling 2 Mini shapes (128 experts, top-8, hidden 2304, ffn 1152), uniform routing, ms and TOPS of
// the token quantise, paired gate/up, h quantise and down launches (plan tiles of <= 256 rows as the HIP plan) (no reference; a rate, not a model receipt).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_iu4a4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
namespace gi = strix::moe_core::grouped_iu4a4;
namespace ad = strix::core::iu4a4_dense;
namespace rot = strix::core::iu4a4_rot;
namespace packed = strix::weight::microscaled;
using strix::moe_core::grouped_microscaled::TileRows;
#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

std::uint32_t bits(float v) {
    std::uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }

template <class T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    CHECK(hipMalloc(&d, std::max<std::size_t>(v.size(), 1U) * sizeof(T)));
    if (!v.empty()) CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <class T>
std::vector<T> download(const T* d, std::size_t count) {
    std::vector<T> v(count);
    if (count) CHECK(hipMemcpy(v.data(), d, count * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

// A layer's experts of one projection: rows x in each, stored contiguously (expert e at e * stride).
struct Bank {
    packed::MatrixFormat format;
    unsigned experts, rows, in;
    packed::RowOrder order;
    std::vector<std::uint8_t> codes, scales;
    std::uint8_t *dcodes = nullptr, *dscales = nullptr;
    std::size_t code_row() const { return format == packed::MatrixFormat::nvfp4 ? in / 2U : in; }
    std::size_t scale_row() const { return in / (format == packed::MatrixFormat::nvfp4 ? 16U : 32U); }
    packed::PackedMatrix matrix(unsigned e, bool device) const {
        const std::uint8_t* c = (device ? dcodes : codes.data()) + std::size_t(e) * rows * code_row();
        const std::uint8_t* s = (device ? dscales : scales.data()) + std::size_t(e) * rows * scale_row();
        return {c, s, rows, in, format == packed::MatrixFormat::nvfp4 ? 0.75F : 1.0F, order};
    }
};
Bank make_bank(packed::MatrixFormat f, unsigned experts, unsigned rows, unsigned in, packed::RowOrder order, std::mt19937& g) {
    Bank b{f, experts, rows, in, order, {}, {}};
    b.codes.resize(std::size_t(experts) * rows * b.code_row());
    b.scales.resize(std::size_t(experts) * rows * b.scale_row());
    for (auto& c : b.codes) {
        c = static_cast<std::uint8_t>(g());
        if (f == packed::MatrixFormat::mxfp8 && (c & 0x7fU) == 0x7fU) c = 0x3cU;  // no E4M3 NaN codes
    }
    // NVFP4: E4M3 block scales 2^-3 .. 2^1; MXFP8: E8M0 2^-8 .. 2^-4.
    for (auto& s : b.scales) s = static_cast<std::uint8_t>(f == packed::MatrixFormat::nvfp4 ? 0x20U + g() % 32U : 119U + g() % 5U);
    b.dcodes = upload(b.codes);
    b.dscales = upload(b.scales);
    return b;
}

struct Image {
    std::uint8_t* codes = nullptr;
    float* steps = nullptr;
    gi::ExpertImage view{};
};
Image requantise(const Bank& b, unsigned pair_rows) {
    const unsigned n = b.rows;
    Image im;
    im.view.experts = b.experts;
    im.view.code_bytes = gi::image_code_bytes(b.in, n);
    im.view.step_floats = gi::image_step_floats(b.in, n);
    CHECK(hipMalloc(&im.codes, im.view.code_bytes * b.experts));
    CHECK(hipMalloc(&im.steps, im.view.step_floats * b.experts * 4U));
    for (unsigned e = 0; e < b.experts; ++e) {
        if (b.format == packed::MatrixFormat::nvfp4)
            gi::enqueue_requantise_expert<packed::MatrixFormat::nvfp4>(b.matrix(e, true), pair_rows, b.in, n, im.codes + e * im.view.code_bytes,
                                                                       im.steps + e * im.view.step_floats, nullptr);
        else
            gi::enqueue_requantise_expert<packed::MatrixFormat::mxfp8>(b.matrix(e, true), pair_rows, b.in, n, im.codes + e * im.view.code_bytes,
                                                                       im.steps + e * im.view.step_floats, nullptr);
    }
    CHECK(hipGetLastError());
    im.view.codes = im.codes;
    im.view.steps = im.steps;
    return im;
}
// Host image of expert e (codes in the rotated tile layout, steps [n][blocks] as iu4a4_rot::dot reads them).
template <packed::MatrixFormat F>
void host_image(const Bank& b, unsigned e, unsigned pair_rows, std::vector<std::uint8_t>& codes, std::vector<float>& steps) {
    const unsigned n = b.rows, blocks = b.in / rot::kBlock;
    codes.assign(gi::image_code_bytes(b.in, n), 0);
    steps.assign(std::size_t(n) * blocks, 0.0F);
    const gi::ExpertSource<F> src{b.matrix(e, false), pair_rows};
    for (unsigned blk = 0; blk < blocks; ++blk)
        for (unsigned c = 0; c < n; ++c) steps[std::size_t(c) * blocks + blk] = src.requantise(c, blk, n, codes.data());
}
void host_image(const Bank& b, unsigned e, unsigned pair_rows, std::vector<std::uint8_t>& codes, std::vector<float>& steps) {
    if (b.format == packed::MatrixFormat::nvfp4) host_image<packed::MatrixFormat::nvfp4>(b, e, pair_rows, codes, steps);
    else host_image<packed::MatrixFormat::mxfp8>(b, e, pair_rows, codes, steps);
}

// Routing: token t picks k distinct experts; assignments sorted by expert; tiles of <= 128 rows per expert.
struct Plan {
    std::vector<std::uint32_t> token;  // assignment -> token
    std::vector<unsigned> expert;      // assignment -> expert
    std::vector<unsigned> tile_first, tile_rows, tile_expert;
};
Plan route(unsigned tokens, unsigned experts, unsigned k, std::mt19937& g, bool skew, unsigned cap) {
    std::vector<std::pair<unsigned, std::uint32_t>> as;
    std::vector<unsigned> ids(experts);
    std::iota(ids.begin(), ids.end(), 0U);
    for (unsigned t = 0; t < tokens; ++t) {
        std::shuffle(ids.begin(), ids.end(), g);
        if (skew) std::sort(ids.begin(), ids.begin() + std::min<std::ptrdiff_t>(experts, 2 * std::ptrdiff_t(k)));  // favour low ids
        for (unsigned j = 0; j < k; ++j) as.push_back({ids[j], t});
    }
    std::stable_sort(as.begin(), as.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    Plan p;
    for (const auto& a : as) p.token.push_back(a.second), p.expert.push_back(a.first);
    for (std::size_t a = 0; a < as.size();) {
        std::size_t end = a;
        while (end < as.size() && as[end].first == as[a].first && end - a < cap) ++end;
        p.tile_first.push_back(static_cast<unsigned>(a));
        p.tile_rows.push_back(static_cast<unsigned>(end - a));
        p.tile_expert.push_back(as[a].first);
        a = end;
    }
    return p;
}

struct Act {
    ad::Act act{};
    void alloc(unsigned rows, unsigned in) {
        act.rows = rows;
        act.in = in;
        CHECK(hipMalloc(&act.codes, ad::act_code_bytes(rows, in)));
        CHECK(hipMalloc(&act.u, ad::act_u_bytes(rows, in)));
        CHECK(hipMalloc(&act.base, ad::act_base_bytes(rows)));
    }
};

__global__ void swiglu_check_kernel(const float* plain, unsigned rows, unsigned n, float* h, gi::SwigluOut epi) {
    const std::size_t id = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned f = n / 2U;
    if (id >= std::size_t(rows) * f) return;
    const unsigned r = static_cast<unsigned>(id / f), c = static_cast<unsigned>(id % f);
    const unsigned gcol = 32U * (c / 16U) + c % 16U;  // paired order: tile 2 j = gate, 2 j + 1 = up of outputs 16 j ..
    h[id] = epi(plain[std::size_t(r) * n + gcol], plain[std::size_t(r) * n + gcol + 16U]);
}

struct Launch {
    std::vector<TileRows> gate_tiles, down_tiles;
    TileRows *dgate = nullptr, *ddown = nullptr;
    std::uint16_t* index_base = nullptr;
    std::uint32_t* dmap = nullptr;
    unsigned* dcount = nullptr;
    unsigned* dfault = nullptr;  // the pass's fault word (kFaultBounds), zeroed
    unsigned rows = 0, max_rows = 0;
};
Launch make_launch(const Plan& p, const Bank& gate, const Bank& down, float* y, unsigned n_down, unsigned count) {
    Launch l;
    l.index_base = upload(std::vector<std::uint16_t>(p.token.size() + 1U, 0));
    for (std::size_t t = 0; t < p.tile_first.size(); ++t) {
        TileRows g{gate.matrix(p.tile_expert[t], true), l.index_base + p.tile_first[t], nullptr, 1U, 0U, p.tile_rows[t]};
        TileRows d{down.matrix(p.tile_expert[t], true), nullptr, y + std::size_t(p.tile_first[t]) * n_down, 0U, n_down, p.tile_rows[t]};
        l.gate_tiles.push_back(g);
        l.down_tiles.push_back(d);
    }
    l.dgate = upload(l.gate_tiles);
    l.ddown = upload(l.down_tiles);
    l.dmap = upload(p.token);
    l.dcount = upload(std::vector<unsigned>{count});
    l.dfault = upload(std::vector<unsigned>{0U});
    l.rows = static_cast<unsigned>(p.token.size());
    l.max_rows = *std::max_element(p.tile_rows.begin(), p.tile_rows.end());
    return l;
}
gi::TileSource source(const Launch& l, const Bank& bank, const Image& im, bool gate_side, unsigned n, float* h, std::size_t h_row) {
    gi::TileSource s{};
    s.tiles = gate_side ? l.dgate : l.ddown;
    s.index_tiles = l.dgate;
    s.index_base = l.index_base;
    s.map = gate_side ? l.dmap : nullptr;
    s.tile_count = l.dcount;
    s.codes_from = bank.dcodes;
    s.codes_stride = std::size_t(bank.rows) * bank.code_row();
    s.image = im.view;
    s.in = bank.in;
    s.n = n;
    s.out = h;
    s.out_row = h_row;
    s.index_rows = l.rows;
    s.max_rows = l.max_rows;
    return s;
}

int fixture(unsigned tokens, unsigned experts, unsigned k, unsigned hidden, unsigned ffn, packed::MatrixFormat f, packed::RowOrder order,
            bool skew, unsigned seed, bool cut, unsigned cap = 128U) {
    std::mt19937 g(seed);
    const Bank gate = make_bank(f, experts, 2U * ffn, hidden, order, g), down = make_bank(f, experts, hidden, ffn, packed::RowOrder::contiguous, g);
    const Plan p = route(tokens, experts, k, g, skew, cap);
    const unsigned A = static_cast<unsigned>(p.token.size()), tiles = static_cast<unsigned>(p.tile_first.size());
    const unsigned count = cut ? tiles - 1U : tiles, live_rows = cut ? A - p.tile_rows.back() : A;
    const unsigned tall = *std::max_element(p.tile_rows.begin(), p.tile_rows.end());
    const unsigned ng = 2U * ffn;
    // Token rows (normal, outlier channels, one zero row) -> token image.
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t(tokens) * hidden);
    for (auto& v : x) v = nd(g);
    for (unsigned t = 0; t < tokens; ++t)
        for (unsigned c = 5; c < hidden; c += 97U) x[std::size_t(t) * hidden + c] *= 30.0F;
    if (tokens > 2U) std::fill(x.begin() + 2 * std::ptrdiff_t(hidden), x.begin() + 3 * std::ptrdiff_t(hidden), 0.0F);
    float* dx = upload(x);
    Act tok, hact;
    tok.alloc(tokens, hidden);
    hact.alloc(A, ffn);
    ad::enqueue_quantise_rows(dx, hidden, tok.act, nullptr);
    const Image gim = requantise(gate, ffn), dim = requantise(down, 0U);
    // Outputs prefilled with a NaN pattern; one guard row past the end.
    float *dplain = nullptr, *dh = nullptr, *dhc = nullptr, *dy = nullptr;
    CHECK(hipMalloc(&dplain, std::size_t(A + 1U) * ng * 4U));
    CHECK(hipMalloc(&dh, std::size_t(A + 1U) * ffn * 4U));
    CHECK(hipMalloc(&dhc, std::size_t(A) * ffn * 4U));
    CHECK(hipMalloc(&dy, std::size_t(A + 1U) * hidden * 4U));
    CHECK(hipMemset(dplain, 0xff, std::size_t(A + 1U) * ng * 4U));
    CHECK(hipMemset(dh, 0xff, std::size_t(A + 1U) * ffn * 4U));
    CHECK(hipMemset(dy, 0xff, std::size_t(A + 1U) * hidden * 4U));
    const Launch l = make_launch(p, gate, down, dy, hidden, count);
    const gi::SwigluOut epi{4.0F, 25.0F};
    // Plain gate/up (rows in assignment order through h = dplain), paired gate/up -> h, its device check, then down.
    ad::enqueue_grouped_gemm(source(l, gate, gim, true, ng, dplain, ng), tiles, tall, ng, tok.act, nullptr);
    ad::enqueue_grouped_gemm(source(l, gate, gim, true, ng, dh, ffn), tiles, tall, ng, tok.act, nullptr, epi);
    hipLaunchKernelGGL(swiglu_check_kernel, dim3(static_cast<unsigned>((std::size_t(A) * ffn + 255U) / 256U)), dim3(256), 0, nullptr, dplain, A, ng, dhc, epi);
    // h rows past the cut are NaN-pattern: quantise them anyway (their down tiles are refused).
    ad::enqueue_quantise_rows(dh, ffn, hact.act, nullptr);
    ad::enqueue_grouped_gemm(source(l, down, dim, false, hidden, dy, hidden), tiles, tall, hidden, hact.act, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    unsigned bad_img = 0, bad_plain = 0, bad_pair = 0, bad_down = 0, bad_guard = 0;
    // Images.
    std::vector<std::uint8_t> hc;
    std::vector<float> hs;
    std::vector<std::vector<std::uint8_t>> gcodes(experts), dcodes(experts);
    std::vector<std::vector<float>> gsteps(experts), dsteps(experts);
    const auto dg = download(gim.codes, gim.view.code_bytes * experts), dd = download(dim.codes, dim.view.code_bytes * experts);
    const auto sg = download(gim.steps, gim.view.step_floats * experts), sd = download(dim.steps, dim.view.step_floats * experts);
    for (unsigned e = 0; e < experts; ++e) {
        for (int side = 0; side < 2; ++side) {
            const Bank& b = side == 0 ? gate : down;
            host_image(b, e, side == 0 ? ffn : 0U, hc, hs);
            const auto& dc = side == 0 ? dg : dd;
            const auto& ds = side == 0 ? sg : sd;
            const std::size_t cb = side == 0 ? gim.view.code_bytes : dim.view.code_bytes, sf = side == 0 ? gim.view.step_floats : dim.view.step_floats;
            const unsigned n = b.rows, blocks = b.in / rot::kBlock;
            for (std::size_t i = 0; i < cb; ++i) bad_img += hc[i] != dc[e * cb + i];
            for (unsigned blk = 0; blk < blocks; ++blk)
                for (unsigned c = 0; c < n; ++c) bad_img += bits(hs[std::size_t(c) * blocks + blk]) != bits(ds[e * sf + ad::step_index(blk, c, n)]);
            (side == 0 ? gcodes : dcodes)[e] = hc;
            (side == 0 ? gsteps : dsteps)[e] = hs;
        }
    }
    // Activation images as the device made them (the quantiser is the dense fixture's subject).
    const unsigned tmp = ad::padded_rows(tokens), hmp = ad::padded_rows(A);
    const auto tc = download(tok.act.codes, ad::act_code_bytes(tokens, hidden));
    const auto tu = download(tok.act.u, ad::act_u_bytes(tokens, hidden) / 4U);
    const auto tb = download(tok.act.base, tmp);
    const auto hcod = download(hact.act.codes, ad::act_code_bytes(A, ffn));
    const auto hu = download(hact.act.u, ad::act_u_bytes(A, ffn) / 4U);
    const auto hb = download(hact.act.base, hmp);
    const auto row_of = [](const std::vector<std::uint8_t>& codes, const std::vector<std::uint32_t>& u, unsigned r, unsigned in, unsigned mp,
                           std::vector<std::int8_t>& rc, std::vector<std::uint8_t>& ru) {
        rc.resize(in);
        ru.resize(in / rot::kGroupA);
        // Row-major image (enqueue_quantise_rows): codes [r][in / 2] low nibble first, u [r][in / 128].
        static_cast<void>(mp);
        for (unsigned c = 0; c < in; ++c) {
            const unsigned nib = (codes[std::size_t(r) * (in / 2U) + c / 2U] >> ((c & 1U) * 4U)) & 15U;
            rc[c] = static_cast<std::int8_t>(nib < 8U ? int(nib) : int(nib) - 16);
        }
        const unsigned blocks = in / rot::kBlock;
        for (unsigned blk = 0; blk < blocks; ++blk)
            for (unsigned q = 0; q < rot::kGroupsPerBlock; ++q) ru[blk * 4U + q] = static_cast<std::uint8_t>((u[std::size_t(r) * blocks + blk] >> (8U * q)) & 255U);
    };
    const auto plain = download(dplain, std::size_t(A + 1U) * ng), h = download(dh, std::size_t(A + 1U) * ffn), hchk = download(dhc, std::size_t(A) * ffn);
    const auto y = download(dy, std::size_t(A + 1U) * hidden);
    std::vector<std::int8_t> rc;
    std::vector<std::uint8_t> ru;
    for (unsigned a = 0; a < A; ++a) {
        const unsigned e = p.expert[a], t = p.token[a];
        const bool live = a < live_rows;
        row_of(tc, tu, t, hidden, tmp, rc, ru);
        for (unsigned c = 0; c < ng; ++c) {
            const float dev = plain[std::size_t(a) * ng + c];
            const float ref = live ? rot::dot(rc.data(), ru.data(), tb[t], gcodes[e].data(), gsteps[e].data(), hidden, ng, c) : dev;
            if (live ? !same(dev, ref) : bits(dev) != 0xffffffffU) {
                if (bad_plain < 3U) std::printf("  plain a %u c %u dev 0x%08x ref 0x%08x\n", a, c, bits(dev), bits(ref));
                ++bad_plain;
            }
        }
        for (unsigned c = 0; c < ffn; ++c) {
            const float dev = h[std::size_t(a) * ffn + c];
            bad_pair += live ? !same(dev, hchk[std::size_t(a) * ffn + c]) : bits(dev) != 0xffffffffU;
        }
        row_of(hcod, hu, a, ffn, hmp, rc, ru);
        for (unsigned c = 0; c < hidden; ++c) {
            const float dev = y[std::size_t(a) * hidden + c];
            const float ref = live ? rot::dot(rc.data(), ru.data(), hb[a], dcodes[e].data(), dsteps[e].data(), ffn, hidden, c) : dev;
            if (live ? !same(dev, ref) : bits(dev) != 0xffffffffU) {
                if (bad_down < 3U) std::printf("  down a %u c %u dev 0x%08x ref 0x%08x\n", a, c, bits(dev), bits(ref));
                ++bad_down;
            }
        }
    }
    for (unsigned c = 0; c < ng; ++c) bad_guard += bits(plain[std::size_t(A) * ng + c]) != 0xffffffffU;
    for (unsigned c = 0; c < ffn; ++c) bad_guard += bits(h[std::size_t(A) * ffn + c]) != 0xffffffffU;
    for (unsigned c = 0; c < hidden; ++c) bad_guard += bits(y[std::size_t(A) * hidden + c]) != 0xffffffffU;
    bad_guard += download(l.dfault, 1U)[0] != 0U;  // a clean pass faults nothing
    const bool ok = !(bad_img | bad_plain | bad_pair | bad_down | bad_guard);
    std::printf("%s %s %s%s%s tokens %u experts %u k %u hidden %u ffn %u tiles %u (run %u): image %u plain %u paired %u down %u guard %u differ\n",
                ok ? "PASS" : "FAIL", f == packed::MatrixFormat::nvfp4 ? "nvfp4" : "mxfp8", order == packed::RowOrder::contiguous ? "contig" : "interleaved",
                skew ? " skew" : "", cut ? " cut" : "", tokens, experts, k, hidden, ffn, tiles, count, bad_img, bad_plain, bad_pair, bad_down, bad_guard);
    return ok ? 0 : 1;
}

int bench(unsigned tokens) {
    const unsigned experts = 128U, k = 8U, hidden = 2304U, ffn = 1152U, ng = 2U * ffn;
    std::mt19937 g(7U);
    const Bank gate = make_bank(packed::MatrixFormat::nvfp4, experts, ng, hidden, packed::RowOrder::contiguous, g);
    const Bank down = make_bank(packed::MatrixFormat::nvfp4, experts, hidden, ffn, packed::RowOrder::contiguous, g);
    const Plan p = route(tokens, experts, k, g, false, 256U);
    const unsigned A = static_cast<unsigned>(p.token.size()), tiles = static_cast<unsigned>(p.tile_first.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t(tokens) * hidden);
    for (auto& v : x) v = nd(g);
    float* dx = upload(x);
    Act tok, hact;
    tok.alloc(tokens, hidden);
    hact.alloc(A, ffn);
    const Image gim = requantise(gate, ffn), dim = requantise(down, 0U);
    float *dh = nullptr, *dy = nullptr;
    CHECK(hipMalloc(&dh, std::size_t(A) * ffn * 4U));
    CHECK(hipMalloc(&dy, std::size_t(A) * hidden * 4U));
    const Launch l = make_launch(p, gate, down, dy, hidden, tiles);
    const gi::SwigluOut epi{4.0F, 25.0F};
    const auto gs = source(l, gate, gim, true, ng, dh, ffn), ds = source(l, down, dim, false, hidden, dy, hidden);
    hipEvent_t ev[5];
    for (auto& e : ev) CHECK(hipEventCreate(&e));
    const unsigned reps = 10U;
    float ms[4] = {0, 0, 0, 0};
    for (unsigned it = 0; it < reps + 2U; ++it) {
        CHECK(hipEventRecord(ev[0], nullptr));
        ad::enqueue_quantise_rows(dx, hidden, tok.act, nullptr);
        CHECK(hipEventRecord(ev[1], nullptr));
        ad::enqueue_grouped_gemm(gs, tiles, 256U, ng, tok.act, nullptr, epi);
        CHECK(hipEventRecord(ev[2], nullptr));
        ad::enqueue_quantise_rows(dh, ffn, hact.act, nullptr);
        CHECK(hipEventRecord(ev[3], nullptr));
        ad::enqueue_grouped_gemm(ds, tiles, 256U, hidden, hact.act, nullptr);
        CHECK(hipEventRecord(ev[4], nullptr));
        CHECK(hipEventSynchronize(ev[4]));
        if (it < 2U) continue;
        for (unsigned s = 0; s < 4U; ++s) {
            float t = 0.0F;
            CHECK(hipEventElapsedTime(&t, ev[s], ev[s + 1U]));
            ms[s] += t / float(reps);
        }
    }
    // Reference: the dense launch of the same contract at the same shapes, one matrix (expert 0) for all A rows
    // (no gather, no tiles, column-layout image): what the arithmetic costs without grouping.
    for (unsigned c = 0; c < k && std::size_t(c + 1U) * tokens <= A; ++c) CHECK(hipMemcpy(dy + std::size_t(c) * tokens * hidden, dx, std::size_t(tokens) * hidden * 4U, hipMemcpyDeviceToDevice));
    Act dg, dd;
    dg.alloc(A, hidden);
    dd.alloc(A, ffn);
    float* dz = nullptr;
    CHECK(hipMalloc(&dz, std::size_t(A) * ng * 4U));
    const ad::Weights wg{gim.codes, gim.steps, hidden, ng}, wd{dim.codes, dim.steps, ffn, hidden};
    float dms[2] = {0, 0};
    for (unsigned it = 0; it < reps + 2U; ++it) {
        CHECK(hipEventRecord(ev[0], nullptr));
        ad::enqueue_gemm(wg, dg.act, ad::OutCols{dz, ng, ng / 16U}, nullptr);
        CHECK(hipEventRecord(ev[1], nullptr));
        ad::enqueue_gemm(wd, dd.act, ad::OutCols{dy, hidden, hidden / 16U}, nullptr);
        CHECK(hipEventRecord(ev[2], nullptr));
        CHECK(hipEventSynchronize(ev[2]));
        if (it == 0U) {  // image of real rows once (the rate does not depend on it after)
            ad::enqueue_quantise(dy, hidden, dg.act, nullptr);
            ad::enqueue_quantise(dz, ffn, dd.act, nullptr);
            CHECK(hipDeviceSynchronize());
        }
        if (it < 2U) continue;
        for (unsigned s = 0; s < 2U; ++s) {
            float t = 0.0F;
            CHECK(hipEventElapsedTime(&t, ev[s], ev[s + 1U]));
            dms[s] += t / float(reps);
        }
    }
    CHECK(hipGetLastError());
    const double fg = 2.0 * A * hidden * ng, fd = 2.0 * A * ffn * hidden;
    std::printf("DENSE-REF rows %u: gate_up %.3f ms %.1f TOPS | down %.3f ms %.1f TOPS\n", A, double(dms[0]), fg / (double(dms[0]) * 1e9),
                double(dms[1]), fd / (double(dms[1]) * 1e9));
    std::printf("BENCH tokens %u assignments %u tiles %u: quant_tok %.3f ms | gate_up %.3f ms %.1f TOPS | quant_h %.3f ms | down %.3f ms %.1f TOPS | visit %.3f ms\n",
                tokens, A, tiles, double(ms[0]), double(ms[1]), fg / (double(ms[1]) * 1e9), double(ms[2]), double(ms[3]), fd / (double(ms[3]) * 1e9),
                double(ms[0] + ms[1] + ms[2] + ms[3]));
    return 0;
}
}  // namespace

// Bounds faults of the grouped launch. Each case corrupts one operand of a clean gate/up launch (plain rows to h in
// assignment order): the bad tile binds no rows (its h rows keep the NaN pattern), the fault word gets kFaultBounds, the
// other tiles still store and the guard row past the end stays untouched. A map entry past the image is the exception: its
// tile reads the image's last row (stores) and faults. The clean launch faults nothing.
int faults() {
    std::mt19937 g(11U);
    const unsigned tokens = 97U, experts = 5U, k = 3U, hidden = 256U, ffn = 128U, ng = 2U * ffn;
    const Bank gate = make_bank(packed::MatrixFormat::nvfp4, experts, ng, hidden, packed::RowOrder::contiguous, g);
    const Plan p = route(tokens, experts, k, g, false, 128U);
    const unsigned A = static_cast<unsigned>(p.token.size()), tiles = static_cast<unsigned>(p.tile_first.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t(tokens) * hidden);
    for (auto& v : x) v = nd(g);
    Act tok;
    tok.alloc(tokens, hidden);
    ad::enqueue_quantise_rows(upload(x), hidden, tok.act, nullptr);
    const Image gim = requantise(gate, ffn);
    const Launch l = make_launch(p, gate, gate, nullptr, 0U, tiles);
    float* dh = nullptr;
    CHECK(hipMalloc(&dh, std::size_t(A + 1U) * ng * 4U));
    unsigned bad_tile = 1U;  // the corrupted tile; the last expert's tile for the expert case
    for (unsigned t = 0; t < tiles; ++t)
        if (p.tile_expert[t] == experts - 1U) bad_tile = t;
    enum Case { clean, map_past, off_stride, expert_past, rows_differ, rows_over_max, index_past, cases };
    const char* names[cases] = {"clean", "map entry past the image", "weights off the slot stride", "expert past the image",
                                "tile rows differ from the index tile", "tile rows above max_rows", "index rows past index_rows"};
    int fails = 0;
    for (int c = 0; c < cases; ++c) {
        CHECK(hipMemset(dh, 0xff, std::size_t(A + 1U) * ng * 4U));
        CHECK(hipMemset(l.dfault, 0, 4U));
        gi::TileSource s = source(l, gate, gim, true, ng, dh, ng);
        std::vector<TileRows> tl = l.gate_tiles;
        std::vector<std::uint32_t> map = p.token;
        unsigned victim = c == expert_past ? bad_tile : 1U;
        if (c == map_past) map[p.tile_first[1]] = tokens;
        if (c == off_stride) tl[1].weights.codes += 16;
        if (c == expert_past) s.image.experts = experts - 1U;
        if (c == rows_differ) tl[1].rows -= 1U;
        if (c == rows_over_max) {
            victim = static_cast<unsigned>(std::max_element(p.tile_rows.begin(), p.tile_rows.end()) - p.tile_rows.begin());
            s.max_rows = p.tile_rows[victim] - 1U;
        }
        if (c == index_past) {
            victim = tiles - 1U;
            s.index_rows = A - 1U;
        }
        TileRows* dt = upload(tl);
        std::uint32_t* dm = upload(map);
        s.tiles = dt;
        s.map = dm;
        ad::enqueue_grouped_gemm(s, tiles, l.max_rows, ng, tok.act, nullptr);
        hipLaunchKernelGGL(gi::detail::validate_kernel, dim3(1), dim3(1024), 0, nullptr, s, s, static_cast<const unsigned*>(nullptr),
                           tiles, A, tokens, l.dfault);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const auto h = download(dh, std::size_t(A + 1U) * ng);
        const unsigned fault = download(l.dfault, 1U)[0];
        unsigned untouched_bad = 0, written_bad = 0, guard = 0;
        for (unsigned t = 0; t < tiles; ++t) {
            const bool refused = c != clean && c != map_past && (t == victim || (c == rows_over_max && p.tile_rows[t] > s.max_rows));
            for (unsigned r = 0; r < p.tile_rows[t]; ++r) {
                const float* row = h.data() + std::size_t(p.tile_first[t] + r) * ng;
                unsigned nan_pattern = 0;
                for (unsigned col = 0; col < ng; ++col) nan_pattern += bits(row[col]) == 0xffffffffU;
                if (refused) untouched_bad += nan_pattern != ng;
                else written_bad += nan_pattern != 0U;
            }
        }
        for (unsigned col = 0; col < ng; ++col) guard += bits(h[std::size_t(A) * ng + col]) != 0xffffffffU;
        const bool want_fault = c != clean;
        const bool ok = (fault == (want_fault ? gi::kFaultBounds : 0U)) && !untouched_bad && !written_bad && !guard;
        std::printf("%s fault case '%s' (tile %u): fault 0x%08x stored-in-refused %u unwritten-live %u guard %u\n", ok ? "PASS" : "FAIL",
                    names[c], victim, fault, untouched_bad, written_bad, guard);
        fails += ok ? 0 : 1;
        CHECK(hipFree(dt));
        CHECK(hipFree(dm));
    }
    // Verdicts of the validation launch alone (count, coverage, planner failure).
    const auto verdict = [&](const char* what, std::vector<TileRows> tl, unsigned count, unsigned planned, unsigned want) {
        gi::TileSource s = source(l, gate, gim, true, ng, dh, ng);
        TileRows* dt = upload(tl);
        unsigned* dc = upload(std::vector<unsigned>{count});
        unsigned* df = upload(std::vector<unsigned>{planned});
        s.tiles = s.index_tiles = dt;
        s.tile_count = dc;
        CHECK(hipMemset(l.dfault, 0, 4U));
        hipLaunchKernelGGL(gi::detail::validate_kernel, dim3(1), dim3(1024), 0, nullptr, s, s, static_cast<const unsigned*>(df), tiles, A,
                           tokens, l.dfault);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        const unsigned got = download(l.dfault, 1U)[0];
        std::printf("%s verdict '%s': 0x%08x (want 0x%08x)\n", got == want ? "PASS" : "FAIL", what, got, want);
        fails += got == want ? 0 : 1;
        CHECK(hipFree(dt));
        CHECK(hipFree(dc));
        CHECK(hipFree(df));
    };
    std::vector<TileRows> overlap = l.gate_tiles;
    overlap[1].input = overlap[0].input;  // tile 1 over tile 0's rows (conflicting destinations), row sum unchanged
    verdict("clean plan", l.gate_tiles, tiles, 0U, 0U);
    verdict("device count one short (truncated cover)", l.gate_tiles, tiles - 1U, 0U, gi::kFaultBounds);
    verdict("device count zero", l.gate_tiles, 0U, 0U, gi::kFaultBounds);
    verdict("device count past the plan", l.gate_tiles, tiles + 1U, 0U, gi::kFaultBounds);
    verdict("overlapping tiles", overlap, tiles, 0U, gi::kFaultBounds);
    verdict("planner failure copied", l.gate_tiles, tiles, 5U, 5U);
    CHECK(hipFree(dh));
    return fails;
}

// The whole pass (gi::launch) at Inkling 2 Mini shapes with a skewed routing: run-to-run identity under different
// workspace contents, refusals that store nothing, derived destinations, workspace admission.
int pass_cases() {
    std::mt19937 g(21U);
    const unsigned tokens = 515U, experts = 128U, k = 8U, hidden = 2304U, ffn = 1152U, ng = 2U * ffn;
    const Bank gate = make_bank(packed::MatrixFormat::nvfp4, experts, ng, hidden, packed::RowOrder::contiguous, g);
    const Bank down = make_bank(packed::MatrixFormat::mxfp8, experts, hidden, ffn, packed::RowOrder::contiguous, g);
    const Plan p = route(tokens, experts, k, g, true, 256U);
    const unsigned A = static_cast<unsigned>(p.token.size()), tiles = static_cast<unsigned>(p.tile_first.size());
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(std::size_t(tokens) * hidden);
    for (auto& v : x) v = nd(g);
    const float* dx = upload(x);
    const Image gim = requantise(gate, ffn), dim = requantise(down, 0U);
    gi::LayerImages images;
    images.gate_up = gim.view;
    images.down = dim.view;
    images.gate_up_codes = gate.dcodes;
    images.gate_up_stride = std::size_t(gate.rows) * gate.code_row();
    images.down_codes = down.dcodes;
    images.down_stride = std::size_t(down.rows) * down.code_row();
    const std::size_t act_size = gi::act_bytes(tokens, A, hidden, ffn), y_floats = std::size_t(A + 1U) * hidden;
    std::uint8_t* act = nullptr;
    float *dh = nullptr, *dy = nullptr;
    CHECK(hipMalloc(&act, act_size));
    CHECK(hipMalloc(&dh, std::size_t(A) * ffn * 4U));
    CHECK(hipMalloc(&dy, y_floats * 4U));
    Launch l = make_launch(p, gate, down, dy, hidden, tiles);
    unsigned* dfail = upload(std::vector<unsigned>{0U});
    gi::Pass pass;
    pass.tokens = dx;
    pass.rows = tokens;
    pass.assignments = A;
    pass.hidden = hidden;
    pass.intermediate = ffn;
    pass.assignment_rows = l.dmap;
    pass.gate_up = l.dgate;
    pass.down = l.ddown;
    pass.tiles = tiles;
    pass.max_tile_rows = l.max_rows;
    pass.tile_count = l.dcount;
    pass.failure = dfail;
    pass.index_base = l.index_base;
    pass.h = dh;
    pass.act = act;
    pass.act_size = act_size;
    pass.images = &images;
    pass.gate_cap = 4.0F;
    pass.up_cap = 25.0F;
    pass.fault = dfail;  // the dispatcher's alias: verdict word = planner failure word
    pass.y = dy;
    const std::size_t inner_at = gi::act_image_bytes(tokens, hidden);
    struct Out {
        std::vector<std::uint8_t> act;
        std::vector<float> y;
        unsigned verdict;
        hipError_t status;
    };
    const auto run = [&](int poison, unsigned planned, unsigned count, const gi::Pass& q) {
        CHECK(hipMemset(act, poison, act_size));
        CHECK(hipMemset(dh, poison, std::size_t(A) * ffn * 4U));
        CHECK(hipMemset(dy, poison, y_floats * 4U));
        CHECK(hipMemcpy(dfail, &planned, 4U, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(l.dcount, &count, 4U, hipMemcpyHostToDevice));
        const hipError_t st = gi::launch(q, nullptr);
        CHECK(hipDeviceSynchronize());
        return Out{download(act, act_size), download(dy, y_floats), download(dfail, 1U)[0], st};
    };
    // Bytes the pass defines: the two images' code / u / base regions (alignment gaps excluded) and the A down rows.
    const auto image_same = [&](const Out& a, const Out& b, std::size_t at, unsigned r, unsigned in) {
        const std::size_t parts[3] = {ad::act_code_bytes(r, in), ad::act_u_bytes(r, in), ad::act_base_bytes(r)};
        unsigned differ = 0;
        for (const std::size_t n : parts) {
            differ += std::memcmp(a.act.data() + at, b.act.data() + at, n) != 0;
            at += gi::align256(n);
        }
        return differ;
    };
    const auto rows_same = [&](const Out& a, const Out& b) {
        return std::memcmp(a.y.data(), b.y.data(), std::size_t(A) * hidden * 4U) == 0;
    };
    const auto poisoned = [&](const std::vector<float>& y, std::size_t from, std::size_t to, int poison) {
        const std::uint32_t want = 0x01010101U * static_cast<std::uint8_t>(poison);
        unsigned bad = 0;
        for (std::size_t i = from; i < to; ++i) bad += bits(y[i]) != want;
        return bad;
    };
    int fails = 0;
    const auto report = [&](bool ok, const char* what) {
        std::printf("%s pass %s (tokens %u assignments %u tiles %u)\n", ok ? "PASS" : "FAIL", what, tokens, A, tiles);
        fails += ok ? 0 : 1;
    };
    const Out a = run(0xff, 0U, tiles, pass), b = run(0x5a, 0U, tiles, pass), c = run(0x00, 0U, tiles, pass);
    const unsigned img_ab = image_same(a, b, 0U, tokens, hidden) + image_same(a, b, inner_at, A, ffn);
    const unsigned img_ac = image_same(a, c, 0U, tokens, hidden) + image_same(a, c, inner_at, A, ffn);
    report(a.status == hipSuccess && a.verdict == 0U && b.verdict == 0U && c.verdict == 0U && img_ab == 0U && img_ac == 0U &&
               rows_same(a, b) && rows_same(a, c) && poisoned(a.y, std::size_t(A) * hidden, y_floats, 0xff) == 0U,
           "three runs over 0xff / 0x5a / 0x00 workspaces: images, down rows and guard row identical");
    unsigned nonfinite = 0;
    for (std::size_t i = 0; i < std::size_t(A) * hidden; ++i) nonfinite += !std::isfinite(a.y[i]);
    report(nonfinite == 0U, "every down row written (finite)");
    const Out f = run(0x5a, 1U, tiles, pass);
    report(f.status == hipSuccess && f.verdict == 1U && poisoned(f.y, 0U, y_floats, 0x5a) == 0U &&
               image_same(f, run(0x5a, 1U, tiles, pass), inner_at, A, ffn) == 0U &&
               f.act[inner_at] == 0x5aU && f.act[inner_at + ad::act_code_bytes(A, ffn) - 1U] == 0x5aU,
           "planner failure: no down row, no h image byte stored");
    const Out t = run(0x5a, 0U, tiles - 1U, pass);
    report(t.verdict == gi::kFaultBounds && poisoned(t.y, 0U, y_floats, 0x5a) == 0U, "short device count: refused, nothing stored");
    std::vector<TileRows> past = l.down_tiles;
    past[0].output = dy + y_floats;  // one past the destination
    past[1].output = past[0].output;  // and a conflicting destination
    TileRows* dpast = upload(past);
    gi::Pass q = pass;
    q.down = dpast;
    const Out o = run(0xff, 0U, tiles, q);
    report(o.verdict == 0U && rows_same(a, o) && poisoned(o.y, std::size_t(A) * hidden, y_floats, 0xff) == 0U,
           "down tile outputs one past / conflicting: ignored, rows as the clean pass, guard untouched");
    gi::Pass shortws = pass;
    --shortws.act_size;
    const Out w = run(0x5a, 0U, tiles, shortws);
    report(w.status == hipErrorInvalidValue && poisoned(w.y, 0U, y_floats, 0x5a) == 0U && w.act[0] == 0x5aU,
           "workspace one byte short: refused, nothing enqueued (exact size admitted above)");
    CHECK(hipFree(dpast));
    CHECK(hipFree(act));
    CHECK(hipFree(dh));
    CHECK(hipFree(dy));
    return fails;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") return bench(argc > 2 ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 8237U);
    int fails = 0;
    using F = packed::MatrixFormat;
    using O = packed::RowOrder;
    fails += fixture(97U, 5U, 3U, 256U, 128U, F::nvfp4, O::contiguous, false, 1U, false);
    fails += fixture(300U, 4U, 2U, 384U, 128U, F::nvfp4, O::interleaved_halves, true, 2U, false);  // tiles of 128 + tails
    fails += fixture(150U, 6U, 4U, 256U, 128U, F::mxfp8, O::contiguous, false, 3U, false);
    fails += fixture(200U, 3U, 2U, 128U, 256U, F::nvfp4, O::contiguous, true, 4U, true);  // device tile count cuts the last tile
    fails += fixture(700U, 3U, 2U, 256U, 128U, F::nvfp4, O::contiguous, true, 5U, false, 256U);  // 256-row tiles: row slices
    fails += faults();
    fails += pass_cases();
    std::printf("%s\n", fails ? "FAIL" : "ALL PASS");
    return fails ? 1 : 0;
}

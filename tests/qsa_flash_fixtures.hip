// qpf2k/attn directed fixtures (standalone; build with the engine TU's flags: amdclang++ -O3 -std=gnu++20
// --offload-arch=gfx1151 -Iinclude -x hip). Exit 0 only if every check passes and every path counter it requires is non-zero.
//  S: scorer cases (crafted BF16 queries and pooled keys) - qsa_flash::score_kernel<Rows, 16, 8> (both query feeds)
//     vs the production kernel of that group size (block_score_parked_kernel<64, 4>; block_score_pooled_rows_kernel
//     <Rows, 4> for 16 / 32 rows), every score bitwise; raw head dots (before relu) of the fused and the separate
//     form vs the production lane tree, bitwise where the guard admits the fused form (NaNs compared as a class).
//  G: guard boundary cases, counted per path.
//  L: selector cases (crafted score rows; maxima from the scorer's own epilogue code) - qsa_flash::select_kernel vs
//     select_rows_kernel, selected lists (canary-filled buffers, whole capacity) and post-selection scores bitwise.
#include "qsa_flash_prod.inc"
#include <map>
#include <chrono>
#include <algorithm>

namespace strix {
namespace {
// Raw head dots of the production lane tree (block_score_parked_kernel's products and qsa_scatter, park[] values
// written out): dots[(r * blocks + b) * 4 + h].
template <unsigned Rows>
__global__ __launch_bounds__(256) void prod_dots_kernel(QsaRowsScoreViews<Rows> views, const float* pooled_keys, std::uint32_t blocks, float* dots) {
#pragma clang fp contract(off)
    constexpr unsigned Heads = 4U, Dim = 128U;
    const auto lane = threadIdx.x % 32U, wave = blockIdx.x * 8U + threadIdx.x / 32U;
    const auto r = wave / blocks, b = wave % blocks;
    if (r >= Rows) return;
    const auto* key = pooled_keys + static_cast<std::uint64_t>(b) * Dim + lane;
    const float k0 = key[0], k1 = key[32], k2 = key[64], k3 = key[96];
    float partial[Heads];
#pragma unroll
    for (unsigned h = 0; h < Heads; ++h) {
        const float* q = views.index_query[r] + h * Dim + lane;
        const float p0 = q[0] * k0, p1 = q[32] * k1, p2 = q[64] * k2, p3 = q[96] * k3;
        partial[h] = (p0 + p2) + (p1 + p3);
    }
    qsa_scatter<Heads, 0U>(partial, lane);
    if ((lane & 7U) == 0U) dots[(static_cast<std::uint64_t>(r) * blocks + b) * Heads + lane / 8U] = partial[0];
}
// Raw head dots of the qpf2k tree in a given form (lane = block).
template <unsigned Rows, bool Fused>
__global__ __launch_bounds__(32) void flash_dots_kernel(QsaRowsScoreViews<Rows> views, const float* pooled_keys, std::uint32_t blocks, float* dots) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32U, b = base + lane, r = blockIdx.y;
    float k[qsa_flash::kDim];
    const auto* key = pooled_keys + static_cast<std::uint64_t>(b < blocks ? b : blocks - 1U) * qsa_flash::kDim;
    for (unsigned d = 0; d < qsa_flash::kDim; ++d) k[d] = key[d];
    const auto q = reinterpret_cast<qsa_flash::uniform_ptr>(reinterpret_cast<std::uintptr_t>(views.index_query[r]));
#pragma unroll
    for (unsigned h = 0; h < qsa_flash::kHeads; ++h) {
        const float d = qsa_flash::head_dot<Fused>(q + h * qsa_flash::kDim, k);
        if (b < blocks) dots[(static_cast<std::uint64_t>(r) * blocks + b) * qsa_flash::kHeads + h] = d;
    }
}
// The guard per (row, 32-block tile) as score_kernel evaluates it (exponents from the same helpers).
template <unsigned Rows>
__global__ __launch_bounds__(32) void guard_kernel(QsaRowsScoreViews<Rows> views, const uint2* ranges, std::uint32_t blocks, std::uint8_t* fused) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32U, b = base + lane, r = blockIdx.y;
    const float* q = views.index_query[r];
    unsigned lo = 0xffffffffU, hi = 0U;
    for (unsigned i = lane; i < qsa_flash::kHeads * qsa_flash::kDim; i += 32U) qsa_flash::magnitude_range(q[i], lo, hi);
    lo = qsa_flash::wave_min_u(lo); hi = qsa_flash::wave_max_u(hi);
    const uint2 range = ranges[b < blocks ? b : blocks - 1U];
    const int klo = qsa_flash::lo_exponent(qsa_flash::wave_min_u(range.x)), khi = qsa_flash::hi_exponent(qsa_flash::wave_max_u(range.y));
    if (lane == 0U) fused[r * ((blocks + 31U) / 32U) + blockIdx.x] =
        qsa_flash::fused_exact(qsa_flash::lo_exponent(lo), qsa_flash::hi_exponent(hi), klo, khi) ? 1U : 0U;
}
// Group maxima of score rows with score_kernel's epilogue statements (lane = block, -inf past the row's blocks).
template <unsigned Rows>
__global__ __launch_bounds__(32) void group_max_kernel(QsaRowsSelectViews<Rows> views, std::uint32_t ratio, float* maxima, std::uint32_t stride) {
    const unsigned lane = threadIdx.x, r = blockIdx.y, b = blockIdx.x * 32U + lane;
    const unsigned blocks = views.visible[r] / ratio;
    if (blockIdx.x * 32U >= blocks) return;
    const float score = b < blocks ? views.block_scores[r][b] : 0.0F;
    float m = b < blocks ? score : -INFINITY;
    m = fmaxf(m, __shfl_xor(m, 1, 32));
    m = fmaxf(m, __shfl_xor(m, 2, 32));
    if ((lane & 3U) == 0U && b < blocks) maxima[static_cast<std::uint64_t>(r) * stride + b / qsa_flash::kGroup] = m;
}
// Squared sums of every (row, head) vector: the production LDS tree (norm_rope_group_kernel's statements) and the wave form.
__global__ __launch_bounds__(256) void square_sum_lds_kernel(const float* x, std::uint32_t dim, float* out) {
    __shared__ float reduction[256];
    const auto feature = static_cast<std::uint32_t>(threadIdx.x);
    const float input = feature < dim ? x[static_cast<std::uint64_t>(blockIdx.x) * dim + feature] : 0.0F;
    if (feature < dim) reduction[feature] = input * input;
    __syncthreads();
    for (std::uint32_t stride = dim / 2U; stride > 0; stride >>= 1U) {
        if (feature < stride) reduction[feature] += reduction[feature + stride];
        __syncthreads();
    }
    if (feature == 0U) out[blockIdx.x] = reduction[0];
}
__global__ __launch_bounds__(32) void square_sum_wave_kernel(const float* x, std::uint32_t dim, float* out) {
    const unsigned lane = threadIdx.x, J = dim / 32U;
    float input[8];
    for (unsigned j = 0; j < 8U; ++j) input[j] = j < J ? x[static_cast<std::uint64_t>(blockIdx.x) * dim + lane + 32U * j] : 0.0F;
    const float t = qsa_flash::wave_square_sum(input, J);
    if (lane == 0U) out[blockIdx.x] = t;
}
} // namespace
} // namespace strix

using namespace strix;
static int failures = 0;
static void check(hipError_t e, const char* what) { if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(2); } }
static void expect(bool ok, const std::string& what) { if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); } }
template <class T> static T* dalloc(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "hipMalloc"); return static_cast<T*>(p); }
static float bf(float v) { std::uint32_t b; std::memcpy(&b, &v, 4); b = (b + 0x7fffU + ((b >> 16U) & 1U)) & 0xffff0000U; float r; std::memcpy(&r, &b, 4); return r; }
static float bits(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::uint32_t ubits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool same_or_nan(float a, float b) { return ubits(a) == ubits(b) || (std::isnan(a) && std::isnan(b)); }
struct Rng { std::uint64_t x; std::uint32_t next() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return static_cast<std::uint32_t>(x >> 16); }
    float normal() { float t = 0; for (int i = 0; i < 4; ++i) t += static_cast<float>(next() >> 8) * (1.0F / 16777216.0F); return (t - 2.0F) * 1.7320508F; } };

constexpr unsigned kDim = 128, kHeads = 4, kRatio = 4, kTopk = 512, kMaxsel = 2051;
constexpr std::uint32_t kStride = 65536U / qsa_flash::kGroup;

// ---------------------------------------------------------------------------------------------- scorer cases
struct ScoreCase { std::string name; std::vector<float> q; std::vector<float> keys; };   // q: Rows x 512, keys: blocks x 128 (BF16 values)

template <unsigned Rows>
static void run_score_case(const ScoreCase& c, std::uint32_t blocks, hipStream_t s, std::map<std::string, std::uint64_t>& paths) {
    auto* d_q = dalloc<float>(c.q.size()); check(hipMemcpy(d_q, c.q.data(), c.q.size() * 4, hipMemcpyHostToDevice), "h2d q");
    auto* d_k = dalloc<float>(c.keys.size()); check(hipMemcpy(d_k, c.keys.data(), c.keys.size() * 4, hipMemcpyHostToDevice), "h2d k");
    auto* d_r = dalloc<uint2>(blocks);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((blocks + 255U) / 256U), dim3(256), 0, s, d_k, 0U, blocks, d_r);
    const std::size_t n = static_cast<std::size_t>(Rows) * 65536U;
    auto* d_ref = dalloc<float>(n); auto* d_new = dalloc<float>(n); auto* d_lds = dalloc<float>(n);
    check(hipMemsetAsync(d_ref, 0xA5, n * 4, s), "memset"); check(hipMemsetAsync(d_new, 0xA5, n * 4, s), "memset"); check(hipMemsetAsync(d_lds, 0xA5, n * 4, s), "memset");
    QsaRowsScoreViews<Rows> ref{}, nw{}, ld{};
    for (unsigned r = 0; r < Rows; ++r) {
        // rows see a varying prefix (some rows score fewer tiles, partial last tiles), every row > 0 blocks
        const std::uint32_t rb = r % 5U == 4U ? blocks - 37U * (r % 3U) - 5U : blocks;
        ref.index_query[r] = nw.index_query[r] = ld.index_query[r] = d_q + static_cast<std::size_t>(r) * kHeads * kDim;
        ref.block_scores[r] = d_ref + static_cast<std::size_t>(r) * 65536U; nw.block_scores[r] = d_new + static_cast<std::size_t>(r) * 65536U;
        ld.block_scores[r] = d_lds + static_cast<std::size_t>(r) * 65536U;
        ref.blocks[r] = nw.blocks[r] = ld.blocks[r] = rb;
    }
    if constexpr (Rows >= 64U) {
        const auto waves = static_cast<std::uint64_t>(Rows / 4U) * ((blocks + 31U) / 32U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::block_score_parked_kernel<Rows, 4U>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)), dim3(256), 0, s, ref, d_k, blocks, kDim);
    } else {
        const auto waves = static_cast<std::uint64_t>(Rows / 4U) * ((blocks + 31U) / 32U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_score_pooled_rows_kernel<Rows, 4U>), dim3(static_cast<std::uint32_t>((waves + 7U) / 8U)), dim3(256), 0, s, ref, d_k, blocks, kDim);
    }
    auto* d_stats = dalloc<std::uint32_t>(qsa_flash::kStats); check(hipMemsetAsync(d_stats, 0, qsa_flash::kStats * 4, s), "memset");
    constexpr unsigned RPC = Rows < qsa_flash::kScoreRows ? Rows : qsa_flash::kScoreRows, W = qsa_flash::kScoreWaves;
    const unsigned tiles = (blocks + 31U) / 32U;
    auto* d_max = dalloc<float>(static_cast<std::size_t>(Rows) * kStride);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, RPC, W, true, false>), dim3((tiles + W - 1U) / W, Rows / RPC), dim3(W * 32U), 0, s,
        nw, d_k, blocks, kDim, d_max, kStride, d_r, d_stats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, RPC, W, false, true>), dim3((tiles + W - 1U) / W, Rows / RPC), dim3(W * 32U), 0, s,
        ld, d_k, blocks, kDim, nullptr, kStride, d_r, nullptr);
    // raw dots
    const std::size_t nd = static_cast<std::size_t>(Rows) * blocks * kHeads;
    auto* d_pd = dalloc<float>(nd); auto* d_fd = dalloc<float>(nd); auto* d_md = dalloc<float>(nd);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prod_dots_kernel<Rows>), dim3((Rows * blocks + 7U) / 8U), dim3(256), 0, s, ref, d_k, blocks, d_pd);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(flash_dots_kernel<Rows, true>), dim3(tiles, Rows), dim3(32), 0, s, nw, d_k, blocks, d_fd);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(flash_dots_kernel<Rows, false>), dim3(tiles, Rows), dim3(32), 0, s, nw, d_k, blocks, d_md);
    auto* d_g = dalloc<std::uint8_t>(static_cast<std::size_t>(Rows) * tiles);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(guard_kernel<Rows>), dim3(tiles, Rows), dim3(32), 0, s, nw, d_r, blocks, d_g);
    check(hipGetLastError(), "launch"); check(hipStreamSynchronize(s), "sync");
    std::vector<float> a(n), b(n), l(n); check(hipMemcpy(a.data(), d_ref, n * 4, hipMemcpyDeviceToHost), "d2h");
    check(hipMemcpy(b.data(), d_new, n * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(l.data(), d_lds, n * 4, hipMemcpyDeviceToHost), "d2h");
    std::size_t diff = 0, diffl = 0;
    for (std::size_t i = 0; i < n; ++i) { diff += ubits(a[i]) != ubits(b[i]); diffl += ubits(a[i]) != ubits(l[i]); }   // incl. canaries past each row's blocks
    std::vector<float> pd(nd), fd(nd), md(nd); std::vector<std::uint8_t> g(static_cast<std::size_t>(Rows) * tiles);
    check(hipMemcpy(pd.data(), d_pd, nd * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(fd.data(), d_fd, nd * 4, hipMemcpyDeviceToHost), "d2h");
    check(hipMemcpy(md.data(), d_md, nd * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(g.data(), d_g, g.size(), hipMemcpyDeviceToHost), "d2h");
    std::size_t mdiff = 0, fdiff = 0, nanpay = 0, fused_tiles = 0;
    for (unsigned r = 0; r < Rows; ++r)
        for (std::uint32_t bl = 0; bl < blocks; ++bl) {
            const bool fused = g[r * tiles + bl / 32U] != 0;
            fused_tiles += fused && bl % 32U == 0U;
            for (unsigned h = 0; h < kHeads; ++h) {
                const auto i = (static_cast<std::size_t>(r) * blocks + bl) * kHeads + h;
                if (!same_or_nan(pd[i], md[i])) ++mdiff;
                else if (ubits(pd[i]) != ubits(md[i])) ++nanpay;
                if (fused && !same_or_nan(pd[i], fd[i])) ++fdiff;
            }
        }
    std::vector<std::uint32_t> st(qsa_flash::kStats); check(hipMemcpy(st.data(), d_stats, st.size() * 4, hipMemcpyDeviceToHost), "d2h");
    paths["fused"] += st[qsa_flash::kFused]; paths["separate"] += st[qsa_flash::kSeparate];
    std::printf("  S %-28s rows %2u blocks %5u: scores differ %zu (lds feed %zu), separate-form dots differ %zu (NaN payload only %zu), "
        "fused-form dots differ where admitted %zu, admitted tiles %zu/%zu, path counts fused %u separate %u\n",
        c.name.c_str(), Rows, blocks, diff, diffl, mdiff, nanpay, fdiff, fused_tiles, static_cast<std::size_t>(Rows) * tiles,
        st[qsa_flash::kFused], st[qsa_flash::kSeparate]);
    expect(diff == 0 && diffl == 0 && mdiff == 0 && fdiff == 0, "scorer case " + c.name);
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_k), static_cast<void*>(d_r), static_cast<void*>(d_ref), static_cast<void*>(d_new),
                    static_cast<void*>(d_lds), static_cast<void*>(d_stats), static_cast<void*>(d_max), static_cast<void*>(d_pd), static_cast<void*>(d_fd),
                    static_cast<void*>(d_md), static_cast<void*>(d_g)}) check(hipFree(p), "free");
}

// ---------------------------------------------------------------------------------------------- selector cases
struct SelectCase { std::string name; std::vector<std::vector<float>> rows; std::vector<std::uint32_t> visible; };

static void run_select_case(const SelectCase& c, hipStream_t s, std::map<std::string, std::uint64_t>& paths) {
    constexpr unsigned Rows = 64;
    const std::size_t n = static_cast<std::size_t>(Rows) * 65536U;
    auto* d_a = dalloc<float>(n); auto* d_b = dalloc<float>(n);
    std::vector<float> host(n, bits(0x7f7fbeefU));
    for (unsigned r = 0; r < Rows; ++r) std::copy(c.rows[r].begin(), c.rows[r].end(), host.begin() + static_cast<std::ptrdiff_t>(r) * 65536);
    check(hipMemcpy(d_a, host.data(), n * 4, hipMemcpyHostToDevice), "h2d"); check(hipMemcpy(d_b, host.data(), n * 4, hipMemcpyHostToDevice), "h2d");
    auto* d_sa = dalloc<std::uint32_t>(Rows * kMaxsel); auto* d_sb = dalloc<std::uint32_t>(Rows * kMaxsel);
    check(hipMemset(d_sa, 0xAA, Rows * kMaxsel * 4), "memset"); check(hipMemset(d_sb, 0xAA, Rows * kMaxsel * 4), "memset");
    auto* d_max = dalloc<float>(static_cast<std::size_t>(Rows) * kStride); check(hipMemset(d_max, 0xCD, static_cast<std::size_t>(Rows) * kStride * 4), "memset");
    QsaRowsSelectViews<Rows> va{}, vb{};
    for (unsigned r = 0; r < Rows; ++r) {
        va.block_scores[r] = d_a + static_cast<std::size_t>(r) * 65536U; vb.block_scores[r] = d_b + static_cast<std::size_t>(r) * 65536U;
        va.selected[r] = d_sa + r * kMaxsel; vb.selected[r] = d_sb + r * kMaxsel; va.visible[r] = vb.visible[r] = c.visible[r];
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(group_max_kernel<Rows>), dim3(65536U / 32U, Rows), dim3(32), 0, s, vb, kRatio, d_max, kStride);
    auto* d_stats = dalloc<std::uint32_t>(qsa_flash::kStats); check(hipMemsetAsync(d_stats, 0, qsa_flash::kStats * 4, s), "memset");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(select_rows_kernel<Rows>), dim3(Rows), dim3(kSelectThreads), 0, s, va, kRatio, kTopk);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_kernel<Rows, true>), dim3(Rows), dim3(kSelectThreads), 0, s, vb, d_max, kStride, kRatio, kTopk, d_stats);
    check(hipGetLastError(), "launch"); check(hipStreamSynchronize(s), "sync");
    std::vector<std::uint32_t> sa(Rows * kMaxsel), sb(Rows * kMaxsel);
    check(hipMemcpy(sa.data(), d_sa, sa.size() * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(sb.data(), d_sb, sb.size() * 4, hipMemcpyDeviceToHost), "d2h");
    std::vector<float> pa(n), pb(n); check(hipMemcpy(pa.data(), d_a, n * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(pb.data(), d_b, n * 4, hipMemcpyDeviceToHost), "d2h");
    std::size_t rows_sel = 0, rows_post = 0;
    for (unsigned r = 0; r < Rows; ++r) {
        rows_sel += std::memcmp(sa.data() + r * kMaxsel, sb.data() + r * kMaxsel, kMaxsel * 4) != 0;
        rows_post += std::memcmp(pa.data() + static_cast<std::size_t>(r) * 65536U, pb.data() + static_cast<std::size_t>(r) * 65536U, 65536U * 4) != 0;
    }
    std::vector<std::uint32_t> st(qsa_flash::kStats); check(hipMemcpy(st.data(), d_stats, st.size() * 4, hipMemcpyDeviceToHost), "d2h");
    paths["select_candidates"] += st[qsa_flash::kSelectCandidates]; paths["select_few_maxima"] += st[qsa_flash::kSelectFewMaxima];
    paths["select_capacity"] += st[qsa_flash::kSelectCapacity]; paths["select_dense"] += st[qsa_flash::kSelectDense];
    std::printf("  L %-30s rows with different selections %zu, different post-selection scores %zu; paths candidates %u few-maxima %u capacity %u dense %u, "
        "candidate groups mean %.0f max %u\n", c.name.c_str(), rows_sel, rows_post, st[qsa_flash::kSelectCandidates], st[qsa_flash::kSelectFewMaxima],
        st[qsa_flash::kSelectCapacity], st[qsa_flash::kSelectDense],
        st[qsa_flash::kSelectCandidates] ? static_cast<double>(st[qsa_flash::kCandidateGroups]) / st[qsa_flash::kSelectCandidates] : 0.0, st[qsa_flash::kCandidateMax]);
    expect(rows_sel == 0 && rows_post == 0, "selector case " + c.name);
    for (void* p : {static_cast<void*>(d_a), static_cast<void*>(d_b), static_cast<void*>(d_sa), static_cast<void*>(d_sb), static_cast<void*>(d_max),
                    static_cast<void*>(d_stats)}) check(hipFree(p), "free");
}


// ---------------------------------------------------------------------------------------------- norm / RoPE cases
template <unsigned Rows>
static void run_norm_case(const char* name, unsigned mode, std::uint64_t pos0, hipStream_t s, std::map<std::string, std::uint64_t>& paths) {
    Rng rng{0x9E3779B97F4A7C15ULL ^ (pos0 * 31U + mode)};
    const unsigned dims[3] = {128U, 256U, 256U}, heads[3] = {4U, 24U, 2U}, strides[3] = {128U, 512U, 256U};
    const std::size_t width[3] = {4U * 128U, 24U * 512U, 2U * 256U};
    std::vector<float> h[3];
    for (unsigned t = 0; t < 3U; ++t) {
        h[t].resize(Rows * width[t]);
        for (std::size_t i = 0; i < h[t].size(); ++i) {
            float v = rng.normal() * (t == 1 ? 3.0F : 1.0F);
            if (mode == 1U && i % 97U == 3U) v = std::ldexp(1.0F, -130);
            if (mode == 1U && i % 211U == 7U) v = std::ldexp(1.5F, 60);
            if (mode == 1U && (i / width[t]) % 7U == 2U) v = 0.0F;
            if (mode == 1U && i % 499U == 11U) v = -0.0F;
            h[t][i] = v;
        }
    }
    std::vector<std::uint16_t> w[3];
    for (unsigned t = 0; t < 3U; ++t) { w[t].resize(dims[t]); for (auto& x : w[t]) { const float v = bf(1.0F + 0.2F * rng.normal()); std::uint32_t b; std::memcpy(&b, &v, 4); x = static_cast<std::uint16_t>(b >> 16U); } }
    float *da[3], *db[3]; std::uint16_t* dw[3];
    for (unsigned t = 0; t < 3U; ++t) {
        da[t] = dalloc<float>(h[t].size()); db[t] = dalloc<float>(h[t].size()); dw[t] = dalloc<std::uint16_t>(dims[t]);
        check(hipMemcpy(da[t], h[t].data(), h[t].size() * 4, hipMemcpyHostToDevice), "h2d"); check(hipMemcpy(db[t], h[t].data(), h[t].size() * 4, hipMemcpyHostToDevice), "h2d");
        check(hipMemcpy(dw[t], w[t].data(), dims[t] * 2, hipMemcpyHostToDevice), "h2d");
    }
    auto* table = dalloc<float2>(static_cast<std::size_t>(Rows) * 32U);
    qsa_index::NormGroupViews<Rows> va{}, vb{};
    for (unsigned t = 0; t < 3U; ++t) {
        for (unsigned r = 0; r < Rows; ++r) { va.values[t][r] = da[t] + r * width[t]; vb.values[t][r] = db[t] + r * width[t]; }
        va.weights[t] = vb.weights[t] = dw[t]; va.heads[t] = vb.heads[t] = heads[t]; va.dim[t] = vb.dim[t] = dims[t]; va.stride[t] = vb.stride[t] = strides[t];
    }
    for (unsigned r = 0; r < Rows; ++r) va.position[r] = vb.position[r] = pos0 + r;
    va.table = vb.table = table;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows>), dim3(Rows), dim3(64), 0, s, va, 64U, 1e7F);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<SmallWeightKind::bf16, Rows>), dim3(30U, Rows), dim3(256), 0, s, va, 1e-6F, 64U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<SmallWeightKind::bf16, Rows>), dim3((Rows * 30U + 7U) / 8U), dim3(256), 0, s, vb, 1e-6F, 64U);
    check(hipGetLastError(), "launch"); check(hipStreamSynchronize(s), "sync");
    std::size_t differ = 0, n = 0;
    for (unsigned t = 0; t < 3U; ++t) {
        std::vector<float> a(h[t].size()), b(h[t].size());
        check(hipMemcpy(a.data(), da[t], a.size() * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(b.data(), db[t], b.size() * 4, hipMemcpyDeviceToHost), "d2h");
        for (std::size_t i = 0; i < a.size(); ++i) differ += ubits(a[i]) != ubits(b[i]);
        n += a.size();
        check(hipFree(da[t]), "free"); check(hipFree(db[t]), "free"); check(hipFree(dw[t]), "free");
    }
    check(hipFree(table), "free");
    paths["norm_wave"] += 1;
    std::printf("  N %-28s rows %2u pos %8llu: %zu of %zu values differ\n", name, Rows, static_cast<unsigned long long>(pos0), differ, n);
    expect(differ == 0, std::string("norm case ") + name);
}


// ---------------------------------------------------------------------------------------------- indexer projection cases
static void run_proj_case(const char* name, unsigned mode, hipStream_t s, std::map<std::string, std::uint64_t>& paths) {
    constexpr unsigned Rows = 64U, H = 2560U, Q = 512U, K = 128U;
    Rng rng{0xC0FFEE123ULL + mode};
    const auto bfbits = [](float v) { const float b2 = bf(v); std::uint32_t b; std::memcpy(&b, &b2, 4); return static_cast<std::uint16_t>(b >> 16U); };
    std::vector<std::uint16_t> in(Rows * H), wq(Q * H), wk(K * H);
    for (std::size_t i = 0; i < in.size(); ++i) {
        float v = rng.normal();
        if (mode == 1U && (i / H) % 9U == 4U) v = 0.0F;
        if (mode == 1U && i % 1013U == 5U) v = std::ldexp(1.0F, 100);
        if (mode == 1U && i % 997U == 9U) v = std::ldexp(1.0F, -126);
        in[i] = bfbits(v);
    }
    for (auto& x : wq) x = bfbits(0.02F * rng.normal());
    for (auto& x : wk) x = bfbits(0.02F * rng.normal());
    auto* d_in = dalloc<std::uint16_t>(in.size()); auto* d_wq = dalloc<std::uint16_t>(wq.size()); auto* d_wk = dalloc<std::uint16_t>(wk.size());
    check(hipMemcpy(d_in, in.data(), in.size() * 2, hipMemcpyHostToDevice), "h2d"); check(hipMemcpy(d_wq, wq.data(), wq.size() * 2, hipMemcpyHostToDevice), "h2d");
    check(hipMemcpy(d_wk, wk.data(), wk.size() * 2, hipMemcpyHostToDevice), "h2d");
    auto* oa = dalloc<float>(Rows * (Q + K)); auto* ob = dalloc<float>(Rows * (Q + K));
    check(hipMemset(oa, 0x5A, Rows * (Q + K) * 4), "memset"); check(hipMemset(ob, 0x5A, Rows * (Q + K) * 4), "memset");
    QsaRowsInputViews<Rows> va{}, vb{};
    va.count = vb.count = 2U; va.weights[0] = vb.weights[0] = d_wq; va.weights[1] = vb.weights[1] = d_wk;
    va.first[0] = vb.first[0] = 0U; va.first[1] = vb.first[1] = Q; va.first[2] = vb.first[2] = Q + K;
    for (unsigned r = 0; r < Rows; ++r) {
        va.input[r] = vb.input[r] = d_in + static_cast<std::size_t>(r) * H;
        va.output[0][r] = oa + r * Q; va.output[1][r] = oa + Rows * Q + r * K;
        vb.output[0][r] = ob + r * Q; vb.output[1][r] = ob + Rows * Q + r * K;
    }
    hipEvent_t e0, e1; check(hipEventCreate(&e0), "ev"); check(hipEventCreate(&e1), "ev");
    float ta = 1e9F, tb = 1e9F;
    for (unsigned rep = 0; rep < 5U; ++rep) {
        float ms = 0;
        check(hipEventRecord(e0, s), "rec");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, 4U, 2U, 16U>), dim3((Q + K) / 32U, Rows / 4U), dim3(512), H * 4U * 4U, s, va, H);
        check(hipEventRecord(e1, s), "rec"); check(hipEventSynchronize(e1), "sync"); check(hipEventElapsedTime(&ms, e0, e1), "el"); ta = std::min(ta, ms);
        check(hipEventRecord(e0, s), "rec");
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::index_rows_kernel<Rows, 16U, 16U>), dim3((Q + K) / 16U, Rows / 16U), dim3(512), 0, s, vb, H);
        check(hipEventRecord(e1, s), "rec"); check(hipEventSynchronize(e1), "sync"); check(hipEventElapsedTime(&ms, e0, e1), "el"); tb = std::min(tb, ms);
    }
    check(hipGetLastError(), "launch");
    std::vector<float> a(Rows * (Q + K)), b(Rows * (Q + K));
    check(hipMemcpy(a.data(), oa, a.size() * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(b.data(), ob, b.size() * 4, hipMemcpyDeviceToHost), "d2h");
    std::size_t differ = 0;
    for (std::size_t i = 0; i < a.size(); ++i) differ += ubits(a[i]) != ubits(b[i]);
    paths["proj_rows16"] += 1;
    std::printf("  P %-28s 64 rows x 640 outputs: %zu of %zu differ; rows_kernel %.1f us, index_rows_kernel %.1f us (64 rows, isolated)\n",
        name, differ, a.size(), ta * 1000.0F, tb * 1000.0F);
    expect(differ == 0, std::string("projection case ") + name);
    // Probe (bench only): production rows_kernel with more outputs per wave (OW) / fewer waves per workgroup (W): the
    // same per-output chains and tree (launch shape only), so every output must stay bitwise equal.
    const auto probe = [&]<unsigned OW, unsigned W>() {
        check(hipMemset(ob, 0x5A, Rows * (Q + K) * 4), "memset");
        float tc = 1e9F;
        for (unsigned rep = 0; rep < 5U; ++rep) {
            float ms = 0;
            check(hipEventRecord(e0, s), "rec");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, 4U, OW, W>), dim3((Q + K + OW * W - 1U) / (OW * W), Rows / 4U),
                dim3(W * 32U), H * 4U * 4U, s, vb, H);
            check(hipEventRecord(e1, s), "rec"); check(hipEventSynchronize(e1), "sync"); check(hipEventElapsedTime(&ms, e0, e1), "el"); tc = std::min(tc, ms);
        }
        check(hipGetLastError(), "launch");
        check(hipMemcpy(b.data(), ob, b.size() * 4, hipMemcpyDeviceToHost), "d2h");
        std::size_t dc = 0;
        for (std::size_t i = 0; i < a.size(); ++i) dc += ubits(a[i]) != ubits(b[i]);
        std::printf("  P %-28s rows_kernel<RB 4, OW %u, W %2u>: %zu of %zu differ; %.1f us (production OW 2 W 16: %.1f us)\n",
            name, OW, W, dc, a.size(), tc * 1000.0F, ta * 1000.0F);
        expect(dc == 0, std::string("projection probe ") + name);
    };
    probe.template operator()<2U, 8U>(); probe.template operator()<4U, 16U>(); probe.template operator()<4U, 8U>();
    probe.template operator()<4U, 4U>(); probe.template operator()<1U, 16U>();
    // Bit 7: qsa_index::rows_batch_kernel over 3 groups (group 0 = va's rows, group 1 = the rows reversed, group 2 = rows
    // rotated by 17) vs one rows_kernel<64, 4, 2, 16> launch per group, every output bitwise (views read from a
    // pinned host table as in the engine).
    {
        constexpr unsigned NG = 3U;
        const std::size_t per = static_cast<std::size_t>(Rows) * (Q + K);
        auto* ref = dalloc<float>(NG * per); auto* bat = dalloc<float>(NG * per);
        check(hipMemset(ref, 0x5A, NG * per * 4), "memset"); check(hipMemset(bat, 0x6B, NG * per * 4), "memset");
        QsaRowsInputViews<Rows>* table = nullptr;
        check(hipHostMalloc(reinterpret_cast<void**>(&table), NG * sizeof(QsaRowsInputViews<Rows>), hipHostMallocDefault), "pinned");
        std::vector<QsaRowsInputViews<Rows>> vr(NG);
        for (unsigned gi = 0; gi < NG; ++gi) {
            QsaRowsInputViews<Rows> v = va, w = va;
            for (unsigned r = 0; r < Rows; ++r) {
                const unsigned src = gi == 0U ? r : gi == 1U ? Rows - 1U - r : (r + 17U) % Rows;
                v.input[r] = w.input[r] = d_in + static_cast<std::size_t>(src) * H;
                v.output[0][r] = ref + gi * per + r * Q; v.output[1][r] = ref + gi * per + Rows * Q + r * K;
                w.output[0][r] = bat + gi * per + r * Q; w.output[1][r] = bat + gi * per + Rows * Q + r * K;
            }
            vr[gi] = v; table[gi] = w;
        }
        for (unsigned gi = 0; gi < NG; ++gi)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_kernel<Rows, 4U, 2U, 16U>), dim3((Q + K) / 32U, Rows / 4U), dim3(512), H * 4U * 4U, s, vr[gi], H);
        float tg = 1e9F;
        for (unsigned rep = 0; rep < 5U; ++rep) {
            float ms = 0;
            check(hipEventRecord(e0, s), "rec");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rows_batch_kernel<Rows, 4U, 2U, 16U>), dim3((Q + K) / 32U, NG * (Rows / 4U)), dim3(512),
                H * 4U * 4U, s, static_cast<const QsaRowsInputViews<Rows>*>(table), H);
            check(hipEventRecord(e1, s), "rec"); check(hipEventSynchronize(e1), "sync"); check(hipEventElapsedTime(&ms, e0, e1), "el"); tg = std::min(tg, ms);
        }
        check(hipGetLastError(), "launch"); check(hipDeviceSynchronize(), "sync");
        std::vector<float> x(NG * per), y(NG * per);
        check(hipMemcpy(x.data(), ref, x.size() * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(y.data(), bat, y.size() * 4, hipMemcpyDeviceToHost), "d2h");
        std::size_t db = 0;
        for (std::size_t i = 0; i < x.size(); ++i) db += ubits(x[i]) != ubits(y[i]);
        // group 0 must also equal the first rows_kernel run (a)
        std::size_t d0 = 0;
        for (std::size_t i = 0; i < per; ++i) d0 += ubits(x[i]) != ubits(a[i]);
        paths["proj_batch"] += 1;
        std::printf("  P %-28s rows_batch_kernel %u groups: %zu of %zu differ (group 0 vs first run: %zu); %.1f us (%.1f us per group; rows_kernel %.1f)\n",
            name, NG, db, x.size(), d0, tg * 1000.0F, tg * 1000.0F / NG, ta * 1000.0F);
        expect(db == 0 && d0 == 0, std::string("projection batch ") + name);
        check(hipHostFree(table), "free"); check(hipFree(ref), "free"); check(hipFree(bat), "free");
    }
    for (void* p : {static_cast<void*>(d_in), static_cast<void*>(d_wq), static_cast<void*>(d_wk), static_cast<void*>(oa), static_cast<void*>(ob)}) check(hipFree(p), "free");
}


// ---------------------------------------------------------------------------------------------- attend cases (share vs production)
// One 64-row group (16 tiles of 4 rows) per case: visible[r], the row's selected token list (ascending, maybe repeating
// tokens 0..3 like the selection's fill), random BF16 KV and q_gate; union_multi_kernel then attend_multi_kernel vs
// attend_share_multi_kernel, every output bitwise.
static void run_attend_case(const char* name, unsigned mode, hipStream_t s, std::map<std::string, std::uint64_t>& paths) {
    constexpr unsigned Rows = 64U, heads = 24U, kvh = 2U, hd = 256U, maxsel = 2051U, TR = 4U, G = 3U, HG = G * (16U / TR);
    Rng rng{0xA77E9D00ULL + mode};
    const auto bfbits = [](float v) { const float b2 = bf(v); std::uint32_t b; std::memcpy(&b, &b2, 4); return static_cast<std::uint16_t>(b >> 16U); };
    std::vector<std::uint32_t> visible(Rows), count(Rows);
    std::vector<std::vector<std::uint32_t>> sel(Rows);
    std::uint32_t maxvis = 0;
    for (unsigned r = 0; r < Rows; ++r) {
        std::uint32_t v = 0; std::vector<std::uint32_t> list; std::size_t fixed = 0;
        if (mode == 0U) { v = 1U + r; for (std::uint32_t t = 0; t < v; ++t) list.push_back(t); }
        else if (mode == 1U) { v = 3000U + 7U * r; for (std::uint32_t t = 0; t < v && list.size() < 2051U; ++t) if ((t / 4U + r) % 3U == 0U) list.push_back(t); }
        else if (mode == 2U) {
            v = 2100U + r; fixed = 4U * (1U + r % 3U);
            for (unsigned z = 0; z < fixed; ++z) list.push_back(z % 4U);
            for (std::uint32_t t = 64U; t < v && list.size() < 2051U; ++t) if (rng.next() % 2U) list.push_back(t);
        } else { v = 9000U + 5000U * (r % 4U); for (std::uint32_t t = 0; t < v && list.size() < 2051U; ++t) if (rng.next() % (1U + r % 9U) == 0U) list.push_back(t); }
        std::sort(list.begin() + static_cast<std::ptrdiff_t>(fixed), list.end());
        if (list.empty()) list.push_back(0U);
        visible[r] = v; count[r] = static_cast<std::uint32_t>(list.size()); sel[r] = list; maxvis = std::max(maxvis, v);
    }
    std::vector<std::uint16_t> kv(static_cast<std::size_t>(maxvis) * kvh * hd * 2U);
    for (auto& x : kv) x = bfbits(0.7F * rng.normal());
    std::vector<float> qg(static_cast<std::size_t>(Rows) * heads * hd * 2U);
    for (std::size_t i = 0; i < qg.size(); ++i) qg[i] = rng.normal() * ((i / (hd * 2U)) % 5U == 3U ? 40.0F : 1.0F); // peaky heads
    auto* d_kv = dalloc<std::uint16_t>(kv.size()); check(hipMemcpy(d_kv, kv.data(), kv.size() * 2, hipMemcpyHostToDevice), "h2d");
    auto* d_qg = dalloc<float>(qg.size()); check(hipMemcpy(d_qg, qg.data(), qg.size() * 4, hipMemcpyHostToDevice), "h2d");
    auto* d_sel = dalloc<std::uint32_t>(Rows * maxsel);
    for (unsigned r = 0; r < Rows; ++r) check(hipMemcpy(d_sel + r * maxsel, sel[r].data(), sel[r].size() * 4, hipMemcpyHostToDevice), "h2d");
    auto* d_scr = dalloc<float>(static_cast<std::size_t>(Rows) * heads * maxsel);
    const std::size_t n = static_cast<std::size_t>(Rows) * heads * hd;
    auto* oa = dalloc<float>(n); auto* ob = dalloc<float>(n);
    check(hipMemset(oa, 0xA5, n * 4), "memset"); check(hipMemset(ob, 0xA5, n * 4), "memset");
    qsa_fast::Views<Rows> va{}, vb{};
    for (unsigned r = 0; r < Rows; ++r) {
        va.q_gate[r] = vb.q_gate[r] = d_qg + static_cast<std::size_t>(r) * heads * hd * 2U; va.selected[r] = vb.selected[r] = d_sel + r * maxsel;
        va.scratch[r] = vb.scratch[r] = d_scr + static_cast<std::size_t>(r) * heads * maxsel;
        va.output[r] = oa + static_cast<std::size_t>(r) * heads * hd; vb.output[r] = ob + static_cast<std::size_t>(r) * heads * hd;
        va.count[r] = vb.count[r] = count[r]; va.visible[r] = vb.visible[r] = visible[r];
    }
    auto* d_va = dalloc<qsa_fast::Views<Rows>>(1); auto* d_vb = dalloc<qsa_fast::Views<Rows>>(1);
    check(hipMemcpy(d_va, &va, sizeof(va), hipMemcpyHostToDevice), "h2d"); check(hipMemcpy(d_vb, &vb, sizeof(vb), hipMemcpyHostToDevice), "h2d");
    const auto union_at = heads * qsa_fast::kPartial;
    const dim3 grid(Rows / TR, kvh * (heads / kvh / HG), 1U), block((2U * G + 4U) * 32U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::union_multi_kernel<Rows, TR>), dim3(Rows / TR), dim3(256), 0, s, d_va, union_at);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_multi_kernel<Rows, TR, G>), grid, block, 0, s, d_va, d_kv, kvh, heads, union_at);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_fast::attend_share_multi_kernel<Rows, TR, G>), grid, block, 0, s, d_vb, d_kv, kvh, heads, union_at);
    check(hipGetLastError(), "launch");
    const auto t0 = std::chrono::steady_clock::now(); // bounded wait: a handoff bug would hang
    while (hipStreamQuery(s) == hipErrorNotReady)
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) { std::printf("FAIL attend case %s: timeout\n", name); std::exit(3); }
    check(hipStreamSynchronize(s), "sync");
    std::vector<float> a(n), b(n);
    check(hipMemcpy(a.data(), oa, n * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(b.data(), ob, n * 4, hipMemcpyDeviceToHost), "d2h");
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) differ += ubits(a[i]) != ubits(b[i]);
    std::vector<std::uint32_t> head(4);
    double chunks = 0; unsigned cmin = 1U << 30, cmax = 0;
    for (unsigned r = 0; r < Rows; r += TR) {
        check(hipMemcpy(head.data(), d_scr + static_cast<std::size_t>(r) * heads * maxsel + union_at, 16, hipMemcpyDeviceToHost), "d2h");
        chunks += head[0] / 4U; cmin = std::min(cmin, head[0] / 4U); cmax = std::max(cmax, head[0] / 4U);
    }
    paths["attend_share"] += 1;
    std::printf("  A %-28s chunks per tile %u..%u (mean %.1f): %zu of %zu outputs differ\n", name, cmin, cmax, chunks / (Rows / TR), differ, n);
    expect(differ == 0, std::string("attend case ") + name);
    for (void* p : {static_cast<void*>(d_kv), static_cast<void*>(d_qg), static_cast<void*>(d_sel), static_cast<void*>(d_scr), static_cast<void*>(oa),
                    static_cast<void*>(ob), static_cast<void*>(d_va), static_cast<void*>(d_vb)}) check(hipFree(p), "free");
}

// Stage batching: use the actual shared production kernels, with independent,
// canary-filled arenas for the serial and stage-major schedules. NaN payloads,
// signed zeros and all unwritten bytes are compared literally (no NaN exemption).
template <SmallWeightKind Kind>
static void run_stage_batch_case(unsigned groups, bool wave, bool edges, hipStream_t s) {
    constexpr unsigned Rows = 64U, Dim = 128U, RowFloats = 1408U, Rotary = 128U;
    constexpr float Epsilon = 1.0e-6F, Theta = 10000.0F;
    const unsigned ratio = groups == 128U ? 32U : 4U, topk = edges ? 18U : 2U;
    const unsigned first = edges ? 1U : 2U * ratio + 1U, rows = groups * Rows;
    const unsigned tokens = first + rows + 2U, capacity = tokens / ratio + 8U, score_stride = capacity + 8U;
    Rng rng{0xcafef00d12345678ULL};
    std::vector<float> input(static_cast<std::size_t>(rows) * RowFloats);
    for (auto& x : input) x = bf(rng.normal());
    if (edges) {
        const std::uint32_t special[] = {0U, 0x80000000U, 0x00010000U, 0x00800000U, 0x7f7f0000U,
            0x7f800000U, 0xff800000U, 0x7fc10000U, 0xffc20000U, 0x7fa10000U};
        for (unsigned r = 0; r < rows; ++r) {
            const auto base = static_cast<std::size_t>(r) * RowFloats;
            for (unsigned set : {0U, 512U, 1024U, 1152U, 1280U})
                input[base + set + r % Dim] = bits(special[r % 10U]);
            if (r % 13U == 0U) std::fill_n(input.begin() + static_cast<std::ptrdiff_t>(base), 512U, -0.0F);
        }
    }
    std::vector<std::uint16_t> initial_raw(static_cast<std::size_t>(tokens) * Dim, 0xa5a5U);
    for (unsigned p = 0; p < first; ++p)
        for (unsigned d = 0; d < Dim; ++d) initial_raw[static_cast<std::size_t>(p) * Dim + d] =
            static_cast<std::uint16_t>(ubits(bf(rng.normal())) >> 16U);
    auto* weights = dalloc<float>(Dim);
    if constexpr (Kind == SmallWeightKind::f32) {
        const std::vector<float> w(Dim, 1.0F);
        check(hipMemcpy(weights, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice), "batch weights");
    } else {
        const std::vector<std::uint16_t> w(Dim, Kind == SmallWeightKind::f16 ? 0x3c00U : 0x3f80U);
        check(hipMemcpy(weights, w.data(), w.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice), "batch weights");
    }
    struct Arena {
        float *values, *pool, *scores, *maxima;
        float2* tables;
        uint2* ranges;
        std::uint16_t *raw, *kv, *journal;
        std::uint32_t* stats;
        qsa_index::NormGroupViews<Rows>* dn;
        QsaRowsStoreViews<Rows>* ds;
        qsa_flash::ScoreGroup<Rows>* dc;
        uint2* db;
        std::vector<qsa_index::NormGroupViews<Rows>> norm;
        std::vector<QsaRowsStoreViews<Rows>> store;
        std::vector<qsa_flash::ScoreGroup<Rows>> score;
        std::vector<uint2> bounds;
        unsigned max_new = 0, max_blocks = 0, max_heads = 0;
    };
    const std::size_t table_count = static_cast<std::size_t>(groups) * Rows * (Rotary / 2U) + 8U;
    const std::size_t score_count = static_cast<std::size_t>(rows) * score_stride;
    const std::size_t maxima_count = static_cast<std::size_t>(rows) * ((capacity + 3U) / 4U + 3U);
    const std::size_t kv_count = static_cast<std::size_t>(tokens) * Dim * 2U;
    const auto setup = [&](Arena& a) {
        a.values = dalloc<float>(input.size()); a.pool = dalloc<float>(static_cast<std::size_t>(capacity) * Dim);
        a.scores = dalloc<float>(score_count); a.maxima = dalloc<float>(maxima_count);
        a.tables = dalloc<float2>(table_count); a.ranges = dalloc<uint2>(capacity);
        a.raw = dalloc<std::uint16_t>(initial_raw.size()); a.kv = dalloc<std::uint16_t>(kv_count);
        a.journal = dalloc<std::uint16_t>(8U * Dim + 8U); a.stats = dalloc<std::uint32_t>(qsa_flash::kStats);
        a.dn = dalloc<qsa_index::NormGroupViews<Rows>>(groups); a.ds = dalloc<QsaRowsStoreViews<Rows>>(groups);
        a.dc = dalloc<qsa_flash::ScoreGroup<Rows>>(groups); a.db = dalloc<uint2>(groups);
        check(hipMemcpy(a.values, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "batch inputs");
        check(hipMemcpy(a.raw, initial_raw.data(), initial_raw.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice), "batch raw prefix");
        for (const auto& [p, bytes] : std::vector<std::pair<void*, std::size_t>>{
            {a.pool, static_cast<std::size_t>(capacity) * Dim * sizeof(float)}, {a.scores, score_count * sizeof(float)},
            {a.maxima, maxima_count * sizeof(float)}, {a.tables, table_count * sizeof(float2)},
            {a.ranges, capacity * sizeof(uint2)}, {a.kv, kv_count * sizeof(std::uint16_t)},
            {a.journal, (8U * Dim + 8U) * sizeof(std::uint16_t)}})
            check(hipMemsetAsync(p, 0xa5, bytes, s), "batch canary");
        check(hipMemsetAsync(a.stats, 0, qsa_flash::kStats * sizeof(std::uint32_t), s), "batch stats");
        a.norm.resize(groups); a.store.resize(groups); a.score.resize(groups); a.bounds.resize(groups);
        unsigned valid = 0;
        std::size_t offset = 0;
        for (unsigned g = 0; g < groups; ++g) {
            auto& n = a.norm[g]; auto& st = a.store[g]; auto& sc = a.score[g];
            for (unsigned r = 0; r < Rows; ++r) {
                const unsigned row = g * Rows + r, position = first + row, complete = (position + 1U) / ratio;
                auto* v = a.values + static_cast<std::size_t>(row) * RowFloats;
                n.values[0][r] = v; sc.views.index_query[r] = v;
                n.values[1][r] = v + 512U; n.values[2][r] = v + 1024U;
                st.index_key[r] = v + 1152U; st.k[r] = v + 1024U; st.v[r] = v + 1280U;
                n.position[r] = st.position[r] = position;
                sc.views.block_scores[r] = a.scores + static_cast<std::size_t>(row) * score_stride;
                sc.views.blocks[r] = complete > topk ? complete : 0U;
                sc.blocks = std::max(sc.blocks, sc.views.blocks[r]);
            }
            n.heads[0] = sc.blocks ? 4U : 0U; n.heads[1] = 2U; n.heads[2] = 1U;
            for (unsigned set = 0; set < 3U; ++set) {
                n.weights[set] = weights; n.dim[set] = Dim; n.stride[set] = set == 1U ? 2U * Dim : Dim;
            }
            n.table = a.tables + static_cast<std::size_t>(g) * Rows * (Rotary / 2U);
            st.journal = a.journal;
            const unsigned count = sc.blocks > valid ? sc.blocks - valid : 0U;
            a.bounds[g] = make_uint2(valid, count); valid = std::max(valid, sc.blocks);
            a.max_new = std::max(a.max_new, count); a.max_blocks = std::max(a.max_blocks, sc.blocks);
            a.max_heads = std::max(a.max_heads, n.heads[0] + 3U);
            sc.maxima_stride = (sc.blocks + 3U) / 4U + 3U;
            sc.maxima = sc.blocks ? a.maxima + offset : nullptr;
            offset += static_cast<std::size_t>(Rows) * sc.maxima_stride;
        }
        check(hipMemcpy(a.dn, a.norm.data(), groups * sizeof(a.norm[0]), hipMemcpyHostToDevice), "batch norm descriptors");
        check(hipMemcpy(a.dc, a.score.data(), groups * sizeof(a.score[0]), hipMemcpyHostToDevice), "batch score descriptors");
        check(hipMemcpy(a.db, a.bounds.data(), groups * sizeof(uint2), hipMemcpyHostToDevice), "batch bounds");
    };
    Arena serial{}, batch{}; setup(serial); setup(batch);
    constexpr unsigned R = qsa_flash::kScoreRows, W = qsa_flash::kScoreWaves;
    for (unsigned g = 0; g < groups; ++g) {
        const auto& n = serial.norm[g]; const auto& st = serial.store[g]; const auto& sc = serial.score[g];
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows>), dim3(Rows), dim3(64), 0, s, n, Rotary, Theta);
        const auto heads = n.heads[0] + 3U;
        if (wave)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<Kind, Rows>), dim3((Rows * heads + 7U) / 8U),
                dim3(256), 0, s, n, Epsilon, Rotary);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<Kind, Rows>), dim3(heads, Rows),
                dim3(256), 0, s, n, Epsilon, Rotary);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kernel<Rows>), dim3(1, Rows), dim3(256), 0, s,
            st, serial.raw, serial.kv, Dim, Dim);
        if (serial.bounds[g].y) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<Kind>), dim3(serial.bounds[g].y), dim3(Dim),
                2U * Dim * sizeof(float), s, serial.raw, serial.bounds[g].x, Dim, ratio, weights, Epsilon, Rotary, Theta, serial.pool);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>), dim3((serial.bounds[g].y + 255U) / 256U),
                dim3(256), 0, s, serial.pool, serial.bounds[g].x, serial.bounds[g].y, serial.ranges);
        }
        if (sc.blocks)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, R, W, true>),
                dim3(((sc.blocks + 31U) / 32U + W - 1U) / W, Rows / R), dim3(W * 32U), 0, s,
                sc.views, serial.pool, sc.blocks, Dim, sc.maxima, sc.maxima_stride, serial.ranges, serial.stats);
    }
    for (unsigned g = 0; g + 1U < groups; ++g) batch.store[g].journal = nullptr;
    check(hipMemcpy(batch.ds, batch.store.data(), groups * sizeof(batch.store[0]), hipMemcpyHostToDevice), "batch store descriptors");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows, true>), dim3(Rows, groups), dim3(64), 0, s,
        qsa_index::NormGroupViews<Rows>{}, Rotary, Theta, batch.dn);
    if (wave)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<Kind, Rows, true>),
            dim3((Rows * batch.max_heads + 7U) / 8U, groups), dim3(256), 0, s,
            qsa_index::NormGroupViews<Rows>{}, Epsilon, Rotary, batch.dn);
    else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<Kind, Rows, true>),
            dim3(batch.max_heads, Rows, groups), dim3(256), 0, s, qsa_index::NormGroupViews<Rows>{}, Epsilon, Rotary, batch.dn);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kernel<Rows, true>), dim3(1, Rows, groups), dim3(256), 0, s,
        QsaRowsStoreViews<Rows>{}, batch.raw, batch.kv, Dim, Dim, batch.ds);
    if (batch.max_new) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<Kind, true>), dim3(batch.max_blocks), dim3(Dim),
            2U * Dim * sizeof(float), s, batch.raw, 0U, Dim, ratio, weights, Epsilon, Rotary, Theta, batch.pool, batch.db, groups);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<true>), dim3((batch.max_blocks + 255U) / 256U),
            dim3(256), 0, s, batch.pool, 0U, batch.max_blocks, batch.ranges, batch.db, groups);
    }
    if (batch.max_blocks)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::score_kernel<Rows, R, W, true, false, true, 1U, true>),
            dim3(((batch.max_blocks + 31U) / 32U + W - 1U) / W, Rows / R, groups), dim3(W * 32U), 0, s,
            QsaRowsScoreViews<Rows>{}, batch.pool, 0U, Dim, nullptr, 0U, batch.ranges, batch.stats, batch.dc);
    check(hipGetLastError(), "stage batch launches"); check(hipStreamSynchronize(s), "stage batch sync");
    const std::string name = "stage batch groups=" + std::to_string(groups) + " wave=" + std::to_string(wave) +
        " edges=" + std::to_string(edges) + " kind=" + std::to_string(static_cast<unsigned>(Kind));
    const auto compare = [&](const void* a, const void* b, std::size_t bytes, const char* stage) {
        std::vector<unsigned char> x(bytes), y(bytes);
        check(hipMemcpy(x.data(), a, bytes, hipMemcpyDeviceToHost), "batch compare serial");
        check(hipMemcpy(y.data(), b, bytes, hipMemcpyDeviceToHost), "batch compare batched");
        expect(x == y, name + " " + stage);
    };
    compare(serial.values, batch.values, input.size() * sizeof(float), "norm and untouched projections");
    compare(serial.tables, batch.tables, table_count * sizeof(float2), "private RoPE tables");
    compare(serial.raw, batch.raw, initial_raw.size() * sizeof(std::uint16_t), "raw index keys");
    compare(serial.kv, batch.kv, kv_count * sizeof(std::uint16_t), "BF16 KV");
    compare(serial.journal, batch.journal, (8U * Dim + 8U) * sizeof(std::uint16_t), "final journal");
    compare(serial.pool, batch.pool, static_cast<std::size_t>(capacity) * Dim * sizeof(float), "pool");
    compare(serial.ranges, batch.ranges, capacity * sizeof(uint2), "ranges");
    compare(serial.scores, batch.scores, score_count * sizeof(float), "scores and masked blocks");
    compare(serial.maxima, batch.maxima, maxima_count * sizeof(float), "private maxima");
    compare(serial.stats, batch.stats, qsa_flash::kStats * sizeof(std::uint32_t), "guard decisions");
    std::vector<std::uint32_t> stats(qsa_flash::kStats);
    check(hipMemcpy(stats.data(), batch.stats, stats.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "batch guard witness");
    expect(!batch.max_blocks || stats[qsa_flash::kFused] + stats[qsa_flash::kSeparate] > 0U, name + " scorer executed");
    std::vector<std::uint16_t> journal(8U * Dim);
    check(hipMemcpy(journal.data(), batch.journal, journal.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost), "batch journal witness");
    bool final_journal = true;
    for (unsigned r = 0; r < 8U; ++r)
        for (unsigned d = 0; d < Dim; ++d) {
            const auto x = input[static_cast<std::size_t>(rows - 8U + r) * RowFloats + 1152U + d];
            const auto u = ubits(x);
            const auto expected = static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
            final_journal = final_journal && journal[r * Dim + d] == expected;
        }
    expect(final_journal, name + " journal contains final eight rows");
    // Exercise the private maxima through both immediate and deferred selection,
    // as the engine does with bit 4 clear/set; compare complete selected capacities.
    const unsigned selected_stride = (topk + 1U) * ratio;
    const auto selected_count = static_cast<std::size_t>(rows) * selected_stride;
    auto* selected_serial = dalloc<std::uint32_t>(selected_count);
    auto* selected_batch = dalloc<std::uint32_t>(selected_count);
    check(hipMemsetAsync(selected_serial, 0xa5, selected_count * sizeof(std::uint32_t), s), "serial selected canary");
    check(hipMemsetAsync(selected_batch, 0xa5, selected_count * sizeof(std::uint32_t), s), "batch selected canary");
    std::vector<qsa_flash::SelectRow> selection(rows);
    for (unsigned g = 0; g < groups; ++g) {
        QsaRowsSelectViews<Rows> v{};
        for (unsigned r = 0; r < Rows; ++r) {
            const unsigned row = g * Rows + r;
            v.block_scores[r] = serial.score[g].views.block_scores[r];
            v.selected[r] = selected_serial + static_cast<std::size_t>(row) * selected_stride;
            v.visible[r] = first + row + 1U;
            const auto& sc = batch.score[g];
            selection[row] = qsa_flash::SelectRow{sc.views.block_scores[r],
                selected_batch + static_cast<std::size_t>(row) * selected_stride,
                sc.maxima ? sc.maxima + static_cast<std::size_t>(r) * sc.maxima_stride : nullptr, v.visible[r], 0U};
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::select_kernel<Rows>), dim3(Rows), dim3(kSelectThreads), 0, s,
            v, serial.score[g].maxima, serial.score[g].maxima_stride, ratio, topk);
    }
    auto* ds = dalloc<qsa_flash::SelectRow>(rows);
    check(hipMemcpy(ds, selection.data(), rows * sizeof(selection[0]), hipMemcpyHostToDevice), "batch selected descriptors");
    hipLaunchKernelGGL(qsa_flash::select_batch_kernel, dim3(rows), dim3(kSelectThreads), 0, s, ds, ratio, topk);
    check(hipGetLastError(), "stage batch selection launch"); check(hipStreamSynchronize(s), "stage batch selection sync");
    compare(selected_serial, selected_batch, selected_count * sizeof(std::uint32_t), "immediate/deferred selected indices");
    compare(serial.scores, batch.scores, score_count * sizeof(float), "post-selection scores");
    check(hipFree(ds), "batch selection free");
    check(hipFree(selected_serial), "serial selected free"); check(hipFree(selected_batch), "batch selected free");
    std::printf("  B %s: byte comparisons complete\n", name.c_str());
    const auto release = [&](Arena& a) {
        for (void* p : {static_cast<void*>(a.values), static_cast<void*>(a.pool), static_cast<void*>(a.scores),
            static_cast<void*>(a.maxima), static_cast<void*>(a.tables), static_cast<void*>(a.ranges),
            static_cast<void*>(a.raw), static_cast<void*>(a.kv), static_cast<void*>(a.journal), static_cast<void*>(a.stats),
            static_cast<void*>(a.dn), static_cast<void*>(a.ds), static_cast<void*>(a.dc), static_cast<void*>(a.db)}) check(hipFree(p), "batch free");
    };
    release(serial); release(batch); check(hipFree(weights), "batch weights free");
}
static void run_stage_batch_cases(hipStream_t s) {
    for (unsigned groups : {1U, 2U, 7U, 128U})
        for (bool wave : {false, true})
            for (bool edges : {false, true}) {
                run_stage_batch_case<SmallWeightKind::f32>(groups, wave, edges, s);
                run_stage_batch_case<SmallWeightKind::f16>(groups, wave, edges, s);
                run_stage_batch_case<SmallWeightKind::bf16>(groups, wave, edges, s);
            }
}

int main(int argc, char** argv) {
    const bool stage_only = argc == 2 && std::strcmp(argv[1], "--stage-batch-only") == 0;
    if (argc != 1 && !stage_only) { std::fprintf(stderr, "usage: %s [--stage-batch-only]\n", argv[0]); return 2; }
    hipStream_t s; check(hipStreamCreate(&s), "stream");
    // qsa_flash_prod.inc also exposes this bench-only kernel; retain its symbol
    // when this fixture is built with the engine's unused-function diagnostics.
    static_cast<void>(&qsa_fast::kv_to_f16_kernel);
    run_stage_batch_cases(s);
    if (stage_only) {
        check(hipStreamDestroy(s), "stream destroy");
        std::printf("stage batch fixtures: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
        return failures ? 1 : 0;
    }
    std::map<std::string, std::uint64_t> paths;
    Rng rng{0x2545F4914F6CDD1DULL};
    const auto normal_q = [&](unsigned rows) { std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kDim); for (auto& x : q) x = bf(rng.normal()); return q; };
    const auto normal_k = [&](unsigned blocks) { std::vector<float> k(static_cast<std::size_t>(blocks) * kDim); for (auto& x : k) x = bf(rng.normal() * 0.3F); return k; };
    // ----- S + G: scorer cases at 64 rows, plus 16 / 32 rows
    std::vector<ScoreCase> cases;
    const std::uint32_t B = 1000;   // 32 tiles, last one partial
    cases.push_back({"normal", normal_q(64), normal_k(B)});
    {   // guard boundaries: per row r a scale 2^a on one query value, per tile t a scale 2^b on one key value
        ScoreCase c{"guard-boundaries", normal_q(64), normal_k(B)};
        const int qa[] = {60, 61, 62, 63, -60, -61, -62, -63, 0, 100, -100, 125, -126, -133, 20, -20};
        const int kb[] = {64, 63, 62, 61, -60, -59, -58, -57, 0, 25, -20, 0, 6, 13, 105, -100};
        for (unsigned r = 0; r < 64; ++r) c.q[r * kHeads * kDim + (r * 7U) % (kHeads * kDim)] = bf(std::ldexp(1.0F + (r & 1U) * 0.5F, qa[r % 16U]));
        for (std::uint32_t b = 0; b < B; b += 32) c.keys[b * kDim + (b / 32U) % kDim] = bf(std::ldexp(1.25F, kb[(b / 32U) % 16U]));
        cases.push_back(c);
    }
    {   // BF16 minimum subnormal / minimum normal on both sides, product underflow (exact and inexact) and overflow
        ScoreCase c{"subnormal-underflow-overflow", normal_q(64), normal_k(B)};
        const std::uint32_t qbits[] = {0x00010000U, 0x00800000U, 0x80010000U, 0x1d800000U, 0x1dc00000U, 0x5f800000U, 0x5fc00000U, 0xdf800000U};
        for (unsigned r = 0; r < 64; ++r) c.q[r * kHeads * kDim + (r * 13U) % (kHeads * kDim)] = bits(qbits[r % 8U]);
        const std::uint32_t kbits[] = {0x00010000U, 0x00800000U, 0x1d800000U, 0x1da00000U, 0x5f800000U, 0x5f400000U, 0x3f800000U, 0x80800000U};
        for (std::uint32_t b = 0; b < B; ++b) if (b % 3U == 0U) c.keys[b * kDim + (b * 5U) % kDim] = bits(kbits[b % 8U]);
        cases.push_back(c);
    }
    {   // cancellation and signed zeros: q_l k_l == -(q_{l+64} k_{l+64}); rows of -0 / +0 only; keys of zeros
        ScoreCase c{"cancellation-signed-zero", normal_q(64), normal_k(B)};
        for (unsigned r = 0; r < 64; ++r)
            for (unsigned h = 0; h < kHeads; ++h)
                for (unsigned l = 0; l < 64; ++l) {
                    auto& a = c.q[r * kHeads * kDim + h * kDim + l]; auto& z = c.q[r * kHeads * kDim + h * kDim + l + 64];
                    if (r % 4U == 0U) { a = (l & 1U) ? -0.0F : 0.0F; z = (l & 2U) ? -0.0F : 0.0F; }
                    else if (r % 4U == 1U) z = bf(-a);
                }
        for (std::uint32_t b = 0; b < B; ++b)
            for (unsigned l = 0; l < 64; ++l) {
                if (b % 7U == 0U) { c.keys[b * kDim + l] = (l & 1U) ? -0.0F : 0.0F; c.keys[b * kDim + l + 64] = (l & 4U) ? -0.0F : 0.0F; }
                else if (b % 7U == 1U) c.keys[b * kDim + l + 64] = c.keys[b * kDim + l];
            }
        cases.push_back(c);
    }
    {   // pair-sum overflow inside the guard: products just below 2^126 that add up past FLT_MAX
        ScoreCase c{"pair-sum-overflow", normal_q(64), normal_k(B)};
        for (unsigned r = 0; r < 64; r += 2)
            for (unsigned l = 0; l < 32; ++l) { c.q[r * kHeads * kDim + l] = bf(std::ldexp(1.9F, 62)); c.q[r * kHeads * kDim + l + 64] = bf(std::ldexp(1.9F, 62)); }
        for (std::uint32_t b = 0; b < B; b += 2)
            for (unsigned l = 0; l < 32; ++l) { c.keys[b * kDim + l] = bf(std::ldexp(1.9F, 62)); c.keys[b * kDim + l + 64] = bf(std::ldexp(1.9F, 62)); }
        cases.push_back(c);
    }
    {   // non-finite: inf / NaN (several payloads, signalling) queries against small keys; zero rows against non-finite keys
        ScoreCase c{"nonfinite", normal_q(64), normal_k(B)};
        const std::uint32_t nf[] = {0x7f800000U, 0xff800000U, 0x7fc00000U, 0xffc00000U, 0x7fc10000U, 0x7fa00000U, 0xffa10000U, 0x7f810000U};
        for (unsigned r = 0; r < 64; ++r) {
            if (r % 4U == 3U) { std::fill_n(c.q.begin() + r * kHeads * kDim, kHeads * kDim, 0.0F); continue; }
            c.q[r * kHeads * kDim + (r * 11U) % (kHeads * kDim)] = bits(nf[r % 8U]);
        }
        for (std::uint32_t b = 0; b < B; ++b) {
            if (b < 64U) for (unsigned d = 0; d < kDim; ++d) c.keys[b * kDim + d] = bf(std::ldexp(1.0F, -4 - static_cast<int>(d % 3U)));
            if (b % 97U == 5U) c.keys[b * kDim + b % kDim] = bits(nf[b % 8U]);
        }
        cases.push_back(c);
    }
    for (const auto& c : cases) run_score_case<64>(c, B, s, paths);
    {   // 16 / 32-row groups against block_score_pooled_rows_kernel (the production scorer of those groups)
        ScoreCase c16{"normal-16", normal_q(16), normal_k(777)}; run_score_case<16>(c16, 777, s, paths);
        ScoreCase c32{"normal-32", normal_q(32), normal_k(1500)}; run_score_case<32>(c32, 1500, s, paths);
        ScoreCase g16{"guard-16", cases[1].q, cases[1].keys}; g16.q.resize(16 * kHeads * kDim); run_score_case<16>(g16, B, s, paths);
        ScoreCase n32{"nonfinite-32", cases[5].q, cases[5].keys}; n32.q.resize(32 * kHeads * kDim); run_score_case<32>(n32, B, s, paths);
    }
    // ----- N: norm / RoPE cases (production norm_rope_group_kernel vs the wave form), every output byte
    for (const std::uint64_t pos : {0ULL, 2051ULL, 131000ULL, 262080ULL}) { run_norm_case<64>("normal", 0U, pos, s, paths); run_norm_case<64>("edges", 1U, pos, s, paths); }
    run_norm_case<16>("normal-16", 0U, 777U, s, paths); run_norm_case<32>("edges-32", 1U, 40000U, s, paths);
    for (const unsigned dim : {128U, 256U, 64U, 32U}) {   // direct squared sums (BF16 output rounding cannot hide a difference)
        const unsigned n = 20000U;
        std::vector<float> x(static_cast<std::size_t>(n) * dim);
        for (std::size_t i = 0; i < x.size(); ++i) { float v = rng.normal() * std::ldexp(1.0F, static_cast<int>(i % 7U) * 9 - 30); if (i % 101U == 0U) v = -0.0F; x[i] = v; }
        auto* dx = dalloc<float>(x.size()); auto* da = dalloc<float>(n); auto* db = dalloc<float>(n);
        check(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "h2d");
        hipLaunchKernelGGL(square_sum_lds_kernel, dim3(n), dim3(256), 0, s, dx, dim, da);
        hipLaunchKernelGGL(square_sum_wave_kernel, dim3(n), dim3(32), 0, s, dx, dim, db);
        check(hipGetLastError(), "launch"); check(hipStreamSynchronize(s), "sync");
        std::vector<float> a2(n), b2(n); check(hipMemcpy(a2.data(), da, n * 4, hipMemcpyDeviceToHost), "d2h"); check(hipMemcpy(b2.data(), db, n * 4, hipMemcpyDeviceToHost), "d2h");
        std::size_t d2 = 0; for (unsigned i = 0; i < n; ++i) d2 += ubits(a2[i]) != ubits(b2[i]);
        std::printf("  N square sums dim %3u: %zu of %u differ\n", dim, d2, n);
        expect(d2 == 0, "square sums dim " + std::to_string(dim));
        check(hipFree(dx), "free"); check(hipFree(da), "free"); check(hipFree(db), "free");
    }
    // ----- P: indexer projection (rows_kernel vs index_rows_kernel), every output
    run_proj_case("normal", 0U, s, paths); run_proj_case("edges", 1U, s, paths);
    // ----- A: attend share candidate vs production attend (bench-only candidate)
    run_attend_case("dense-1to4-chunks", 0U, s, paths); run_attend_case("masked-blocks", 1U, s, paths);
    run_attend_case("token0-multiplicity", 2U, s, paths); run_attend_case("sparse-long", 3U, s, paths);
    // ----- L: selector cases (64 rows each; complete blocks = visible / 4; scores as given)
    const auto mk = [&](const std::string& name, auto&& gen) {
        SelectCase c{name, std::vector<std::vector<float>>(64), std::vector<std::uint32_t>(64)};
        for (unsigned r = 0; r < 64; ++r) { std::uint32_t visible = 0; c.rows[r] = gen(r, visible); c.visible[r] = visible; }
        return c;
    };
    std::vector<SelectCase> sel;
    sel.push_back(mk("random", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 513U + 997U * r; v = 4U * blocks + r % 4U;
        std::vector<float> x(blocks); for (auto& y : x) y = std::fabs(rng.normal()); return x; }));
    sel.push_back(mk("ties-coarse", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 600U + 311U * r; v = 4U * blocks + (r + 1U) % 4U;
        std::vector<float> x(blocks); for (auto& y : x) y = static_cast<float>(rng.next() % (2U + r % 7U)) * 0.5F; return x; }));
    sel.push_back(mk("signed-zero-negative", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 2000U + 37U * r; v = 4U * blocks + (r + 2U) % 4U;
        std::vector<float> x(blocks); for (auto& y : x) { const auto k = rng.next() % 6U; y = k == 0 ? 0.0F : k == 1 ? -0.0F : k == 2 ? -std::fabs(rng.normal()) : std::fabs(rng.normal()) * 1e-3F; } return x; }));
    sel.push_back(mk("inf-nan-ninf", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 1024U + 515U * r; v = 4U * blocks + (r + 3U) % 4U;
        const std::uint32_t sp[] = {0x7f800000U, 0x7fc00000U, 0xff800000U, 0xffc00000U, 0x7fa00000U};
        std::vector<float> x(blocks); for (auto& y : x) { const auto k = rng.next() % 16U; y = k < 5U ? bits(sp[k]) : std::fabs(rng.normal()); } return x; }));
    sel.push_back(mk("eligible-0-511-512-513", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 3000U + r; v = 4U * blocks + r % 4U;
        const std::uint32_t eligible[] = {0U, 511U, 512U, 513U, 1U, 1024U, 2047U, 3000U}; const auto e = eligible[r % 8U];
        std::vector<float> x(blocks, -INFINITY);
        for (std::uint32_t i = 0; i < blocks; ++i) if (i % 2U == 1U) x[i] = std::numeric_limits<float>::quiet_NaN();
        for (std::uint32_t i = 0; i < e && i < blocks; ++i) x[(i * 7919U) % blocks] = std::fabs(rng.normal());
        return x; }));
    sel.push_back(mk("complete-511-512-513", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 511U + r % 3U + (r / 3U) % 2U * 4U; v = 4U * blocks + r % 4U;
        std::vector<float> x(blocks); for (auto& y : x) y = std::fabs(rng.normal()); return x; }));
    sel.push_back(mk("candidates-1536-1537", [&](unsigned r, std::uint32_t& v) {
        // all scores equal: every group is a candidate; 1536 groups fit (6144 blocks), 1537 do not; partial last groups
        const std::uint32_t blocks = r % 4U == 0U ? 6144U : r % 4U == 1U ? 6145U : r % 4U == 2U ? 6141U : 6148U; v = 4U * blocks + r % 4U;
        return std::vector<float>(blocks, 0.75F); }));
    sel.push_back(mk("zero-scores-long", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 4096U + 3000U * (r % 8U); v = 4U * blocks + 3U;
        std::vector<float> x(blocks, 0.0F); for (std::uint32_t i = 0; i < blocks; i += 1U + r) x[i] = 0.5F * static_cast<float>(i % 5U); return x; }));
    sel.push_back(mk("long-65536", [&](unsigned r, std::uint32_t& v) { const std::uint32_t blocks = 65536U - (r % 4U) * 3U; v = 4U * blocks + r % 4U;
        std::vector<float> x(blocks); for (std::uint32_t i = 0; i < blocks; ++i) x[i] = std::fabs(rng.normal()) + (i > blocks - 3000U ? 1.0F : 0.0F); return x; }));
    for (const auto& c : sel) run_select_case(c, s, paths);
    std::printf("paths:");
    for (const auto& [k, v] : paths) std::printf(" %s=%llu", k.c_str(), static_cast<unsigned long long>(v));
    std::printf("\n");
    for (const char* need : {"fused", "separate", "select_candidates", "select_few_maxima", "select_capacity", "select_dense", "norm_wave", "proj_rows16", "attend_share"})
        expect(paths[need] > 0, std::string("path never taken: ") + need);
    std::printf(failures ? "FIXTURES FAIL (%d)\n" : "FIXTURES PASS\n", failures);
    return failures ? 1 : 0;
}

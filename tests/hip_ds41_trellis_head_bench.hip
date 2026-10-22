// Trellis LM head micro-benchmark and row-invariance check (not a ctest; built only when STRIX_DS41_TRELLIS_LIB is set).
// Head of the trellis pack (dense.head.safetensors: head.trellis I16 [K/16][N/16][16 * bits], head.suh F16 [K],
// head.svh F16 [N], head.mul1) through the strix_trellis C API, as the split-vocabulary decode uses it: rank r keeps
// output rows [r * N / 2, (r + 1) * N / 2) as its own pre-sliced transport (trellis tiles [:, r * N / 32 .., :],
// svh[r * N / 2 ..], suh whole). Checks, all bitwise:
//   rows    every row of a B-row GEMV (B = 1..8, rows in order and reversed) == the same input alone at B = 1;
//   repeat  the same call twice;
//   slice   each rank slice == the matching rows of the whole-vocabulary transport (same inputs).
// Timing (median of hip events, the weights evicted between reps by a 256 MB write): the rank slice and the whole
// head at B = 1..8, and today's head kernel (enqueue_q8_0_a8 over a rank's 64,640 rows, Q8_0 and U6 interleaved rows
// of the same shape, A8 of the same inputs).
// usage: strix-hip-ds41-trellis-head-bench <dense.head.safetensors> [reps=20] [sets=4] [baseline=1]

#include "strix/ds41_packed.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix_trellis.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
void check(strix_status s, const char* what) {
    if (s != STRIX_SUCCESS) { std::fprintf(stderr, "%s: %s\n", what, strix_status_string(s)); std::exit(2); }
}

// Minimal safetensors header reader: name -> {dtype, shape, begin, end} (offsets relative to the data section).
struct TensorInfo { std::string dtype; std::vector<std::uint64_t> shape; std::uint64_t begin = 0, end = 0; };
struct Safetensors {
    std::vector<std::uint8_t> bytes;
    std::uint64_t data_start = 0;
    std::string header;
    TensorInfo find(const std::string& name) const {
        const auto key = "\"" + name + "\"";
        const auto at = header.find(key);
        if (at == std::string::npos) { std::fprintf(stderr, "no tensor %s\n", name.c_str()); std::exit(2); }
        const auto close = header.find('}', at);
        const std::string body = header.substr(at, close - at);
        TensorInfo t;
        const auto d = body.find("\"dtype\"");
        const auto q1 = body.find('"', body.find(':', d) + 1), q2 = body.find('"', q1 + 1);
        t.dtype = body.substr(q1 + 1, q2 - q1 - 1);
        auto nums = [&](const char* field) {
            std::vector<std::uint64_t> v;
            const auto f = body.find(field);
            const auto b = body.find('[', f), e = body.find(']', b);
            std::string s = body.substr(b + 1, e - b - 1);
            std::size_t p = 0;
            while (p < s.size()) {
                while (p < s.size() && (s[p] < '0' || s[p] > '9')) ++p;
                if (p >= s.size()) break;
                std::uint64_t x = 0;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9') x = x * 10 + static_cast<std::uint64_t>(s[p++] - '0');
                v.push_back(x);
            }
            return v;
        };
        t.shape = nums("\"shape\"");
        const auto o = nums("\"data_offsets\"");
        t.begin = o.at(0);
        t.end = o.at(1);
        return t;
    }
    const std::uint8_t* data(const TensorInfo& t) const { return bytes.data() + data_start + t.begin; }
};
Safetensors load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    Safetensors s;
    f.seekg(0, std::ios::end);
    s.bytes.resize(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(s.bytes.data()), static_cast<std::streamsize>(s.bytes.size()));
    std::uint64_t n = 0;
    std::memcpy(&n, s.bytes.data(), 8);
    s.header.assign(reinterpret_cast<const char*>(s.bytes.data()) + 8, n);
    s.data_start = 8 + n;
    return s;
}

// Host blob + metadata of one row range [first, first + count) of the head (count, first multiples of 128).
struct Transport {
    std::vector<std::uint8_t> blob;
    strix_exl3_metadata_v1 meta{};
    strix_exl3_transport_desc desc() const {
        strix_exl3_transport_desc d{};
        d.struct_size = sizeof(d);
        d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.format = STRIX_TRANSPORT_EXL3_CANONICAL;
        d.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        d.data = blob.data();
        d.data_bytes = blob.size();
        d.metadata = &meta;
        d.metadata_bytes = sizeof(meta);
        d.rows = meta.rows;
        d.columns = meta.columns;
        return d;
    }
};
Transport slice(const Safetensors& st, const std::string& prefix, std::uint64_t first, std::uint64_t count) {
    const auto tr = st.find(prefix + ".trellis"), su = st.find(prefix + ".suh"), sv = st.find(prefix + ".svh"), mk = st.find(prefix + ".mul1");
    const std::uint64_t kt = tr.shape.at(0), nt = tr.shape.at(1), words = tr.shape.at(2);
    const std::uint64_t columns = kt * 16, rows = nt * 16;
    if (first % 128 || count % 128 || first + count > rows || words % 16) { std::fprintf(stderr, "bad slice\n"); std::exit(2); }
    Transport t;
    auto& m = t.meta;
    m.struct_size = sizeof(m);
    m.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    m.rows = count;
    m.columns = columns;
    m.bits_x2 = static_cast<std::uint32_t>(2 * (words / 16));
    m.codebook = STRIX_CODEBOOK_MUL1;
    std::memcpy(&m.marker, st.data(mk), 4);
    // trellis tiles: [kt][nt][words] I16 -> [kt][count / 16][words]
    const std::uint64_t tile_bytes = words * 2, take = count / 16;
    m.trellis_offset = 0;
    m.trellis_bytes = kt * take * tile_bytes;
    t.blob.resize(m.trellis_bytes);
    for (std::uint64_t k = 0; k < kt; ++k)
        std::memcpy(t.blob.data() + k * take * tile_bytes, st.data(tr) + (k * nt + first / 16) * tile_bytes, take * tile_bytes);
    m.suh_offset = t.blob.size();
    m.suh_bytes = columns * 2;
    t.blob.insert(t.blob.end(), st.data(su), st.data(su) + columns * 2);
    m.svh_offset = t.blob.size();
    m.svh_bytes = count * 2;
    t.blob.insert(t.blob.end(), st.data(sv) + first * 2, st.data(sv) + (first + count) * 2);
    (void)rows;
    return t;
}

struct Linear {
    strix_context ctx{};
    strix_weights w{};
    std::uint64_t rows = 0, columns = 0;
    void* ws = nullptr;
    std::uint64_t ws_bytes = 0;
};
strix_stream sstream(hipStream_t s) {
    strix_stream x{};
    x.struct_size = sizeof(x);
    x.abi_version = STRIX_TRELLIS_ABI_VERSION;
    x.hip_stream = reinterpret_cast<void*>(s);
    return x;
}
strix_matmul_desc desc_for(const Linear& L, const float* x, float* y, std::uint32_t B) {
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = L.w;
    d.input.data = const_cast<float*>(x);
    d.input.buffer_bytes = std::uint64_t{B} * L.columns * 4;
    d.input.rank = 2;
    d.input.shape[0] = B;
    d.input.shape[1] = L.columns;
    d.input.strides[0] = L.columns;
    d.input.strides[1] = 1;
    d.output.data = y;
    d.output.buffer_bytes = std::uint64_t{B} * L.rows * 4;
    d.output.rank = 2;
    d.output.shape[0] = B;
    d.output.shape[1] = L.rows;
    d.output.strides[0] = L.rows;
    d.output.strides[1] = 1;
    return d;
}
Linear prepare(strix_context ctx, const Transport& t, hipStream_t s, float* qx, float* qy) {
    Linear L;
    L.ctx = ctx;
    L.rows = t.meta.rows;
    L.columns = t.meta.columns;
    const auto d = t.desc();
    const auto ss = sstream(s);
    check(strix_weights_prepare(ctx, &d, &ss, &L.w), "weights_prepare");
    check(hipStreamSynchronize(s), "prepare sync");
    for (std::uint32_t B = 1; B <= 8; ++B) {
        auto md = desc_for(L, qx, qy, B);
        std::uint64_t b = 0;
        check(strix_matmul_workspace_bytes(ctx, &md, &b), "workspace_bytes");
        L.ws_bytes = std::max(L.ws_bytes, b);
    }
    check(hipMalloc(&L.ws, std::max<std::uint64_t>(L.ws_bytes, 256)), "workspace");
    return L;
}
void gemv(const Linear& L, const float* x, float* y, std::uint32_t B, hipStream_t s) {
    auto d = desc_for(L, x, y, B);
    d.workspace = L.ws;
    d.workspace_bytes = L.ws_bytes;
    const auto ss = sstream(s);
    check(strix_gemv(L.ctx, &d, &ss), "gemv");
}

std::vector<float> down(const float* p, std::size_t n, hipStream_t s) {
    std::vector<float> h(n);
    check(hipStreamSynchronize(s), "sync");
    check(hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost), "download");
    return h;
}
bool same(const float* a, const float* b, std::size_t n, std::size_t* first_diff = nullptr) {
    for (std::size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, 4) != 0) {
            if (first_diff) *first_diff = i;
            return false;
        }
    return true;
}

double median_ms(int reps, hipStream_t s, void* flush, std::size_t flush_bytes, const std::function<void()>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 2; ++i) launch();
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        check(hipMemsetAsync(flush, i & 0xff, flush_bytes, s), "flush");
        check(hipEventRecord(a, s), "record");
        launch();
        check(hipEventRecord(b, s), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <dense.head.safetensors> [reps] [sets] [baseline]\n", argv[0]); return 2; }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int sets = argc > 3 ? std::atoi(argv[3]) : 4;
    const bool baseline = argc > 4 ? std::atoi(argv[4]) != 0 : true;
    if (strix_get_abi_version() != STRIX_TRELLIS_ABI_VERSION) { std::fprintf(stderr, "ABI mismatch\n"); return 2; }
    const auto st = load(argv[1]);
    const auto tr = st.find("head.trellis");
    const std::uint64_t K = tr.shape[0] * 16, N = tr.shape[1] * 16, bits = tr.shape[2] / 16, half = N / 2;
    std::printf("head: K %llu N %llu bits %llu, rank slice %llu rows\n", (unsigned long long)K, (unsigned long long)N,
                (unsigned long long)bits, (unsigned long long)half);
    hipStream_t s;
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    strix_context_desc cd{};
    cd.struct_size = sizeof(cd);
    cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
    cd.device_ordinal = 0;
    strix_context ctx{};
    check(strix_context_create(&cd, &ctx), "context_create");
    // inputs: sets x 8 rows, RMS-normalised random rows with a few large channels (as after the final norm)
    std::mt19937 rng(20261021);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> xs(static_cast<std::size_t>(sets) * 8 * K);
    for (auto& v : xs) v = nd(rng);
    for (int i = 0; i < sets * 8; ++i)
        for (int j = 0; j < 8; ++j) xs[static_cast<std::size_t>(i) * K + (rng() % K)] *= 12.0F;
    float *dx, *dxr, *dy, *dy1;
    check(hipMalloc(&dx, xs.size() * 4), "x");
    check(hipMalloc(&dxr, 8 * K * 4), "xr");
    check(hipMalloc(&dy, 8 * N * 4), "y");
    check(hipMalloc(&dy1, 8 * N * 4), "y1");
    check(hipMemcpy(dx, xs.data(), xs.size() * 4, hipMemcpyHostToDevice), "x upload");
    std::vector<Linear> L;
    {
        const auto t0 = slice(st, "head", 0, half), t1 = slice(st, "head", half, half), tf = slice(st, "head", 0, N);
        L.push_back(prepare(ctx, t0, s, dx, dy));
        L.push_back(prepare(ctx, t1, s, dx, dy));
        L.push_back(prepare(ctx, tf, s, dx, dy));
        std::printf("transport bytes: slice %zu, whole %zu; workspace (max over B) slice %llu, whole %llu\n", t0.blob.size(),
                    tf.blob.size(), (unsigned long long)L[0].ws_bytes, (unsigned long long)L[2].ws_bytes);
    }
    const char* names[3] = {"rank0", "rank1", "whole"};


    // --- exactness
    std::size_t bad = 0, compared = 0;
    for (int set = 0; set < sets; ++set) {
        const float* x = dx + static_cast<std::size_t>(set) * 8 * K;
        std::vector<std::vector<float>> alone(3);
        for (int l = 0; l < 3; ++l) {
            const auto n = L[l].rows;
            alone[l].resize(8 * n);
            for (int r = 0; r < 8; ++r) {
                gemv(L[l], x + r * K, dy1, 1, s);
                const auto y = down(dy1, n, s);
                std::copy(y.begin(), y.end(), alone[l].begin() + static_cast<std::ptrdiff_t>(r * n));
            }
            for (std::uint32_t B = 1; B <= 8; ++B) {
                for (int order = 0; order < 2; ++order) {
                    const float* in = x;
                    if (order == 1) {  // rows reversed: input row b = x row B - 1 - b
                        for (std::uint32_t b = 0; b < B; ++b)
                            check(hipMemcpyAsync(dxr + b * K, x + (B - 1 - b) * K, K * 4, hipMemcpyDeviceToDevice, s), "reverse");
                        in = dxr;
                    }
                    gemv(L[l], in, dy, B, s);
                    const auto y = down(dy, B * n, s);
                    for (std::uint32_t b = 0; b < B; ++b) {
                        const std::uint32_t src = order == 0 ? b : B - 1 - b;
                        std::size_t at = 0;
                        ++compared;
                        if (!same(y.data() + b * n, alone[l].data() + src * n, n, &at)) {
                            ++bad;
                            if (bad <= 20)
                                std::printf("ROWS DIFFER %s set %d B %u order %d row %u: first output %zu %.9g vs %.9g\n", names[l], set, B,
                                            order, b, at, y[b * n + at], alone[l][src * n + at]);
                        }
                    }
                }
            }
            for (std::uint32_t B = 1; B < 8; ++B) {  // padded: rows B..7 NaN, a B = 8 call; rows < B must equal alone
                check(hipMemcpyAsync(dxr, x, std::size_t{B} * K * 4, hipMemcpyDeviceToDevice, s), "pad copy");
                check(hipMemsetAsync(dxr + B * K, 0xff, (8 - B) * K * 4, s), "pad nan");
                gemv(L[l], dxr, dy, 8, s);
                const auto y = down(dy, 8 * n, s);
                for (std::uint32_t b = 0; b < B; ++b) {
                    std::size_t at = 0;
                    ++compared;
                    if (!same(y.data() + b * n, alone[l].data() + b * n, n, &at)) {
                        ++bad;
                        if (bad <= 30) std::printf("PADDED DIFFERS %s set %d B %u row %u: output %zu\n", names[l], set, B, b, at);
                    }
                }
            }
            gemv(L[l], x, dy, 8, s);  // repeat
            const auto y2 = down(dy, 8 * n, s);
            gemv(L[l], x, dy, 8, s);
            const auto y3 = down(dy, 8 * n, s);
            ++compared;
            if (!same(y2.data(), y3.data(), y2.size())) { ++bad; std::printf("REPEAT DIFFERS %s set %d\n", names[l], set); }
        }
        for (int r = 0; r < 2; ++r)  // slices vs the whole head, every row alone
            for (int row = 0; row < 8; ++row) {
                std::size_t at = 0;
                ++compared;
                if (!same(alone[r].data() + row * half, alone[2].data() + row * N + r * half, half, &at)) {
                    ++bad;
                    if (bad <= 40)
                        std::printf("SLICE DIFFERS rank %d set %d row %d: output %zu %.9g vs %.9g\n", r, set, row, at,
                                    alone[r][row * half + at], alone[2][row * N + r * half + at]);
                }
            }
        // finite check + a magnitude print
        double mx = 0;
        for (float v : alone[2]) mx = std::max(mx, static_cast<double>(std::fabs(v)));
        std::printf("set %d: max |logit| %.4g\n", set, mx);
    }
    std::printf("EXACTNESS %s: %zu comparisons, %zu differ\n", bad == 0 ? "PASS" : "FAIL", compared, bad);

    // --- timing
    const std::size_t flush_bytes = std::size_t{256} << 20;
    void* flush = nullptr;
    check(hipMalloc(&flush, flush_bytes), "flush");
    const double slice_gb = static_cast<double>(half * K * bits / 8 + half * 2) / 1e9;
    for (std::uint32_t B = 1; B <= 8; ++B) {
        const double t0 = median_ms(reps, s, flush, flush_bytes, [&] { gemv(L[0], dx, dy, B, s); });
        const double t1 = median_ms(reps, s, flush, flush_bytes, [&] { gemv(L[1], dx, dy, B, s); });
        const double tw = median_ms(reps, s, flush, flush_bytes, [&] { gemv(L[2], dx, dy, B, s); });
        std::printf("TRELLIS B %u: rank0 %.1f us (%.1f GB/s)  rank1 %.1f us (%.1f GB/s)  whole %.1f us (%.1f GB/s)\n", B, t0 * 1e3,
                    slice_gb / (t0 * 1e-3), t1 * 1e3, slice_gb / (t1 * 1e-3), tw * 1e3, 2 * slice_gb / (tw * 1e-3));
    }
    if (baseline) {  // today's kernel over a rank slice: Q8_0 and U6 rows of random codes, A8 of the same inputs
        using namespace strix::ds41;
        const std::size_t groups = K / 32, q8_row = groups * 34;
        std::vector<std::uint8_t> q8(half * q8_row);
        for (std::size_t r = 0; r < half; ++r)
            for (std::size_t g = 0; g < groups; ++g) {
                std::uint8_t* b = q8.data() + r * q8_row + g * 34;
                const auto d = static_cast<_Float16>(0.001F + 0.0001F * static_cast<float>(rng() % 100));
                std::memcpy(b, &d, 2);
                for (int j = 0; j < 32; ++j) b[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(rng() % 64) - 32));
            }
        PackedChoice u6;
        u6.kind = PackedKind::u6;
        const auto packed = pack_rows(q8, K, u6, PackedLayout::interleaved);
        std::uint8_t *dq8, *du6;
        check(hipMalloc(&dq8, q8.size()), "q8");
        check(hipMalloc(&du6, packed.size()), "u6");
        check(hipMemcpy(dq8, q8.data(), q8.size(), hipMemcpyHostToDevice), "q8 up");
        check(hipMemcpy(du6, packed.data(), packed.size(), hipMemcpyHostToDevice), "u6 up");
        std::int8_t* codes;
        float* deltas;
        std::int32_t* sums;
        check(hipMalloc(&codes, 8 * K), "codes");
        check(hipMalloc(&deltas, 8 * groups * 4), "deltas");
        check(hipMalloc(&sums, 8 * groups * 4), "sums");
        enqueue_quantize_a8(dx, static_cast<std::uint32_t>(K), 8, codes, deltas, sums, s);
        for (std::uint32_t B = 1; B <= 8; ++B) {
            double t[2];
            for (int f = 0; f < 2; ++f) {
                Q8A8View v;
                v.weight = f == 0 ? dq8 : du6;
                v.packed = f == 0 ? 0U : 6U;
                v.width = static_cast<std::uint32_t>(K);
                v.first = 0;
                v.count = static_cast<std::uint32_t>(half);
                v.per_group = static_cast<std::uint32_t>(half);
                v.codes = codes;
                v.deltas = deltas;
                v.rows = B;
                v.x_stride = static_cast<std::uint32_t>(K);
                v.out_stride = static_cast<std::uint32_t>(N);
                v.out = dy;
                t[f] = median_ms(reps, s, flush, flush_bytes, [&] {
                    enqueue_quantize_a8(dx, static_cast<std::uint32_t>(K), B, codes, deltas, sums, s);
                    enqueue_q8_0_a8(v, s);
                });
            }
            const double q8_gb = static_cast<double>(q8.size()) / 1e9, u6_gb = static_cast<double>(packed.size()) / 1e9;
            std::printf("TODAY B %u: q8_0 %.1f us (%.1f GB/s)  u6 %.1f us (%.1f GB/s)  [incl. A8 of the rows]\n", B, t[0] * 1e3,
                        q8_gb / (t[0] * 1e-3), t[1] * 1e3, u6_gb / (t[1] * 1e-3));
        }
    }
    for (auto& l : L) {
        (void)strix_weights_destroy(l.w);
        (void)hipFree(l.ws);
    }
    (void)strix_context_destroy(ctx);
    return bad == 0 ? 0 : 1;
}

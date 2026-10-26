// MiniMax DSpark drafter, device tests (gfx1151, on the node GPU):
// - every kernel bitwise == the host reference on synthetic data (rows 1..8 and 32, row invariance: a row's bits do not
//   depend on how many rows share the launch);
// - with STRIX_MM_DSPARK_DIR + STRIX_MM_DSPARK_CASE: the whole drafter on the real weights bitwise == the host fixtures
//   (host_y.f32, host.json written by strix-mm-dspark-tests), the Markov chain split in two vocabulary halves == full;
// - timings (hipEvent, median of 10): block pass, append of 1 / 8 / 32 rows, Markov step full / half vocabulary.
#include "../hip_minimax_dspark.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace strix::minimax::dspark;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
template <class T>
T* dev_copy(const std::vector<T>& v) {
    T* d = nullptr;
    ok(hipMalloc(&d, std::max<std::size_t>(1, v.size()) * sizeof(T)), "malloc");
    ok(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T>
std::vector<T> host_copy(const T* d, std::size_t n) {
    std::vector<T> v(n);
    ok(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return v;
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b, std::size_t* first = nullptr) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            if (first != nullptr) *first = i;
            return false;
        }
    return true;
}
template <class T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}
std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
std::vector<double> json_array(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error("no " + key);
    const auto b = text.find('[', at), e = text.find(']', b);
    std::vector<double> out;
    const char* p = text.c_str() + b + 1;
    while (p < text.c_str() + e) {
        char* stop = nullptr;
        const double d = std::strtod(p, &stop);
        if (stop == p) { ++p; continue; }
        out.push_back(d);
        p = stop;
    }
    return out;
}
double json_number(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error("no " + key);
    return std::strtod(text.c_str() + text.find(':', at) + 1, nullptr);
}
template <class F>
double median_ms(F f, int reps = 10) {
    hipEvent_t a, b;
    ok(hipEventCreate(&a), "event");
    ok(hipEventCreate(&b), "event");
    f();  // warm
    ok(hipDeviceSynchronize(), "sync");
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        ok(hipEventRecord(a, nullptr), "rec");
        f();
        ok(hipEventRecord(b, nullptr), "rec");
        ok(hipEventSynchronize(b), "sync");
        float ms = 0;
        ok(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Synthetic weights of the right shapes, small vocabulary-independent pieces only (kernel tests).
struct Synthetic {
    std::vector<std::uint8_t> codes, scales;
    std::vector<std::uint16_t> bits;
};
Synthetic make(std::uint32_t outputs, std::uint32_t width, std::mt19937& rng) {
    Synthetic s;
    s.codes.resize(static_cast<std::size_t>(outputs) * width);
    s.scales.resize(s.codes.size() / 32U);
    s.bits.resize(s.codes.size());
    for (auto& c : s.codes) {
        c = static_cast<std::uint8_t>(rng());
        if ((c & 0x7fU) == 0x7fU) c = 0;  // no NaN codes
    }
    for (auto& c : s.scales) c = static_cast<std::uint8_t>(112 + rng() % 24);
    for (auto& b : s.bits) b = bf16_bits(std::normal_distribution<float>(0.0F, 0.05F)(rng));
    return s;
}

void test_kernels(DeviceDrafter& dd) {
    dd.set_kernels(false, 0);  // untiled first; the tiled variant is checked below
    std::mt19937 rng(11);
    const std::uint32_t outputs = 520, width = 6144;  // 520: last block partly empty
    const Synthetic s = make(outputs, width, rng);
    std::vector<float> x(32U * width);
    for (auto& v : x) v = std::normal_distribution<float>(0.0F, 1.0F)(rng);
    auto* dc = dev_copy(s.codes);
    auto* ds = dev_copy(s.scales);
    auto* db = dev_copy(s.bits);
    auto* dx = dev_copy(x);
    float* dout = nullptr;
    ok(hipMalloc(&dout, std::max<std::size_t>(32U * outputs, 7U * hidden) * 4), "malloc");  // linears and rms_norm
    const Mxfp8Matrix hm{s.codes.data(), s.scales.data(), outputs, width}, dm{dc, ds, outputs, width};
    const Bf16Matrix hb{s.bits.data(), outputs, width}, dbm{db, outputs, width};
    bool mx_ok = true, bf_ok = true;
    std::vector<float> ref(32U * outputs);
    for (std::size_t rows : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{6},
                             std::size_t{7}, std::size_t{8}, std::size_t{32}}) {
        const auto cut = static_cast<std::ptrdiff_t>(rows * outputs);
        mxfp8_linear(hm, x.data(), rows, ref.data());
        dd.mxfp8_linear(dm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        mx_ok &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
        bf16_linear(hb, x.data(), rows, ref.data());
        dd.bf16_linear(dbm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        bf_ok &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
    }
    check(mx_ok, "mxfp8 rows kernel == host, rows 1..8 and 32 (row invariant by construction of the order)");
    check(bf_ok, "bf16 rows kernel == host, rows 1..8 and 32");
    // The same with the LDS-tiled kernels (rows <= 8; width 6144 is a multiple of the 1024 tile).
    dd.set_kernels(true, attention_chunk);
    bool tmx = true, tbf = true;
    for (std::size_t rows = 1; rows <= 8; ++rows) {
        const auto cut = static_cast<std::ptrdiff_t>(rows * outputs);
        mxfp8_linear(hm, x.data(), rows, ref.data());
        dd.mxfp8_linear(dm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        tmx &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
        bf16_linear(hb, x.data(), rows, ref.data());
        dd.bf16_linear(dbm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        tbf &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
    }
    check(tmx && tbf, "LDS-tiled mxfp8 and bf16 rows kernels == host, rows 1..8");
    dd.set_kernels(true, attention_chunk, true);
    bool pmx = true;
    for (std::size_t rows = 1; rows <= 8; ++rows) {
        const auto cut = static_cast<std::ptrdiff_t>(rows * outputs);
        mxfp8_linear(hm, x.data(), rows, ref.data());
        dd.mxfp8_linear(dm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        pmx &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
    }
    check(pmx, "tiled + register-prefetch mxfp8 rows kernel == host, rows 1..8");
    dd.set_kernels(false, attention_chunk, false, true);
    bool mmx = true;
    for (std::size_t rows = 1; rows <= 8; ++rows) {
        const auto cut = static_cast<std::ptrdiff_t>(rows * outputs);
        mxfp8_linear(hm, x.data(), rows, ref.data());
        dd.mxfp8_linear(dm, dx, rows, dout, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        mmx &= same_bits(host_copy(dout, rows * outputs), std::vector<float>(ref.begin(), ref.begin() + cut));
    }
    check(mmx, "4-outputs-per-wave mxfp8 rows kernel == host, rows 1..8 (520 outputs: partial last block)");
    // RMSNorm and head norm + RoPE.
    std::vector<std::uint16_t> nw(hidden), hw(head_dim);
    for (auto& b : nw) b = bf16_bits(std::normal_distribution<float>(0.0F, 0.3F)(rng));
    for (auto& b : hw) b = bf16_bits(std::normal_distribution<float>(0.0F, 0.3F)(rng));
    auto* dnw = dev_copy(nw);
    auto* dhw = dev_copy(hw);
    std::vector<float> hn(7U * hidden);
    rms_norm(x.data(), nw.data(), 7, hn.data());
    dd.rms_norm(dx, dnw, 7, dout, nullptr);
    ok(hipDeviceSynchronize(), "sync");
    check(same_bits(host_copy(dout, 7U * hidden), hn), "rms_norm kernel == host");
    std::vector<float> hq(x.begin(), x.begin() + 7U * q_width);
    for (std::uint32_t r = 0; r < 7; ++r) {
        std::array<float, 32> c{}, sn{};
        rope_table(4100 + r, c.data(), sn.data());
        head_norm_rope(hq.data() + r * q_width, heads, hw.data(), c.data(), sn.data());
    }
    auto* dq = dev_copy(std::vector<float>(x.begin(), x.begin() + 7U * q_width));
    dd.head_norm_rope(dq, 7, heads, dhw, 4100, nullptr);
    ok(hipDeviceSynchronize(), "sync");
    check(same_bits(host_copy(dq, 7U * q_width), hq), "head norm + RoPE kernel == host (positions 4100..4106)");
    for (void* p : {static_cast<void*>(dc), static_cast<void*>(ds), static_cast<void*>(db), static_cast<void*>(dx),
                    static_cast<void*>(dout), static_cast<void*>(dnw), static_cast<void*>(dhw), static_cast<void*>(dq)})
        (void)hipFree(p);
}

void test_real(DeviceDrafter& dd, const std::string& cases) {
    for (std::size_t b = 0; b < cases.size();) {
        const std::size_t e = std::min(cases.find(',', b), cases.size());
        const std::string cs = cases.substr(b, e - b);
        b = e + 1;
        const std::string meta = read_text(cs + "/meta.json");
        const auto n = static_cast<std::size_t>(json_number(meta, "context"));
        const auto P = static_cast<std::uint64_t>(json_number(meta, "position"));
        const auto anchor = static_cast<std::uint32_t>(json_number(meta, "anchor"));
        const auto taps_v = read_file<float>(cs + "/taps.f32");
        const auto embed = read_file<float>(cs + "/embed.f32");
        const auto base = read_file<float>(cs + "/base.f32");
        const auto host_y = read_file<float>(cs + "/host_y.f32");
        const std::string hj = read_text(cs + "/host.json");
        const auto hd = json_array(hj, "drafts"), hc = json_array(hj, "confidence_bits");
        std::vector<std::uint16_t> rows16(2U * hidden);
        for (std::uint32_t i = 0; i < hidden; ++i) {
            rows16[i] = bf16_bits(embed[i]);
            rows16[hidden + i] = bf16_bits(embed[hidden + i]);  // row 1 = the noise token's row
        }
        auto* d_taps = dev_copy(taps_v);
        auto* d_rows = dev_copy(rows16);
        auto* d_base = dev_copy(base);
        // Pieces of 3, 32, 1, 7, ... (different from the host test's pieces: rows are independent, bits must not change).
        const std::size_t start = n > context_keys ? n - context_keys : 0;
        const std::size_t pieces[] = {3, 32, 1, 7, 32, 8, 5};
        std::size_t pi = 0;
        for (std::size_t p = start; p < n;) {
            const std::size_t m = std::min(n - p, pieces[pi++ % 7]);
            dd.append(0, d_taps + p * tap_width, m, p, nullptr);
            p += m;
        }
        // v1 kernels (untiled linears, one-chunk attention) vs the host's v1 order; then the defaults (tiled, chunk 64).
        const auto host_y1 = read_file<float>(cs + "/host_y_v1.f32");
        dd.set_kernels(false, 0);
        dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        std::size_t first = 0;
        const auto yv1 = host_copy(dd.y(), static_cast<std::size_t>(block) * hidden);
        const bool v1_ok = same_bits(yv1, host_y1, &first);
        check(v1_ok, "real " + cs + ": v1 kernels, block y bitwise == host v1 order" + (v1_ok ? "" : " (first diff at " + std::to_string(first) + ")"));
        dd.set_kernels(true, 0);
        dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        check(same_bits(host_copy(dd.y(), static_cast<std::size_t>(block) * hidden), host_y1), "real " + cs + ": tiled linears == untiled (bitwise, v1 attention)");
        dd.set_kernels(true, attention_chunk);
        dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        const auto y = host_copy(dd.y(), static_cast<std::size_t>(block) * hidden);
        const bool y_ok = same_bits(y, host_y, &first);
        check(y_ok, "real " + cs + ": v2 kernels (tiled, chunked attention), block y bitwise == host" + (y_ok ? "" : " (first diff at " + std::to_string(first) + ")"));
        dd.set_kernels(true, attention_chunk, true);
        dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        check(same_bits(host_copy(dd.y(), static_cast<std::size_t>(block) * hidden), host_y), "real " + cs + ": prefetch-tiled linears + chunked attention == host");
        dd.set_kernels(false, attention_chunk, false, true);
        dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        check(same_bits(host_copy(dd.y(), static_cast<std::size_t>(block) * hidden), host_y), "real " + cs + ": 4-output linears + chunked attention == host");
        const Draft d = dd.sample(anchor, d_base, nullptr);
        bool tok = true, conf = true;
        for (std::uint32_t k = 0; k < block; ++k) {
            tok &= d.tokens[k] == static_cast<std::uint32_t>(hd[k]);
            conf &= __builtin_bit_cast(std::uint32_t, d.confidence[k]) == static_cast<std::uint32_t>(hc[k]);
        }
        check(tok && conf, "real " + cs + ": drafts and confidence bits == host");
        // Two-rank split of the vocabulary (the engine's halves): candidates combined == the full chain.
        constexpr std::uint32_t split = 782U * 128U;
        Candidate* dc = nullptr;
        ok(hipMalloc(&dc, 2 * sizeof(Candidate)), "malloc");
        bool split_ok = true;
        std::uint32_t t = anchor;
        for (std::uint32_t k = 0; k < block; ++k) {
            const float* row = d_base + static_cast<std::size_t>(k) * vocab;
            dd.markov_step(t, row, 0, split, dc, nullptr);
            dd.markov_step(t, row + split, split, vocab, dc + 1, nullptr);
            const auto c = host_copy(dc, 2);
            t = combine(c[0], c[1]).id;
            split_ok &= t == d.tokens[k];
        }
        check(split_ok, "real " + cs + ": Markov over two vocabulary halves + combine == full");
        // ABA x3 of the block pass: A = v1 kernels, B = v2 kernels (median of 10 per block).
        std::printf("ABA %s block ms:", cs.c_str());
        for (int rep = 0; rep < 3; ++rep)
            for (int arm = 0; arm < 2; ++arm) {
                dd.set_kernels(false, attention_chunk, false, arm == 1);
                std::printf(" %s=%.3f", arm == 0 ? "A" : "B", median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); }));
            }
        dd.set_kernels(false, attention_chunk);  // A again = the current default
        std::printf(" | A again=%.3f\n", median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); }));
        dd.set_kernels(true, 0);
        const double t_tiled = median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); });
        dd.set_kernels(false, attention_chunk);
        const double t_chunk = median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); });
        dd.set_kernels(true, attention_chunk);
        dd.set_kernels(true, attention_chunk, true);
        const double t_pf = median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); });
        dd.set_kernels(false, attention_chunk);
        std::printf("ATTRIB %s: tiled linears + v1 attention %.3f ms; untiled linears + chunked attention %.3f ms; "
                    "prefetch-tiled linears + chunked attention %.3f ms\n", cs.c_str(), t_tiled, t_chunk, t_pf);
        // Timings (this case's context), default kernels.
        const double t_block = median_ms([&] { dd.forward_block(0, d_rows, d_rows + hidden, P, nullptr); });
        const double t_a1 = median_ms([&] { dd.append(0, d_taps + (n - 1) * tap_width, 1, n - 1, nullptr); });
        const double t_a8 = median_ms([&] { dd.append(0, d_taps + (n - 8) * tap_width, 8, n - 8, nullptr); });
        const double t_a32 = median_ms([&] { dd.append(0, d_taps + (n - 32) * tap_width, 32, n - 32, nullptr); });
        const double t_mf = median_ms([&] { dd.markov_step(anchor, d_base, 0, vocab, dc, nullptr); });
        const double t_mh = median_ms([&] { dd.markov_step(anchor, d_base, 0, split, dc, nullptr); });
        std::printf("TIME %s context %zu: block %.3f ms, append 1/8/32 rows %.3f/%.3f/%.3f ms, markov step full %.3f ms half %.3f ms "
                    "=> draft (block + 7 half steps) %.3f ms\n",
                    cs.c_str(), static_cast<std::size_t>(P - (P > context_keys ? P - context_keys : 0)), t_block, t_a1, t_a8, t_a32, t_mf,
                    t_mh, t_block + 7 * t_mh);
        for (void* p : {static_cast<void*>(d_taps), static_cast<void*>(d_rows), static_cast<void*>(d_base), static_cast<void*>(dc)})
            (void)hipFree(p);
    }
}

} // namespace

int main() {
    try {
        const char* dir = std::getenv("STRIX_MM_DSPARK_DIR");
        const char* cases = std::getenv("STRIX_MM_DSPARK_CASE");
        std::unique_ptr<DraftFile> file;
        Weights w{};
        if (dir != nullptr) {
            const auto t0 = std::chrono::steady_clock::now();
            file = DraftFile::open(dir);
            w = file->weights();
            std::printf("draft %s read in %.1f s\n", dir, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        } else {
            std::printf("SKIP real weights (STRIX_MM_DSPARK_DIR unset): kernel tests with a zero-filled drafter shell are not run\n");
            return 0;
        }
        DeviceDrafter dd(w, {1, 8256});
        std::printf("device weights %.3f GB, ring %.1f MB per slot\n", static_cast<double>(dd.weight_bytes()) / 1e9,
                    static_cast<double>(dd.ring_bytes()) / 1e6);
        test_kernels(dd);
        if (cases != nullptr) test_real(dd, cases);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}

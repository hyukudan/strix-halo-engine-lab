// FP32 prefill identity tests, shared by the gfx1151 (tests/hip_mm_fp32_identity_tests.hip) and RTX PRO
// (cuda/mm_fp32_identity_tests.cu) drivers. The includer defines GPU_MALLOC / GPU_FREE / GPU_H2D / GPU_D2H / GPU_SYNC
// and includes the platform's fp32_identity API. Every case is checked bitwise against the host reference and prints an
// FNV-1a hash of the device output: the two platforms must print the same hashes.
#include "minimax/rtxpf/fp32_identity_ref.hpp"

#include <cstdio>
#include <vector>

namespace fp32_identity_tests {

namespace R = strix::minimax::rtxpf::ref;
using strix::minimax::rtxpf::F32Job;

struct Matrix {
    unsigned input, output;
    int bits;
    std::vector<std::uint32_t> trellis;
    std::vector<std::uint16_t> suh, svh;
    std::vector<float> w;  // host decode
};
inline Matrix make_matrix(R::Gen& g, unsigned input, unsigned output, int bits) {
    Matrix m{input, output, bits, {}, {}, {}, {}};
    m.trellis.resize(static_cast<std::size_t>(input / 16) * (output / 16) * static_cast<std::size_t>(bits) * 8);
    for (auto& w : m.trellis) w = static_cast<std::uint32_t>(g.next());
    m.suh.resize(input);
    m.svh.resize(output);
    for (auto& s : m.suh) s = g.half_scale();
    for (auto& s : m.svh) s = g.half_scale();
    m.w = R::decode(m.trellis.data(), input, output, bits);
    return m;
}
template <class T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    GPU_MALLOC(&d, h.size() * sizeof(T));
    GPU_H2D(d, h.data(), h.size() * sizeof(T));
    return d;
}

// Grouped case: `mats` share input / output; job j has rows[j] entries; xmap picks source rows of x (with repeats), ymap
// scatters to a permutation of y rows. tile_rows 0 = mixed: 128-row tiles plus the 64-row tail core (split_tails).
inline long long grouped_case(const char* name, R::Gen& g, std::vector<Matrix> mats, const std::vector<unsigned>& rows, unsigned x_rows,
                              bool maps, std::uint32_t tile_rows = 64) {
    const unsigned input = mats[0].input, output = mats[0].output;
    std::vector<F32Job> jobs;
    std::vector<std::uint32_t*> dtr;
    std::vector<std::uint16_t*> dsuh, dsvh;
    for (std::size_t j = 0; j < mats.size(); ++j) {
        dtr.push_back(upload(mats[j].trellis));
        dsuh.push_back(upload(mats[j].suh));
        dsvh.push_back(upload(mats[j].svh));
        jobs.push_back({dtr.back(), dsuh.back(), dsvh.back(), input, output, static_cast<std::uint32_t>(mats[j].bits), rows[j], 0, 0});
    }
    const bool mixed = tile_rows == 0;
    auto ext = strix::minimax::rtxpf::layout_jobs(jobs, mixed ? 128U : tile_rows);
    const std::vector<F32Job> full = jobs;  // rows / row_begin before split_tails (the reference loop)
    std::vector<F32Job> tail;
    std::uint32_t tail_tiles = 0;
    if (mixed) {
        tail_tiles = strix::minimax::rtxpf::split_tails(jobs, tail, ext.tiles);
        jobs.insert(jobs.end(), tail.begin(), tail.end());
    }
    std::vector<float> x(static_cast<std::size_t>(x_rows) * input);
    for (auto& v : x) v = g.value(3, 12);
    std::vector<std::uint32_t> xmap(ext.entries), ymap(ext.entries);
    for (unsigned e = 0; e < ext.entries; ++e) { xmap[e] = maps ? static_cast<std::uint32_t>(g.next() % x_rows) : e; ymap[e] = e; }
    if (maps)
        for (unsigned e = ext.entries; e-- > 1;) std::swap(ymap[e], ymap[g.next() % (e + 1)]);
    float* dx = upload(x);
    std::uint32_t* dxm = upload(xmap);
    std::uint32_t* dym = upload(ymap);
    F32Job* djobs = upload(jobs);
    float *dy = nullptr, *xt = nullptr, *raw = nullptr;
    GPU_MALLOC(&dy, static_cast<std::size_t>(ext.entries) * output * 4);
    GPU_MALLOC(&xt, static_cast<std::size_t>(ext.entries) * input * 4);
    GPU_MALLOC(&raw, static_cast<std::size_t>(ext.entries) * output * 4);
    strix::minimax::rtxpf::GroupedCall c;
    c.jobs_d = djobs;
    c.jobs = static_cast<std::uint32_t>(jobs.size() - tail.size());
    c.tiles = ext.tiles;
    c.tile_rows = mixed ? 128U : tile_rows;
    c.tail_jobs = static_cast<std::uint32_t>(tail.size());
    c.tail_tiles = tail_tiles;
    c.input = input;
    c.output = output;
    c.entries = ext.entries;
    c.x = dx;
    c.x_stride = input;
    c.xmap = maps ? dxm : nullptr;
    c.y = dy;
    c.y_stride = output;
    c.ymap = maps ? dym : nullptr;
#ifdef FP32_TEST_CONFIGURE_CORE
    FP32_TEST_CONFIGURE_CORE(c);
#endif
    strix::minimax::rtxpf::enqueue_grouped(c, xt, raw, nullptr);
    GPU_SYNC();
    std::vector<float> y(static_cast<std::size_t>(ext.entries) * output);
    GPU_D2H(y.data(), dy, y.size() * 4);
    long long bad = 0;
    std::vector<float> want(output);
    for (std::size_t j = 0; j < full.size(); ++j)
        for (unsigned r = 0; r < full[j].rows; ++r) {
            const unsigned e = full[j].row_begin + r;
            R::linear_row(mats[j].w.data(), mats[j].suh.data(), mats[j].svh.data(), input, output, x.data() + static_cast<std::size_t>(xmap[e]) * input,
                          want.data());
            for (unsigned o = 0; o < output; ++o) bad += R::bits_of(want[o]) != R::bits_of(y[static_cast<std::size_t>(ymap[e]) * output + o]);
        }
    std::printf("%-44s entries %5u tiles %5u: %lld mismatches  fnv %016llx%s\n", name, ext.entries, ext.tiles, bad,
                static_cast<unsigned long long>(R::fnv(y.data(), y.size())), bad ? "  FAIL" : "");
    for (auto p : dtr) GPU_FREE(p);
    for (auto p : dsuh) GPU_FREE(p);
    for (auto p : dsvh) GPU_FREE(p);
    GPU_FREE(dx); GPU_FREE(dxm); GPU_FREE(dym); GPU_FREE(djobs); GPU_FREE(dy); GPU_FREE(xt); GPU_FREE(raw);
    return bad;
}

inline long long combine_case(R::Gen& g, unsigned rows) {
    const unsigned pairs = rows * 4;
    std::vector<float> ye(static_cast<std::size_t>(pairs) * 6144), ys(static_cast<std::size_t>(rows) * 6144), w(pairs);
    for (auto& v : ye) v = g.value(2, 10);
    for (auto& v : ys) v = g.value(2, 10);
    for (auto& v : w) v = g.value(1, 4);
    std::vector<std::uint32_t> slot(pairs);
    for (unsigned p = 0; p < pairs; ++p) slot[p] = static_cast<std::uint32_t>(g.next() % pairs);
    float *dye = upload(ye), *dys = upload(ys), *dw = upload(w), *dout = nullptr;
    std::uint32_t* ds = upload(slot);
    GPU_MALLOC(&dout, ys.size() * 4);
    strix::minimax::rtxpf::enqueue_combine(dye, dys, ds, dw, rows, dout, nullptr);
    GPU_SYNC();
    std::vector<float> out(ys.size());
    GPU_D2H(out.data(), dout, out.size() * 4);
    long long bad = 0;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < 6144; ++c) {
            float acc = 0.0F;
            for (unsigned i = 0; i < 4; ++i) acc = std::fmaf(w[r * 4 + i], ye[static_cast<std::size_t>(slot[r * 4 + i]) * 6144 + c], acc);
            acc = acc + ys[static_cast<std::size_t>(r) * 6144 + c];
            bad += R::bits_of(acc) != R::bits_of(out[static_cast<std::size_t>(r) * 6144 + c]);
        }
    std::printf("%-44s rows %u: %lld mismatches  fnv %016llx%s\n", "combine (fmaf ascending + shared)", rows, bad,
                static_cast<unsigned long long>(R::fnv(out.data(), out.size())), bad ? "  FAIL" : "");
    GPU_FREE(dye); GPU_FREE(dys); GPU_FREE(dw); GPU_FREE(ds); GPU_FREE(dout);
    return bad;
}

inline long long swiglu_case(R::Gen& g, unsigned n) {
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = g.value(4, 16);
    for (auto& v : b) v = g.value(4, 16);
    float *da = upload(a), *db = upload(b), *dy = nullptr;
    GPU_MALLOC(&dy, n * 4);
    strix::minimax::rtxpf::enqueue_swiglu(da, db, dy, n, nullptr);
    GPU_SYNC();
    std::vector<float> y(n);
    GPU_D2H(y.data(), dy, n * 4);
    long long bad = 0;
    for (unsigned i = 0; i < n; ++i) bad += R::bits_of(R::swiglu(a[i], b[i])) != R::bits_of(y[i]);
    std::printf("%-44s n %u: %lld mismatches  fnv %016llx%s\n", "swiglu (portable exp)", n, bad,
                static_cast<unsigned long long>(R::fnv(y.data(), y.size())), bad ? "  FAIL" : "");
    GPU_FREE(da); GPU_FREE(db); GPU_FREE(dy);
    return bad;
}

inline int run_all() {
    R::Gen g{20261024ULL};
    long long bad = 0;
    // half-precision helper self-check against the platform conversion is done by the drivers
    bad += grouped_case("dense q-like K6 6144->256, 17 rows", g, {make_matrix(g, 6144, 256, 6)}, {17}, 17, false);
    bad += grouped_case("dense o-like K6 512->384, 70 rows", g, {make_matrix(g, 512, 384, 6)}, {70}, 70, false);
    bad += grouped_case("grouped gate-like K3 768->256, 3 jobs, maps", g,
                        {make_matrix(g, 768, 256, 3), make_matrix(g, 768, 256, 3), make_matrix(g, 768, 256, 5)}, {70, 1, 33}, 50, true);
    bad += grouped_case("grouped down-like K3 1536->6144(part) 2 jobs", g, {make_matrix(g, 1536, 640, 3), make_matrix(g, 1536, 640, 3)},
                        {9, 64}, 73, true);
    {  // the same grouped case at both core tile heights: identical bits (and hash)
        R::Gen g2{99};
        auto m1 = make_matrix(g2, 768, 256, 3), m2 = make_matrix(g2, 768, 256, 5);
        bad += grouped_case("grouped 2 jobs (130, 200 rows), tile 64", g2, {m1, m2}, {130, 200}, 300, true, 64);
        R::Gen g3{99};
        auto n1 = make_matrix(g3, 768, 256, 3), n2 = make_matrix(g3, 768, 256, 5);
        bad += grouped_case("grouped 2 jobs (130, 200 rows), tile 128", g3, {n1, n2}, {130, 200}, 300, true, 128);
        R::Gen g4{99};
        auto o1 = make_matrix(g4, 768, 256, 3), o2 = make_matrix(g4, 768, 256, 5);
        bad += grouped_case("grouped 2 jobs (130, 200 rows), 128 + tail 64", g4, {o1, o2}, {130, 200}, 300, true, 0);
    }
    {  // mixed tiles: a remainder > 64 stays in a 128-row CTA, a job below 64 rows lives only in the tail core
        R::Gen g5{7};
        bad += grouped_case("grouped 4 jobs (130, 200, 40, 300), 128 + tail", g5,
                            {make_matrix(g5, 768, 256, 3), make_matrix(g5, 768, 256, 4), make_matrix(g5, 768, 256, 3), make_matrix(g5, 768, 256, 5)},
                            {130, 200, 40, 300}, 400, true, 0);
    }
    bad += combine_case(g, 37);
    bad += swiglu_case(g, 1U << 20);
    std::printf("%s\n", bad ? "FP32 IDENTITY TESTS FAIL" : "FP32 IDENTITY TESTS PASS");
    return bad ? 1 : 0;
}

}  // namespace fp32_identity_tests

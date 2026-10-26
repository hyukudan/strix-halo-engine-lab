// CPU tests of the minimax-m3 MSA host reference: norm + RoPE and attention against double precision, the block
// indexer and selection against a brute-force sort, and (STRIX_MINIMAX_MSA_CASE=DIR) a real case exported from the
// float64 oracle by tools/minimax/export_msa_case.py.
#include "strix/minimax_msa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace msa = strix::minimax::msa;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

double rel_err(const std::vector<float>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - ref[i]) * (a[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

void test_norm_rope() {
    std::mt19937 rng(1);
    std::normal_distribution<float> n(0.0F, 3.0F);
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<float> x(128), w(128);
        for (auto& v : x) v = n(rng);
        for (auto& v : w) v = n(rng) * 0.2F;
        const std::uint64_t pos = static_cast<std::uint64_t>(rng() % 1000000);
        const auto rope = msa::rope_row(pos, 1e7);
        std::vector<double> ref(128);
        double ss = 0;
        for (float v : x) ss += static_cast<double>(v) * v;
        const double r = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
        for (int i = 0; i < 128; ++i) ref[i] = x[i] * r * (1.0 + w[i]);
        for (int i = 0; i < 32; ++i) {
            const double ang = static_cast<double>(pos) / std::pow(1e7, 2.0 * i / 64.0);
            const double a = ref[i], b = ref[i + 32];
            ref[i] = a * std::cos(ang) - b * std::sin(ang);
            ref[i + 32] = b * std::cos(ang) + a * std::sin(ang);
        }
        msa::head_norm_rope(x.data(), w.data(), rope);
        check(rel_err(x, ref) < 2e-6, "norm+rope vs double");
    }
}

void test_selection() {
    std::mt19937 rng(2);
    for (unsigned blocks = 1; blocks < 400; blocks += 7) {
        std::vector<float> s(blocks);
        for (auto& v : s) v = static_cast<float>(static_cast<int>(rng() % 20)) * 0.5F;  // many ties
        const auto got = msa::select_blocks(s);
        std::vector<std::uint32_t> want;
        if (blocks <= msa::kTopK) {
            for (unsigned b = 0; b < blocks; ++b) want.push_back(b);
        } else {
            std::vector<std::pair<float, std::uint32_t>> all;
            for (unsigned b = 0; b + 1 < blocks; ++b) all.emplace_back(-s[b], b);
            std::sort(all.begin(), all.end());
            for (unsigned i = 0; i + 1 < msa::kTopK; ++i) want.push_back(all[i].second);
            want.push_back(blocks - 1);
            std::sort(want.begin(), want.end());
        }
        check(got == want, "selection vs sort");
    }
}

void test_attention() {
    std::mt19937 rng(3);
    std::normal_distribution<float> n(0.0F, 1.0F);
    for (std::uint64_t count : {1ULL, 77ULL, 128ULL, 2048ULL, 2049ULL, 5000ULL}) {
        const unsigned heads = 16;
        std::vector<float> q(heads * 128), k(count * 128), v(count * 128), out(heads * 128);
        for (auto& x : q) x = n(rng) * 2.0F;
        for (auto& x : k) x = n(rng) * 2.0F;
        for (auto& x : v) x = n(rng);
        std::vector<float> scores(msa::block_count(count));
        for (std::size_t b = 0; b < scores.size(); ++b) scores[b] = n(rng);
        const auto blocks = msa::select_blocks(scores);
        msa::sparse_attention(q.data(), heads, k.data(), v.data(), count, blocks, out.data());
        std::vector<double> ref(heads * 128, 0.0);
        for (unsigned h = 0; h < heads; ++h) {
            std::vector<std::uint64_t> pos;
            for (auto b : blocks)
                for (std::uint64_t t = b * 128ULL; t < std::min<std::uint64_t>(count, (b + 1) * 128ULL); ++t) pos.push_back(t);
            std::vector<double> s(pos.size());
            double mx = -1e300;
            for (std::size_t j = 0; j < pos.size(); ++j) {
                double d = 0;
                for (int i = 0; i < 128; ++i) d += static_cast<double>(q[h * 128 + i]) * k[pos[j] * 128 + i];
                s[j] = d / std::sqrt(128.0);
                mx = std::max(mx, s[j]);
            }
            double den = 0;
            for (auto& x : s) den += (x = std::exp(x - mx));
            for (std::size_t j = 0; j < pos.size(); ++j)
                for (int i = 0; i < 128; ++i) ref[h * 128 + i] += s[j] / den * v[pos[j] * 128 + i];
        }
        check(rel_err(out, ref) < 2e-6, "sparse attention vs double");
    }
}

std::vector<float> read_f32(const std::string& path, std::size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    return v;
}

void test_real_case(const std::string& dir) {
    std::ifstream meta(dir + "/meta.txt");
    std::uint64_t T = 0;
    std::size_t rows = 0;
    double theta = 0;
    int layer = 0;
    meta >> T >> rows >> theta >> layer;
    std::vector<std::uint64_t> pos(rows);
    for (auto& p : pos) meta >> p;
    const auto q = read_f32(dir + "/q.f32", rows * 64 * 128);
    const auto k = read_f32(dir + "/k.f32", T * 4 * 128);
    const auto v = read_f32(dir + "/v.f32", T * 4 * 128);
    const auto iq = read_f32(dir + "/iq.f32", rows * 4 * 128);
    const auto ik = read_f32(dir + "/ik.f32", T * 128);
    const auto attn = read_f32(dir + "/attn.f32", rows * 64 * 128);
    auto q_raw0 = read_f32(dir + "/q_raw0.f32", 64 * 128);
    const auto qw = read_f32(dir + "/q_norm_w.f32", 128);
    std::ifstream self(dir + "/sel.txt");
    // norm + RoPE of row 0 against the oracle's normed/roped q.
    const auto rope = msa::rope_row(pos[0], theta);
    double worst = 0;
    for (unsigned h = 0; h < 64; ++h) {
        msa::head_norm_rope(q_raw0.data() + h * 128, qw.data(), rope);
        std::vector<float> got(q_raw0.begin() + h * 128, q_raw0.begin() + (h + 1) * 128);
        std::vector<double> want(q.begin() + h * 128, q.begin() + (h + 1) * 128);
        worst = std::max(worst, rel_err(got, want));
    }
    std::printf("real case layer %d T %llu: norm+rope row %llu worst head rel err %.3e\n", layer,
                static_cast<unsigned long long>(T), static_cast<unsigned long long>(pos[0]), worst);
    check(worst < 2e-6, "real norm+rope");
    // Per-group K/V and selection.
    unsigned sel_equal = 0, sel_total = 0;
    double attn_worst = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::uint64_t count = pos[r] + 1;
        for (unsigned g = 0; g < 4; ++g) {
            std::string line;
            std::getline(self, line);
            std::istringstream ls(line);
            std::vector<std::uint32_t> oracle_sel;
            for (std::uint32_t b; ls >> b;) oracle_sel.push_back(b);
            std::vector<float> scores(msa::block_count(count));
            msa::index_block_scores(iq.data() + (r * 4 + g) * 128, ik.data(), count, scores);
            const auto sel = msa::select_blocks(scores);
            ++sel_total;
            sel_equal += sel == oracle_sel ? 1U : 0U;
            std::vector<float> kg(count * 128), vg(count * 128), out(16 * 128);
            for (std::uint64_t t = 0; t < count; ++t)
                for (int i = 0; i < 128; ++i) {
                    kg[t * 128 + i] = k[(t * 4 + g) * 128 + i];
                    vg[t * 128 + i] = v[(t * 4 + g) * 128 + i];
                }
            msa::sparse_attention(q.data() + (r * 64 + g * 16) * 128, 16, kg.data(), vg.data(), count, sel, out.data());
            std::vector<double> want(attn.begin() + (r * 64 + g * 16) * 128, attn.begin() + (r * 64 + g * 16 + 16) * 128);
            const double e = rel_err(out, want);
            if (sel == oracle_sel) attn_worst = std::max(attn_worst, e);
        }
    }
    std::printf("real case: selections equal %u/%u; attention worst rel err (equal selections) %.3e\n", sel_equal,
                sel_total, attn_worst);
    check(sel_equal == sel_total, "real selection equals oracle");
    check(attn_worst < 1e-5, "real attention vs oracle");
}

}  // namespace

int main() {
    test_norm_rope();
    test_selection();
    test_attention();
    if (const char* dir = std::getenv("STRIX_MINIMAX_MSA_CASE")) test_real_case(dir);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("strix-minimax-msa-tests: all passed\n");
    return 0;
}

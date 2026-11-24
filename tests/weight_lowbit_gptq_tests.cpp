// Calibrated low-bit packing (strix/weight/host_lowbit_gptq.hpp): without importance or feedback the bytes are the
// load-time encoder's; a diagonal second-moment matrix moves no error; the bytes and the second moments never depend
// on the thread count; the decoded copy equals the reference decode of the bytes; error feedback and importance lower
// the output error they target on correlated inputs; a group scale beyond FP16 is refused.
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/calib_moments.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace gq = strix::weight::lowbit::gptq;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

std::vector<float> random_weights(std::size_t n, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::vector<float> w(n * k);
    for (auto& v : w) v = normal(rng);
    for (std::size_t i = 0; i < w.size(); i += 97U) w[i] *= 6.0F;  // outliers so the clip search matters
    return w;
}

// Correlated inputs: a few shared factors plus per-column noise of uneven scale (activation outlier columns).
std::vector<float> correlated_rows(std::size_t rows, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    constexpr std::size_t factors = 8;
    std::vector<float> mix(factors * k), x(rows * k);
    for (auto& v : mix) v = normal(rng);
    for (std::size_t r = 0; r < rows; ++r) {
        float f[factors];
        for (auto& v : f) v = normal(rng);
        for (std::size_t i = 0; i < k; ++i) {
            float v = 0.3F * normal(rng) * (i % 29U == 0U ? 8.0F : 1.0F);
            for (std::size_t p = 0; p < factors; ++p) v += f[p] * mix[p * k + i];
            x[r * k + i] = v;
        }
    }
    return x;
}

template <class Storage>
std::vector<std::uint8_t> rtn(const std::vector<float>& w, std::size_t n, std::size_t k) {
    std::vector<std::uint8_t> out(n * Storage::row_bytes(k));
    for (std::size_t b = 0; b < n * k / Storage::elements; ++b)
        check(wei::lowbit::encode_block<Storage>(w.data() + b * Storage::elements, out.data() + b * Storage::bytes),
              "reference encode");
    return out;
}

template <class Storage>
std::vector<float> decode_all(const std::vector<std::uint8_t>& bytes, std::size_t n, std::size_t k) {
    std::vector<float> out(n * k);
    for (std::size_t e = 0; e < n * k; ++e)
        out[e] = wei::lowbit::decode<Storage>(bytes.data() + (e / Storage::elements) * Storage::bytes,
                                              static_cast<unsigned>(e % Storage::elements));
    return out;
}

template <class Storage>
void check_storage(const char* name) {
    const std::string tag = name;
    constexpr std::size_t n = 37, k = 256, rows = 768;
    const auto w = random_weights(n, k, 7U);
    const auto reference = rtn<Storage>(w, n, k);

    // No technique: the load-time bytes.
    std::vector<std::uint8_t> plain(reference.size());
    check(gq::encode_rows<Storage>(w.data(), n, k, plain.data()), tag + " plain encode");
    check(plain == reference, tag + " plain == encode_block");

    // Diagonal second moments: the feedback moves nothing.
    std::vector<double> diag(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i) diag[i * k + i] = 0.5 + static_cast<double>(i % 13U);
    std::vector<std::uint8_t> dead;
    const auto ud = gq::inverse_factor(diag, k, 0.01, dead, 2U);
    std::vector<std::uint8_t> fed(reference.size());
    gq::Options od;
    od.factor = ud.data();
    od.dead = dead.data();
    check(gq::encode_rows<Storage>(w.data(), n, k, fed.data(), od), tag + " diagonal encode");
    check(fed == reference, tag + " diagonal H == encode_block");

    // Correlated calibration rows: thread invariance of the moments and of the bytes, decode agreement, error drop.
    const auto x = correlated_rows(rows, k, 11U);
    std::vector<double> h1, h4;
    gq::accumulate_second_moments(x.data(), rows, k, h1, 1U);
    gq::accumulate_second_moments(x.data(), rows, k, h4, 4U);
    check(h1 == h4, tag + " moments thread invariant");
    const auto u = gq::inverse_factor(h1, k, 0.01, dead, 3U);
    std::vector<float> importance(k);
    for (std::size_t i = 0; i < k; ++i) importance[i] = static_cast<float>(h1[i * k + i]);

    std::vector<std::uint8_t> g1(reference.size()), g3(reference.size());
    std::vector<float> dq(n * k);
    gq::Options o1;
    o1.factor = u.data();
    o1.dead = dead.data();
    o1.threads = 1U;
    o1.dequant = dq.data();
    gq::Options o3 = o1;
    o3.threads = 3U;
    o3.dequant = nullptr;
    check(gq::encode_rows<Storage>(w.data(), n, k, g1.data(), o1), tag + " feedback encode");
    check(gq::encode_rows<Storage>(w.data(), n, k, g3.data(), o3), tag + " feedback encode threads");
    check(g1 == g3, tag + " feedback bytes thread invariant");
    check(dq == decode_all<Storage>(g1, n, k), tag + " dequant == decode(bytes)");

    std::vector<std::uint8_t> wi(reference.size());
    gq::Options oi;
    oi.importance = importance.data();
    check(gq::encode_rows<Storage>(w.data(), n, k, wi.data(), oi), tag + " importance encode");

    const auto e_rtn = gq::output_error(w.data(), decode_all<Storage>(reference, n, k).data(), n, k, h1).relative();
    const auto e_fed = gq::output_error(w.data(), dq.data(), n, k, h1).relative();
    check(e_fed < 0.9 * e_rtn, tag + " feedback lowers the output error (" + std::to_string(e_fed) + " vs " +
                                   std::to_string(e_rtn) + ")");
    // Importance targets the diagonal part of the output error.
    const auto diag_error = [&](const std::vector<float>& q) {
        double e = 0.0;
        for (std::size_t c = 0; c < n; ++c)
            for (std::size_t i = 0; i < k; ++i) {
                const double d = static_cast<double>(q[c * k + i]) - w[c * k + i];
                e += h1[i * k + i] * d * d;
            }
        return e;
    };
    check(diag_error(decode_all<Storage>(wi, n, k)) <= diag_error(decode_all<Storage>(reference, n, k)),
          tag + " importance lowers the weighted error");
    std::cout << tag << " relative output error rtn " << e_rtn << " feedback " << e_fed << "\n";

    // A group whose unclipped scale exceeds FP16 is refused.
    auto big = w;
    big[3] = 1.0e9F;
    std::vector<std::uint8_t> refused(reference.size());
    check(gq::encode_rows<Storage>(big.data(), n, k, refused.data()).code == wei::lowbit::EncodeStatus::scale_overflow,
          tag + " scale overflow refused");
    big[3] = NAN;
    check(gq::encode_rows<Storage>(big.data(), n, k, refused.data()).code == wei::lowbit::EncodeStatus::nonfinite,
          tag + " nonfinite refused");
}

// Calibration statistics: one row set added to several inputs equals each input's own sum; the file round-trips
// (float32 storage); consumers resolve; a conflicting redeclaration is refused.
void check_moments() {
    namespace cal = strix::weight::calib;
    constexpr std::size_t k = 128, rows = 200;
    const auto x = correlated_rows(rows, k, 5U);
    cal::MomentSet set(3U);
    const std::array<std::string, 1> qc{"model.layers.0.q.weight"}, kc{"model.layers.0.k.weight"};
    auto* q = &set.declare("layers.0.query_input", static_cast<std::uint32_t>(k), qc);
    auto* key = &set.declare("layers.0.key_input", static_cast<std::uint32_t>(k), kc);
    const std::array<cal::InputMoments*, 2> both{q, key};
    set.add(both, x.data(), rows);
    set.add(std::span(&q, 1U), x.data(), rows / 2U);
    std::vector<double> once, twice;
    gq::accumulate_second_moments(x.data(), rows, k, once, 1U);
    twice = once;
    gq::accumulate_second_moments(x.data(), rows / 2U, k, twice, 1U);
    check(key->h == once && key->rows == rows, "moments shared add == own sum");
    check(q->h == twice && q->rows == rows + rows / 2U, "moments repeated add");
    const auto path = (std::filesystem::temp_directory_path() / "strix_calib_moments_test.bin").string();
    set.write(path);
    const auto back = cal::MomentSet::read(path);
    std::remove(path.c_str());
    const auto* bq = back.find_consumer("model.layers.0.q.weight");
    check(bq != nullptr && bq->name == "layers.0.query_input" && bq->rows == q->rows, "moments file round trip header");
    bool same = bq != nullptr;
    for (std::size_t i = 0; same && i < k; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            same = same && bq->h[i * k + j] == static_cast<double>(static_cast<float>(q->h[i * k + j]));
    check(same, "moments file round trip values");
    check(back.find_consumer("model.layers.0.v.weight") == nullptr, "moments unknown consumer");
    bool refused = false;
    try {
        (void)set.declare("layers.0.query_input", static_cast<std::uint32_t>(k + 64U), qc);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "moments conflicting redeclaration refused");
}

// Provenance: windows and every 32-id run travel with the statistics file; a v1 file (none) is refused; overlap with
// a gate manifest (identical window, or a shared stretch of >= 32 ids at any offset) is found, disjoint text is not.
void check_provenance() {
    namespace cal = strix::weight::calib;
    std::vector<std::uint32_t> text(400);
    for (std::size_t i = 0; i < text.size(); ++i) text[i] = static_cast<std::uint32_t>((i * 7919U) % 50000U);
    const std::span<const std::uint32_t> all(text);
    cal::MomentSet set(1U);
    const std::array<std::string, 1> qc{"q.weight"};
    (void)set.declare("in", 64U, qc);
    set.add_window(all.subspan(0, 200));
    set.add_window(all.subspan(200, 100));
    const auto dir = std::filesystem::temp_directory_path() / "strix_calib_provenance_test";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "m.bin").string();
    set.write(path);
    const auto back = cal::MomentSet::read(path);
    check(back.provenance().windows().size() == 2U && back.provenance().shingles() == set.provenance().shingles() &&
              back.provenance().windows()[1].tokens == 100U &&
              back.provenance().windows()[1].digest.bytes == cal::window_identity(all.subspan(200, 100)).bytes,
          "provenance file round trip");
    {
        std::ofstream v1((dir / "v1.bin").string(), std::ios::binary | std::ios::trunc);
        v1 << "strix-calib-moments/v1 0\n";
    }
    bool v1_refused = false;
    try {
        (void)cal::MomentSet::read((dir / "v1.bin").string());
    } catch (const std::runtime_error&) {
        v1_refused = true;
    }
    check(v1_refused, "v1 moments (no provenance) refused");
    const auto manifest = [&](const std::string& name, std::span<const std::uint32_t> ids) {
        {
            std::ofstream f((dir / (name + ".ids")).string(), std::ios::trunc);
            for (const auto id : ids) f << id << ' ';
        }
        {
            std::ofstream m((dir / (name + ".txt")).string(), std::ios::trunc);
            m << "# gate cases\ncase0 " << ids.size() << ' ' << name << ".ids\n";
        }
        cal::Provenance p;
        p.add_manifest((dir / (name + ".txt")).string());
        return p;
    };
    check(!cal::overlap(back.provenance(), manifest("same", all.subspan(200, 100))).empty(), "identical window found");
    check(!cal::overlap(back.provenance(), manifest("inside", all.subspan(37, 40))).empty(),
          "40-id stretch at an unaligned offset found");
    check(cal::overlap(back.provenance(), manifest("short", all.subspan(250, 20))).empty(),
          "20-id stretch below the run length not claimed");
    check(cal::overlap(back.provenance(), manifest("disjoint", all.subspan(300, 100))).empty(), "disjoint text passes");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// Worker failures: an exception thrown by one slice reaches the caller after every thread joined (no terminate).
void check_parallel_errors() {
    bool caught = false;
    std::vector<int> done(64, 0);
    try {
        gq::detail::parallel_for(64U, 4U, [&](std::size_t c) {
            if (c == 13U) throw std::runtime_error("slice failure");
            done[c] = 1;
        });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "slice failure";
    }
    check(caught, "worker exception rethrown on the caller");
    std::size_t ran = 0;
    for (const int d : done) ran += static_cast<std::size_t>(d);
    check(ran >= 48U, "other slices ran to completion");
}

}  // namespace

int main() {
    check_moments();
    check_provenance();
    check_parallel_errors();
    check_storage<wei::Q4G64Storage>("q4_g64");
    check_storage<wei::Q6G64Storage>("q6_g64");
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight lowbit gptq tests passed\n";
    return 0;
}

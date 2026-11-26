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
#include <iterator>
#include <limits>
#include <new>
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
    std::vector<std::uint32_t> ids(64);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint32_t>(i);
    const auto window = set.add_window(ids);
    set.add(window, both, x.data(), rows);
    set.add(window, std::span(&q, 1U), x.data(), rows / 2U);
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
    check(bq != nullptr && bq->name == "layers.0.query_input" && bq->rows == q->rows &&
              bq->window_rows == std::vector<std::uint64_t>{rows + rows / 2U},
          "moments file round trip header");
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
    bool bad_window = false;
    try {
        set.add(7U, std::span(&q, 1U), x.data(), 1U);
    } catch (const std::invalid_argument&) {
        bad_window = true;
    }
    check(bad_window, "moments rows for an unregistered window refused");
}

template <class Fn>
bool throws_runtime(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Every accumulated row belongs to a registered window: rows counted outside any window (mixed fixture) are refused on
// write and on read; a fully attributed set round-trips with its per-window counts; v2 files are refused.
void check_attribution() {
    namespace cal = strix::weight::calib;
    constexpr std::size_t k = 64, rows = 40;
    const auto x = correlated_rows(rows, k, 9U);
    std::vector<std::uint32_t> ids(100);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint32_t>(i * 13U);
    const auto dir = std::filesystem::temp_directory_path() / "strix_calib_attribution_test";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "m.bin").string();
    const std::array<std::string, 1> c{"q.weight"};
    {
        cal::MomentSet set(1U);
        auto* in = &set.declare("in", static_cast<std::uint32_t>(k), c);
        const auto w0 = set.add_window(std::span<const std::uint32_t>(ids).subspan(0, 50));
        const auto w1 = set.add_window(std::span<const std::uint32_t>(ids).subspan(50, 50));
        set.add(w0, std::span(&in, 1U), x.data(), rows);
        set.add(w1, std::span(&in, 1U), x.data(), rows / 2U);
        set.write(path);
        const auto back = cal::MomentSet::read(path);
        const auto* bin = back.find("in");
        check(bin != nullptr && bin->rows == rows + rows / 2U &&
                  bin->window_rows == std::vector<std::uint64_t>{rows, rows / 2U},
              "attributed moments round trip per-window rows");
    }
    {  // one registered window's rows + rows that no window owns
        cal::MomentSet set(1U);
        auto* in = &set.declare("in", static_cast<std::uint32_t>(k), c);
        const auto w0 = set.add_window(std::span<const std::uint32_t>(ids).subspan(0, 50));
        set.add(w0, std::span(&in, 1U), x.data(), rows);
        in->rows += 5U;  // rows that reached the sum without a window
        check(throws_runtime([&] { set.write(path); }, "attributed"), "mixed registered/unregistered rows refused on write");
        in->rows -= 5U;
        set.write(path);
        std::string text;
        {
            std::ifstream f(path, std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        const std::string good = "input in 64 40 40 ";
        const auto at = text.find(good);
        check(at != std::string::npos, "attribution fixture header found");
        if (at != std::string::npos) text.replace(at, good.size(), "input in 64 45 40 ");  // total 45, attributed 40
        {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            f.write(text.data(), static_cast<std::streamsize>(text.size()));
        }
        check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "attributed"),
              "mixed registered/unregistered rows refused on read");
    }
    {  // rows but no window at all
        cal::MomentSet set(1U);
        auto* in = &set.declare("in", static_cast<std::uint32_t>(k), c);
        in->rows = 3U;
        check(throws_runtime([&] { set.write(path); }, "attributed"), "rows without any window refused on write");
    }
    {
        std::ofstream v2(path, std::ios::binary | std::ios::trunc);
        v2 << "strix-calib-moments/v2 0 0 0\n";
    }
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "v2"), "v2 moments refused");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void spit(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

int fail_stage = -1;
void failing_hook(int stage) {
    if (stage == fail_stage) throw std::bad_alloc();
}

// The window ids are the source of the identities: a file whose 32-id run records are removed (or whose ids are missing
// or edited) is refused on read, so a window that differs from a held-out window only in its last token still shows its
// shared 32-id prefix to the separation rule.
void check_identity_from_ids() {
    namespace cal = strix::weight::calib;
    std::vector<std::uint32_t> held(33), eval(33);
    for (std::size_t i = 0; i < held.size(); ++i) held[i] = eval[i] = static_cast<std::uint32_t>(1000U + i * 3U);
    eval[32] = 7U;  // same first 32 ids, different last token
    const auto dir = std::filesystem::temp_directory_path() / "strix_calib_ids_test";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "e.bin").string();
    cal::MomentSet set(1U);
    (void)set.add_window(eval);
    set.write(path);
    const auto good = slurp(path);
    const auto back = cal::MomentSet::read(path);
    cal::Provenance heldout;
    heldout.add_window(held);
    check(back.provenance().shingles().size() == 2U, "33-id window has two 32-id runs after read");
    check(throws_runtime([&] { cal::require_separation(back.provenance(), nullptr, heldout); }, "share"),
          "last-token-different window still shares its 32-id prefix");
    // SHINGLES zero and the run records dropped, window digest and ids kept.
    const auto head_end = good.find('\n');
    const auto win_end = good.find('\n', head_end + 1U) + 1U + 33U * 4U;
    const auto window_part = good.substr(head_end + 1U, win_end - head_end - 1U);
    spit(path, "strix-calib-moments/v4 0 1 0\n" + window_part);
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "run hashes"), "file with the run records removed refused");
    spit(path, "strix-calib-moments/v4 0 1 1\n" + window_part + good.substr(win_end, 8U));  // one of two runs kept
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "run hashes"), "file with incomplete run records refused");
    spit(path, "strix-calib-moments/v4 0 1 0\n" + window_part.substr(0, window_part.size() - 40U));  // ids missing
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "truncated"), "file with missing ids refused");
    auto edited = good;  // an id edited after the digest was computed
    edited[win_end - 1U] = static_cast<char>(edited[win_end - 1U] ^ 1);
    spit(path, edited);
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "digest"), "file with edited ids refused");
    spit(path, "strix-calib-moments/v3 0 0 0\n");
    check(throws_runtime([&] { (void)cal::MomentSet::read(path); }, "v3"), "v3 moments refused");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// add() publishes sums and counts together: a failure at any preparation stage leaves the serialized set equal to a
// clean capture, and the retry equals a clean capture that never failed (no double count).
void check_add_atomic() {
    namespace cal = strix::weight::calib;
    constexpr std::size_t k = 32, rows = 20;
    const auto x = correlated_rows(rows, k, 21U);
    const auto y = correlated_rows(rows, k, 22U);
    std::vector<std::uint32_t> ids(80);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint32_t>(i * 5U);
    const auto dir = std::filesystem::temp_directory_path() / "strix_calib_atomic_test";
    std::filesystem::create_directories(dir);
    const std::array<std::string, 1> c1{"a.weight"}, c2{"b.weight"};
    const auto build = [&](cal::MomentSet& set, bool second_rows) {
        auto* a = &set.declare("a", static_cast<std::uint32_t>(k), c1);
        auto* b = &set.declare("b", static_cast<std::uint32_t>(k), c2);
        const std::array<cal::InputMoments*, 2> both{a, b};
        const auto w0 = set.add_window(std::span<const std::uint32_t>(ids).subspan(0, 40));
        set.add(w0, both, x.data(), rows);
        set.add(w0, std::span(&a, 1U), y.data(), rows);
        const auto w1 = set.add_window(std::span<const std::uint32_t>(ids).subspan(40, 40));
        if (second_rows) {
            set.add(w1, both, y.data(), rows);
            set.add(w1, std::span(&b, 1U), x.data(), rows);
        }
    };
    const auto bytes = [&](const cal::MomentSet& set) {
        const auto p = (dir / "s.bin").string();
        set.write(p);
        return slurp(p);
    };
    cal::MomentSet clean_before(1U), clean_after(1U);
    build(clean_before, false);
    build(clean_after, true);
    const auto before = bytes(clean_before), after = bytes(clean_after);
    check(before != after, "atomic fixture: rows change the file");
    for (const int stage : {0, 1}) {
        {
            cal::MomentSet set(1U);
            build(set, false);
            auto* a = const_cast<cal::InputMoments*>(set.find("a"));
            auto* b = const_cast<cal::InputMoments*>(set.find("b"));
            const std::array<cal::InputMoments*, 2> both{a, b};
            const auto w1 = set.provenance().windows().size() - 1U;
            const auto feed = [&] {
                set.add(w1, both, y.data(), rows);
                set.add(w1, std::span(&b, 1U), x.data(), rows);
            };
            fail_stage = stage;
            set.set_prepare_hook(&failing_hook);
            bool threw = false;
            try {
                feed();
            } catch (const std::bad_alloc&) {
                threw = true;
            }
            const std::string tag = " (stage " + std::to_string(stage) + ")";
            check(threw, "atomic: injected failure thrown" + tag);
            check(bytes(set) == before, "atomic: state after failure equals the clean capture" + tag);
            check(a->rows == 2U * rows && b->rows == rows, "atomic: row totals untouched after failure" + tag);
            set.set_prepare_hook(nullptr);
            feed();
            check(bytes(set) == after, "atomic: retry equals the clean capture" + tag);
        }
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// A declared evaluation input that no row fed (zero covariance) is refused at pack time, and a zero proxy signal is
// reported as unavailable, never as a zero error.
void check_empty_inputs() {
    namespace cal = strix::weight::calib;
    constexpr std::size_t k = 16;
    std::vector<std::uint32_t> ids(40);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint32_t>(i + 500U);
    const auto dir = std::filesystem::temp_directory_path() / "strix_calib_empty_test";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "eval.bin").string();
    const std::array<std::string, 1> c{"q.weight"};
    cal::MomentSet eval(1U);
    (void)eval.declare("in", static_cast<std::uint32_t>(k), c);
    (void)eval.add_window(ids);
    eval.write(path);
    const auto back = cal::MomentSet::read(path);
    const auto* held = back.find("in");
    check(held != nullptr && held->rows == 0U, "declared unused evaluation input round-trips");
    check(held != nullptr && throws_runtime([&] { cal::require_rows(*held, "evaluation"); }, "in has no attributed rows"),
          "declared unused evaluation input refused at pack time");
    const auto w = random_weights(4U, k, 3U);
    auto q = w;
    for (auto& v : q) v += 0.01F;
    const auto err = gq::output_error(w.data(), q.data(), 4U, k, held->h);
    check(!err.available() && std::isnan(err.relative()), "zero proxy signal is unavailable, not a zero error");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
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
    (void)set.add_window(all.subspan(0, 200));
    (void)set.add_window(all.subspan(200, 100));
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
    // Proxy evaluation windows follow the same rule against the held-out windows (whole window and shared run), and
    // must exist.
    const auto& fit = back.provenance();
    std::vector<std::uint32_t> other(300);
    for (std::size_t i = 0; i < other.size(); ++i) other[i] = static_cast<std::uint32_t>(60000U + i * 31U);
    const std::span<const std::uint32_t> alt(other);
    const auto make_eval = [&](std::span<const std::uint32_t> ids) {
        cal::Provenance p;
        p.add_window(ids);
        return p;
    };
    const auto held_whole = manifest("held_whole", alt.subspan(0, 100));
    const auto held_run = manifest("held_run", alt.subspan(150, 100));
    check(throws_runtime([&] { cal::require_separation(fit, nullptr, manifest("hf", all.subspan(37, 40))); }, "held-out"),
          "fit shingle overlap with held-out refused");
    const auto clean = make_eval(alt.subspan(250, 50));
    bool accepted = true;
    try {
        cal::require_separation(fit, &clean, held_whole);
        cal::require_separation(fit, &clean, held_run);
    } catch (const std::exception&) {
        accepted = false;
    }
    check(accepted, "clean evaluation accepted");
    const auto eval_whole = make_eval(alt.subspan(0, 100));
    check(throws_runtime([&] { cal::require_separation(fit, &eval_whole, held_whole); }, "evaluation statistics share"),
          "evaluation window equal to a held-out window refused");
    const auto eval_run = make_eval(alt.subspan(170, 50));
    check(throws_runtime([&] { cal::require_separation(fit, &eval_run, held_run); }, "evaluation statistics share"),
          "evaluation sharing a 32-id run with a held-out window refused");
    const auto eval_fit = make_eval(all.subspan(210, 50));
    check(throws_runtime([&] { cal::require_separation(fit, &eval_fit, held_whole); }, "calibration and evaluation"),
          "evaluation sharing a run with the fit refused");
    const cal::Provenance none;
    check(throws_runtime([&] { cal::require_separation(fit, &none, held_whole); }, "no evaluation window"),
          "evaluation without provenance refused");
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

void check_nonfinite_feedback() {
    using Storage = wei::Q4G64Storage;
    constexpr std::size_t k = Storage::elements;
    std::vector<float> w(k, 0.01F), factor(k * k, 0.0F);
    std::vector<std::uint8_t> packed(Storage::bytes);
    for (std::size_t i = 0; i < k; ++i) factor[i * k + i] = 1.0F;
    gq::Options opts;
    opts.factor = factor.data();
    factor[0] = 0.0F;
    check(gq::encode_rows<Storage>(w.data(), 1U, k, packed.data(), opts).code == wei::lowbit::EncodeStatus::nonfinite,
          "zero pivot rejected before coding");
    factor[0] = std::numeric_limits<float>::min();
    factor[1] = std::numeric_limits<float>::max();
    w[0] = 0.003F;
    check(gq::encode_rows<Storage>(w.data(), 1U, k, packed.data(), opts).code == wei::lowbit::EncodeStatus::nonfinite,
          "feedback overflow rejected before converting it to a code");
    std::vector<std::uint8_t> dead;
    for (const auto damp : {-0.25, std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            (void)gq::inverse_factor({1.0}, 1U, damp, dead);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "invalid damping rejected before factorization");
    }
}

}  // namespace

int main() {
    check_moments();
    check_attribution();
    check_provenance();
    check_identity_from_ids();
    check_add_atomic();
    check_empty_inputs();
    check_parallel_errors();
    check_nonfinite_feedback();
    check_storage<wei::Q4G64Storage>("q4_g64");
    check_storage<wei::Q6G64Storage>("q6_g64");
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight lowbit gptq tests passed\n";
    return 0;
}

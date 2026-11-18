// Split-K policy (strix/split_k_policy.hpp) and its one CLI resolution (strix/qwen35/prefill_controls.hpp resolve_split_k).
// EXACT invariant: the policy's factor at the EXACT target equals the pre-policy NVFP4 formula on every shape, so the EXACT
// identity keeps its bits. FAST: the table's split factors are the measured ones, keyed by device, never by rows.
#include "strix/split_k_policy.hpp"
#include "strix/qwen35/prefill_controls.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sk = strix::split_k;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

// The NVFP4 split factor before the policy existed (src/cuda_core/dense/nvfp4.cu, target 1000 unless a test set it).
std::uint32_t old_split_k(std::uint32_t target, std::uint32_t out, std::uint32_t in) {
    const std::uint32_t nch = in >> 7, groups = std::max<std::uint32_t>(1, (out / 16 + 15) / 16);
    const std::uint32_t v = (target + groups / 2) / groups;
    return v < 1 ? 1 : (nch < v ? nch : v);  // std::clamp(v, 1, nch) for nch >= 1
}

struct Options {
    sk::Profile nvfp4_split = sk::Profile::exact;
};

constexpr sk::Device kPro{188, 120};
}  // namespace

int main() {
    // 1. EXACT bits: identical factor on every shape (in = 128 .. 65536, out = 1 .. 262144 and the model shapes) and target.
    std::uint64_t n = 0, bad = 0;
    for (std::uint32_t in = 128; in <= 65536; in += 128)
        for (std::uint32_t out = 1; out <= 262144; out = out < 64 ? out + 1 : out + out / 7 + 13)
            for (const std::uint32_t t : {1u, 250u, 500u, 1000u, 2000u, 4096u}) {
                ++n;
                bad += sk::factor(t, out, in) != old_split_k(t, out, in);
            }
    for (const std::uint32_t out : {1024u, 5120u, 6144u, 10240u, 12288u, 17408u, 248320u})
        for (const std::uint32_t in : {5120u, 6144u, 17408u}) {
            ++n;
            bad += sk::factor(sk::target(sk::Profile::exact, kPro, out, in), out, in) != old_split_k(1000, out, in);
        }
    check(bad == 0, "EXACT factor differs from the pre-policy formula");
    std::printf("exact: %llu shape x target cases, %llu differ\n", static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(bad));

    // 2. EXACT profile ignores the device and the table; the test hook target passes through.
    for (const sk::Rule& r : sk::kFastV1) {
        check(sk::target(sk::Profile::exact, r.device, r.out, r.in) == sk::kExactTarget, "exact profile used a FAST row");
        check(sk::target(sk::Profile::exact, r.device, r.out, r.in, 777) == 777, "exact profile lost the test target");
    }

    // 3. FAST on the measured device: the split factors of the sweep (ks columns of the receipt), 1000 elsewhere.
    struct Want { std::uint32_t out, in, ks_exact, ks_fast; };
    const Want want[] = {{12288, 5120, 21, 5},  {10240, 5120, 25, 6}, {6144, 5120, 40, 10}, {5120, 6144, 48, 13},
                         {17408, 5120, 15, 7},  {5120, 17408, 50, 50}, {1024, 5120, 40, 40}, {248320, 5120, 1, 1}};
    for (const Want& w : want) {
        const std::uint32_t ke = sk::factor(sk::target(sk::Profile::exact, kPro, w.out, w.in), w.out, w.in);
        const std::uint32_t kf = sk::factor(sk::target(sk::Profile::fast, kPro, w.out, w.in), w.out, w.in);
        check(ke == w.ks_exact && kf == w.ks_fast, "split factor of a measured shape");
        std::printf("shape %6u x %5u: ks exact %2u fast %2u\n", w.out, w.in, ke, kf);
    }

    // 4. FAST on a device without rows (or another SM count / architecture) is EXACT, and reports the EXACT identity.
    for (const sk::Device d : {sk::Device{}, sk::Device{170, 120}, sk::Device{188, 121}, sk::Device{40, 1151}}) {
        check(!sk::fast_applies(d) && sk::identity(sk::Profile::fast, d) == sk::kExactName, "FAST identity off its device");
        for (const sk::Rule& r : sk::kFastV1)
            check(sk::target(sk::Profile::fast, d, r.out, r.in) == sk::kExactTarget, "FAST row applied off its device");
    }
    check(sk::identity(sk::Profile::fast, kPro) == sk::kFastName && sk::identity(sk::Profile::exact, kPro) == sk::kExactName,
          "identity names");

    // 5. Table hygiene: one row per (device, out, in); every row changes something (target != EXACT) and was a measured arm.
    for (std::size_t i = 0; i < std::size(sk::kFastV1); ++i) {
        const sk::Rule& r = sk::kFastV1[i];
        check(r.target != sk::kExactTarget && (r.target == 250 || r.target == 500), "row target not a measured arm");
        check(sk::factor(r.target, r.out, r.in) != sk::factor(sk::kExactTarget, r.out, r.in), "row does not change ks");
        for (std::size_t j = i + 1; j < std::size(sk::kFastV1); ++j)
            check(!(sk::kFastV1[j].device == r.device && sk::kFastV1[j].out == r.out && sk::kFastV1[j].in == r.in), "duplicate row");
    }

    // 6. One resolution for the decode and the speculative CLI: same words -> same policy; empty = exact while the FAST
    //    gate has not been qualified; a bad word throws and leaves the options unchanged.
    using strix::qwen35::resolve_split_k;
    for (const std::string_view word : {std::string_view{}, std::string_view{"exact"}, std::string_view{"fast"}}) {
        Options decode_cli, spec_cli;
        resolve_split_k(decode_cli, word, false);
        resolve_split_k(spec_cli, word, false);
        check(decode_cli.nvfp4_split == spec_cli.nvfp4_split, "decode / spec CLI resolve differently");
        const sk::Profile want_p = word == "fast" ? sk::Profile::fast : sk::Profile::exact;
        check(decode_cli.nvfp4_split == want_p, "explicit word");
        Options fast_class;
        resolve_split_k(fast_class, word, true);
        const bool default_fast = strix::qwen35::kSplitKFastQualified;
        check(fast_class.nvfp4_split == (word.empty() ? (default_fast ? sk::Profile::fast : sk::Profile::exact) : want_p),
              "default in the FAST class");
    }
    Options keep;
    keep.nvfp4_split = sk::Profile::fast;
    bool threw = false;
    try {
        resolve_split_k(keep, "fast2", true);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("expects") != std::string::npos;
    }
    check(threw && keep.nvfp4_split == sk::Profile::fast, "bad --split-k value");

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}

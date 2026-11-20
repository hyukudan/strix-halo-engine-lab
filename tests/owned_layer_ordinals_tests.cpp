// Owned-layer ordinals (include/strix/owned_layer_ordinals.hpp): the GDN state ledger of a layer split.
// For splits of 1 / 2 / N ranks over a Qwen 3.5 27B-shaped stack (64 layers, full attention every 4th: 48 GDN layers):
// every GDN layer is owned by exactly one rank, ordinals are dense 0..count-1 in layer order inside each range, attention
// and other ranks' layers have none, the counts add up to 48, and the bytes the HIP engine allocates per rank
// (2 x state + 2 x conv + raw qkv / beta / gate per owned GDN layer) drop by the non-owned share: -162.9 MiB at the 24/48 cut.
#include "strix/owned_layer_ordinals.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

constexpr std::uint32_t kLayers = 64, kGdn = 48;
bool is_gdn(std::uint32_t l) { return (l + 1U) % 4U != 0U; }  // layers 3, 7, ... are full attention

// Bytes the HIP engine allocates per owned GDN layer (src/qwen35/hip/engine.hip load(): gdn_all + bak_state, conv_all +
// bak_conv, raw_qkv / raw_beta / raw_gate of kMaxRows rows), 27B dims: 48 value heads, key / value dim 128, conv kernel 4.
constexpr std::uint64_t kGH = 48, kKD = 128, kVD = 128, kGC = 2U * 16U * 128U + 48U * 128U, kRows = 16, kConvK = 4;
constexpr std::uint64_t kPerGdn = 2U * (kGH * kKD * kVD * 4U) + 2U * (kGC * kConvK * 2U) + kRows * kGC * 4U + 2U * kRows * kGH * 4U;

void test_split(const std::vector<std::uint32_t>& cuts) {
    std::vector<int> owners(kLayers, 0);
    std::uint32_t total = 0;
    for (std::size_t r = 0; r + 1 < cuts.size(); ++r) {
        const auto o = strix::owned_layer_ordinals(kLayers, cuts[r], cuts[r + 1], is_gdn);
        std::uint32_t next = 0;
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const bool owned = l >= cuts[r] && l < cuts[r + 1];
            if (owned && is_gdn(l)) {
                check(o.ordinal[l] == next++, "owned GDN ordinals are dense and in layer order");
                ++owners[l];
            } else {
                check(!o.has(l), "attention / other ranks' layers have no ordinal");
            }
        }
        check(o.count == next, "count = owned GDN layers");
        total += o.count;
    }
    check(total == kGdn, "the ranks' counts add up to every GDN layer");
    for (std::uint32_t l = 0; l < kLayers; ++l) check(owners[l] == (is_gdn(l) ? 1 : 0), "each GDN layer owned exactly once");
}

}  // namespace

int main() {
    test_split({0, kLayers});                // one rank: the global ordinals
    test_split({0, 32, kLayers});            // two ranks, 24 / 24 GDN
    test_split({0, 21, 40, kLayers});        // three uneven ranks
    test_split({0, 1, 3, 4, 16, kLayers});   // tiny ranges, one attention-only rank ([3, 4))
    {
        const auto whole = strix::owned_layer_ordinals(kLayers, 0, kLayers, is_gdn);
        const auto half = strix::owned_layer_ordinals(kLayers, 0, 32, is_gdn);
        check(whole.count == kGdn && half.count == 24U, "27B: 48 GDN layers, 24 in [0, 32)");
        const double saved_mib = double(std::uint64_t(whole.count - half.count) * kPerGdn) / double(1U << 20);
        std::printf("per GDN layer %.3f MiB; 24/48 cut saves %.1f MiB per rank\n", double(kPerGdn) / double(1U << 20), saved_mib);
        check(std::fabs(saved_mib - 162.9) < 0.05, "24/48 cut: -162.9 MiB per rank");
        const auto attn_only = strix::owned_layer_ordinals(kLayers, 3, 4, is_gdn);
        check(attn_only.count == 0U, "an attention-only range owns no GDN state");
    }
    bool threw = false;
    try {
        (void)strix::owned_layer_ordinals(kLayers, 10, 10, is_gdn);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty range refused");
    std::printf("%s: owned layer ordinals (%d failure(s))\n", failures ? "FAIL" : "PASS", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

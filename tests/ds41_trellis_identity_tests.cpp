// Host-side trellis accounting and announced identities (ds41_trellis.hpp): site bytes of whole tensors and slices,
// step identities with the pack's K, the configuration identity of single packs and mixes (declared mixA / mixB
// aliases), and the weights namespace without trellis.

#include "strix/ds41_trellis.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void expect_eq(const std::string& got, const std::string& want, const std::string& what) {
    expect(got == want, what + ": got '" + got + "', want '" + want + "'");
}

void site_bytes() {
    namespace d = strix::ds41;
    d::TrellisTensorInfo w1{"layers.0.ffn.shared_experts.w1", 2304, 5120, 5};
    expect(d::trellis_site_bytes(w1) == 2304ULL * 5120 * 5 / 8 + 2ULL * (2304 + 5120), "whole shared w1 K5");
    expect(d::trellis_site_bytes(w1, {d::TrellisSlice::Kind::rows, 1152, 1152}) == 1152ULL * 5120 * 5 / 8 + 2ULL * (1152 + 5120),
           "shexp-tp rows slice");
    d::TrellisTensorInfo w2{"layers.0.ffn.shared_experts.w2", 5120, 2304, 4};
    expect(d::trellis_site_bytes(w2, {d::TrellisSlice::Kind::columns, 0, 1152}) == 5120ULL * 1152 * 4 / 8 + 2ULL * (5120 + 1152),
           "shexp-tp down columns slice K4");
    d::TrellisTensorInfo head{"head", 129280, 5120, 5};
    expect(d::trellis_site_bytes(head, {d::TrellisSlice::Kind::rows, 64640, 64640}) == 64640ULL * 5120 * 5 / 8 + 2ULL * (64640 + 5120),
           "head rank slice");
}

void identities() {
    namespace d = strix::ds41;
    expect_eq(d::trellis_step_identity(5, "head"), "ds41-trellis-k5-v1:head", "step K5");
    expect_eq(d::trellis_step_identity(4, "q_b"), "ds41-trellis-k4-v1:q_b", "step K4");
    expect_eq(d::trellis_shexp_step_identity(5, 5, false), "ds41-trellis-k5-v1:shexp", "shexp K5");
    expect_eq(d::trellis_shexp_step_identity(4, 4, true), "ds41-trellis-k4-v1:shexp-tp", "shexp-tp K4");
    expect_eq(d::trellis_shexp_step_identity(4, 3, false), "ds41-trellis-mix-v1{shexp_w13=K4,shexp_w2=K3}:shexp", "shexp mixed K");

    const std::map<std::string, std::uint32_t> k5{{"q_b", 5}, {"o_a", 5}, {"o_b", 5}, {"shexp_w13", 5}, {"shexp_w2", 5}, {"head", 5}};
    expect_eq(d::trellis_config_identity(k5, {"head"}, false), "ds41-trellis-k5-v1:head", "K5 head");
    expect_eq(d::trellis_config_identity(k5, {"head", "shexp"}, false), "ds41-trellis-k5-v1:shexp,head", "K5 shexp,head (fixed order)");
    expect_eq(d::trellis_config_identity(k5, {"shexp", "head"}, true), "ds41-trellis-k5-v1:shexp,head:shexp-tp", "K5 shexp-tp");
    expect_eq(d::trellis_config_identity(k5, {"head"}, true), "ds41-trellis-k5-v1:head", "shexp-tp without shexp: no suffix");
    expect_eq(d::trellis_config_identity(k5, {"q_a", "kv"}, false), "", "q_a / kv have no engine hook");
    expect_eq(d::trellis_config_identity(k5, {}, false), "", "no class");

    // mix-B: every dense class K4, head K5
    const std::map<std::string, std::uint32_t> mix_b{{"q_b", 4}, {"o_a", 4}, {"o_b", 4}, {"shexp_w13", 4}, {"shexp_w2", 4}, {"head", 5}};
    expect_eq(d::trellis_config_identity(mix_b, {"shexp", "head"}, false),
              "ds41-trellis-mix-v1{head=K5,shexp_w13=K4,shexp_w2=K4}:shexp,head (= ds41-trellis-mixB-v1)", "mix-B shexp,head");
    expect_eq(d::trellis_config_identity(mix_b, {"head"}, false), "ds41-trellis-k5-v1:head", "mix-B pack, head only: K5");
    expect_eq(d::trellis_config_identity(mix_b, {"q_b", "o_a", "o_b"}, false), "ds41-trellis-k4-v1:q_b,o_a,o_b",
              "mix-B pack, attention only: K4");
    // mix-A: q_b, o_a, o_b, shared down K3; shared gate / up K4; head K5
    const std::map<std::string, std::uint32_t> mix_a{{"q_b", 3}, {"o_a", 3}, {"o_b", 3}, {"shexp_w13", 4}, {"shexp_w2", 3}, {"head", 5}};
    expect_eq(d::trellis_config_identity(mix_a, {"q_b", "o_a", "o_b", "shexp", "head"}, false),
              "ds41-trellis-mix-v1{head=K5,o_a=K3,o_b=K3,q_b=K3,shexp_w13=K4,shexp_w2=K3}:q_b,o_a,o_b,shexp,head (= ds41-trellis-mixA-v1)",
              "mix-A all");
    // a mix that is neither: no alias
    const std::map<std::string, std::uint32_t> other{{"shexp_w13", 5}, {"shexp_w2", 5}, {"head", 4}};
    expect_eq(d::trellis_config_identity(other, {"shexp", "head"}, false), "ds41-trellis-mix-v1{head=K4,shexp_w13=K5,shexp_w2=K5}:shexp,head",
              "undeclared mix");
    bool threw = false;
    try {
        (void)d::trellis_config_identity({{"head", 5}}, {"shexp", "head"}, false);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "an enabled class without its K throws");
}

void namespace_without_trellis() {
#if defined(_WIN32)
    _putenv("STRIX_DS41_TRELLIS=");
    _putenv("STRIX_DS41_TRELLIS_MIX=");
#else
    unsetenv("STRIX_DS41_TRELLIS");
    unsetenv("STRIX_DS41_TRELLIS_MIX");
#endif
    expect_eq(strix::ds41::trellis_weights_namespace(), "", "no trellis environment: empty namespace");
}

}  // namespace

int main() {
    site_bytes();
    identities();
    namespace_without_trellis();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ds41 trellis identity tests: OK\n");
    return 0;
}

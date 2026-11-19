#pragma once
// iHC spec the HC tests exercise: 4 streams x 3072, identity comb, F32 fn, bf16 stored streams, post magnitude 2,
// pre / post eps 1e-6, norm eps 1e-5. A model binding keeps its own Spec values in its model package.
#include "strix/hc/hc_contract.hpp"

namespace hc_fixture {

[[nodiscard]] constexpr strix::hc::Spec identity_bf16_spec(strix::hc::Site site) {
    strix::hc::Spec s;
    s.streams = 4;
    s.hidden = 3072;
    s.site = site;
    s.comb = strix::hc::Comb::identity;
    s.fn = strix::hc::FnType::f32;
    s.store = strix::hc::Store::bf16;
    s.post_magnitude = 2.0F;
    s.pre_eps = 1e-6F;
    s.post_eps = 1e-6F;
    s.norm_eps = 1e-5F;
    return s;
}

}  // namespace hc_fixture

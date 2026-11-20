#pragma once
// Hy4-flash HIP adapter (strix-hy4-rank, one node or a layer-split pair) product profile: ONE tier word
// '--profile default|fast|quality' resolved by the shared resolver (strix::profile::resolve, strix/engine_profile.hpp).
// This header is data: one row per lever. The device path has no EXACT class today (A8 routed experts, absorbed MLA,
// A8 rows prefill: compared with the CPU oracle by ids, never called exact), so `default` is the plain invocation
// (prompt position by position, exports nothing). The batched prompt windows (prompt_rows) failed their logit gate on
// prose: the row is UNQUALIFIED, pending in `fast`; it runs only when set explicitly (--prompt-rows, or --allow-pending)
// and every such run is stamped "unqualified: prompt_rows". The layer split (pair
// cut, windows crossing as FP32 rows) changes no bit and is not a profile lever. No TURBO version: turbo is refused.
// Promoting a row after its gate is one edit of `ready` / `provisional` below, never code. Host-only.
#include "strix/engine_profile.hpp"

namespace strix::hy4::profile {
namespace sp = ::strix::profile;

inline constexpr sp::Lever table[] = {
    {.name = "prompt_rows", .env = "", .option = "--prompt-rows", .value = {"", "2048", "", ""}, .builtin = "0",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "2048 (A8 rows prefill windows)", .off = "0",
     .source = "batched prompt: 1K next-32 ids == position by position, window size bitwise (512 = 1024, 2048 = 4096); "
               "logit gate code PASS, prose FAIL (dNLL above the bound): UNQUALIFIED, opt-in only",
     .provisional = {false, false, false, false}},
};

[[nodiscard]] inline const sp::Engine& engine() {
    static const sp::Engine e{"hy4-hip", "STRIX_HY4_PROFILE_ALLOW_PENDING", table};
    return e;
}

}  // namespace strix::hy4::profile

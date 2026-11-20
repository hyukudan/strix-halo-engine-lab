// Host regression for the selected-flash Pf (pipelined loads) policy and the GLM causal list bound that feeds it.
// Pf is bitwise, so these checks pin the launch choice only: T 32 keeps the DS41-qualified 320 threshold, T 16 stays on
// at every bound (no short-list T 16 receipt), explicit STRIX_ATTN_FLASH_PF=0 / 1 wins, and GLM FAST chunks keep
// today's choice whenever the bound is unknown or the chunk ends at or beyond 320 tokens (every 8K chunk past the first
// 320 positions).
#include "attn_core/flash_pf_policy.hpp"
#include "glm/rtxpf/import/glm_dims.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace S = strix::attn_core::sparse;
constexpr int kAuto = -1, kOff = 0, kOn = 1;
// GLM selected-list capacity (glm_prefill.hpp kMaxTokens): top-k pools of kKpool tokens plus the open pool's tail.
constexpr unsigned kGlmListCap = strix::glm::kIdxTopk + strix::glm::kKpool - 1U;
static_assert(kGlmListCap == 2051U);

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
bool glm_pf(int mode, unsigned T, std::uint64_t t0, unsigned rows) {
    return S::flash_pf_decide(mode, T, S::causal_list_bound(t0 + rows, kGlmListCap));
}

void auto_threshold() {
    require(!S::flash_pf_decide(kAuto, 32U, S::kPfMinKeys - 1U), "T 32 auto: 319 keys stays single stage");
    require(S::flash_pf_decide(kAuto, 32U, S::kPfMinKeys), "T 32 auto: 320 keys pipelines");
    require(S::flash_pf_decide(kAuto, 32U, ~0U), "T 32 auto: unknown bound keeps Pf on (previous behaviour)");
    require(!S::flash_pf_decide(kAuto, 32U, 0U), "T 32 auto: empty lists stay single stage");
    for (unsigned keys : {0U, 1U, 128U, 228U, 319U, 320U, 2051U, ~0U})
        require(S::flash_pf_decide(kAuto, 16U, keys), "T 16 auto: on at every bound (" + std::to_string(keys) + ")");
}

void explicit_override() {
    for (unsigned T : {16U, 32U})
        for (unsigned keys : {1U, 319U, 320U, ~0U}) {
            require(S::flash_pf_decide(kOn, T, keys), "STRIX_ATTN_FLASH_PF=1 forces on");
            require(!S::flash_pf_decide(kOff, T, keys), "STRIX_ATTN_FLASH_PF=0 forces off");
        }
    require(std::strcmp(S::flash_pf_reason_text(kOn, 32U, 1U), "on (STRIX_ATTN_FLASH_PF=1)") == 0, "forced-on reason");
    require(std::strcmp(S::flash_pf_reason_text(kOff, 16U, ~0U), "off (STRIX_ATTN_FLASH_PF=0)") == 0, "forced-off reason");
    require(std::strcmp(S::flash_pf_reason_text(kAuto, 32U, 319U), "off (auto: key lists < 320, nothing to overlap)") == 0,
            "T 32 short reason");
    require(std::strcmp(S::flash_pf_reason_text(kAuto, 32U, 320U), "on (auto: key lists >= 320)") == 0, "T 32 long reason");
    require(std::strcmp(S::flash_pf_reason_text(kAuto, 16U, 1U), "on (auto: T 16, no occupancy cost)") == 0, "T 16 reason");
}

void causal_bound() {
    require(S::causal_list_bound(0U, kGlmListCap) == 0U, "empty chunk");
    require(S::causal_list_bound(128U, kGlmListCap) == 128U, "short prompt: rows 0..127 read <= 128 keys");
    require(S::causal_list_bound(kGlmListCap, kGlmListCap) == kGlmListCap, "cap reached exactly");
    require(S::causal_list_bound(8192U, kGlmListCap) == kGlmListCap, "8K: capped at the list capacity");
    require(S::causal_list_bound((std::uint64_t{1} << 32) + 5U, kGlmListCap) == kGlmListCap, "no 32-bit wrap");
    // Unknown bound (~0U callers: kvshard, versions APIs) is clamped to the capacity by mla_flash_rows: Pf on as before.
    require(S::flash_pf_decide(kAuto, 32U, kGlmListCap), "unknown GLM bound keeps Pf on");
}

void glm_chunks() {
    // Short context: a 128-token prompt in one chunk. T 32 now skips Pf; T 16 (FAST default) is unchanged.
    require(!glm_pf(kAuto, 32U, 0U, 128U), "GLM T 32 short prompt: single stage");
    require(glm_pf(kAuto, 16U, 0U, 128U), "GLM T 16 short prompt: Pf kept");
    require(!glm_pf(kAuto, 32U, 0U, 319U) && glm_pf(kAuto, 32U, 0U, 320U), "GLM T 32 crossover at chunk end 320");
    require(glm_pf(kAuto, 32U, 256U, 64U), "GLM T 32: a later short chunk ending at 320 pipelines");
    // 8K: every chunk that ends at or past 320 keeps Pf on for both key blocks, exactly as before the binding.
    for (unsigned chunk : {256U, 512U, 1024U, 2048U, 4096U, 8192U})
        for (std::uint64_t t0 = 0; t0 < 8192U; t0 += chunk)
            if (t0 + chunk >= S::kPfMinKeys)
                for (unsigned T : {16U, 32U})
                    require(glm_pf(kAuto, T, t0, chunk), "GLM 8K chunk keeps Pf (chunk " + std::to_string(chunk) + ", t0 " +
                                                            std::to_string(t0) + ", T " + std::to_string(T) + ")");
}
}  // namespace

int main() {
    try {
        auto_threshold();
        explicit_override();
        causal_bound();
        glm_chunks();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    std::cout << "flash_pf_policy_tests: PASS\n";
    return 0;
}

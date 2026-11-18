#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include "strix/cuda_core/attention_policy.hpp"
#include "strix/cuda_core/operand_cache_plan.hpp"

namespace strix::qwen35 {
// Host-only resolution shared by inspection and execution. Scheduling controls preserve
// arithmetic; attn8 is explicitly bit-changing and retains the engine's own identity/gates.
// Validate every value before changing the options, including inspection-only invocations.
template<class Options>
void resolve_flash_controls(Options& options, std::string_view pipe, std::string_view tail, std::string_view attn8) {
    if (!pipe.empty() && pipe != "pad2" && pipe != "swz3") throw std::runtime_error("--flash-pipe expects pad2|swz3");
    if (!tail.empty() && tail != "off" && tail != "split") throw std::runtime_error("--flash-tail expects split|off");
    if (!attn8.empty() && attn8 != "on" && attn8 != "off") throw std::runtime_error("--prefill-attn8 expects on|off");
    if (!pipe.empty()) options.flash_pipe = pipe == "swz3" ? cuda_core::FlashPipe::swz3 : cuda_core::FlashPipe::pad2;
    if (!tail.empty()) options.flash_tail = tail == "split" ? cuda_core::FlashTail::split : cuda_core::FlashTail::off;
    if (!attn8.empty()) options.prefill_attn8 = attn8 == "on";
}

// Resident expanded prefill operands (bitwise neutral; see cuda_engine.hpp prefill_operand_cache). Empty = keep the default (auto).
template<class Options>
void resolve_operand_cache(Options& options, std::string_view mode, std::string_view cap_mib) {
    cuda_core::operand_cache::Mode m = options.prefill_operand_cache;
    if (!mode.empty() && !cuda_core::operand_cache::parse_mode(std::string(mode).c_str(), &m))
        throw std::runtime_error("--prefill-operand-cache expects auto|on|off");
    std::uint32_t cap = options.prefill_operand_cache_cap_mib;
    if (!cap_mib.empty()) {
        std::uint64_t v = 0;
        for (const char ch : cap_mib) {
            if (ch < '0' || ch > '9' || v > 0xFFFFFFFFULL / 10) throw std::runtime_error("--prefill-operand-cache-cap-mib expects 0..4294967295");
            v = v * 10 + static_cast<std::uint64_t>(ch - '0');
        }
        if (v > 0xFFFFFFFFULL) throw std::runtime_error("--prefill-operand-cache-cap-mib expects 0..4294967295");
        cap = static_cast<std::uint32_t>(v);
    }
    options.prefill_operand_cache = m;
    options.prefill_operand_cache_cap_mib = cap;
}

// The bit-changing tier levers (prefill_arith, prefill_gdn, prefill_attn8, split_k) are resolved by the ONE product
// profile resolver of this engine (strix/profiles/qwen_dense_cuda.hpp: --profile exact|fast + per-lever overrides), shared
// by the decode and the speculative CLI. The scheduling-only controls above stay here.
}  // namespace strix::qwen35

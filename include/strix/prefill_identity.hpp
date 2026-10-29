#pragma once

// Prefill arithmetic options that change bits relative to the FP16 prefill identity (fast-prefill-f16wmma-v1 with
// the exact HC output chains, the token-serial GDN recurrence and the ordered FP32 indexer chains). Every READY option is ON by
// default and individually switchable off; one master switch restores the plain FP16 identity at once.
//   environment: STRIX_PREFILL_EXACT_FP16=1 (all off), STRIX_PREFILL_<NAME>=0 (one off), e.g. STRIX_PREFILL_HC_OUTPUT_WMMA=0
//   command line: --prefill-exact-fp16, --prefill-disable <name>[,<name>...] (names as below, lower case)
// Environment and startup restrictions combine monotonically. The active set names the prefill identity (prefill_identity_name/hash),
// which receipts and the serving capability endpoint report.

#include "strix/decode_numerical_profile.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix {

enum class PrefillOption : unsigned {
    hc_output_wmma,     // HC output: FP16 low values and coefficients, WMMA accumulation (was ordered FP32 chains)
    index_proj_wmma,    // Candidate BF16 indexer projections; explicit opt-in until qualified
    count
};

struct PrefillOptionInfo {
    std::string_view name;  // command line / config name
    std::string_view env;   // STRIX_PREFILL_<NAME>
    std::string_view tag;   // identity name component
};

inline constexpr std::array<PrefillOptionInfo, static_cast<unsigned>(PrefillOption::count)> kPrefillOptions{{
    {"hc_output_wmma", "STRIX_PREFILL_HC_OUTPUT_WMMA", "hcout16"},
    {"index_proj_wmma", "STRIX_PREFILL_INDEX_PROJ_WMMA", "idxproj16"},
}};

// Options whose implementation is built in and enabled by default. An option joins this mask only when it is READY;
// Unqualified candidates stay default OFF and require an explicit environment opt-in.
inline constexpr unsigned kPrefillOptionsReady = 1U << static_cast<unsigned>(PrefillOption::hc_output_wmma);

// Implemented candidates may be exercised explicitly without promotion to READY.
inline constexpr unsigned kPrefillOptionsCandidates = 1U << static_cast<unsigned>(PrefillOption::index_proj_wmma);

// Candidate index arithmetic is not qualified for MTP or nested scheduling.
inline constexpr bool index_wmma_unqualified_route(unsigned mask, unsigned fast_components) {
    return (fast_components & kFastPrefillQsa) &&
        (mask & (1U << static_cast<unsigned>(PrefillOption::index_proj_wmma)));
}

namespace detail {
struct PrefillIdentityOverride {
    bool set = false;
    bool exact = false;
    unsigned disabled = 0U;
};
inline PrefillIdentityOverride& prefill_identity_override() {
    static PrefillIdentityOverride o;
    return o;
}
}  // namespace detail

inline unsigned prefill_option_bit(PrefillOption option) { return 1U << static_cast<unsigned>(option); }

// Command line: --prefill-exact-fp16 and --prefill-disable a,b. Call before the first prefill (the mask is read per call).
inline void set_prefill_identity_override(bool exact, unsigned disabled) {
    auto& o = detail::prefill_identity_override();
    o.set = true;
    o.exact = o.exact || exact;
    o.disabled |= disabled;
}
inline unsigned parse_prefill_option_list(std::string_view list) {
    if (list.empty() || list.back() == ',')
        throw std::invalid_argument("prefill option list must contain names without a trailing comma");
    unsigned mask = 0U;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto item = list.substr(0, comma);
        bool found = false;
        for (unsigned i = 0; i < kPrefillOptions.size(); ++i)
            if (item == kPrefillOptions[i].name) { mask |= 1U << i; found = true; }
        if (!found) throw std::invalid_argument("unknown prefill option '" + std::string(item) + "'");
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    }
    return mask;
}

// Resolve only the supplied environment and restrictions; no process-global state.
template<class Get>
inline unsigned resolve_prefill_options(Get get, bool exact = false, unsigned disabled = 0U) {
    const char* master = get("STRIX_PREFILL_EXACT_FP16");
    if (exact || (master && master[0] == '1')) return 0U;
    unsigned mask = kPrefillOptionsReady & ~disabled;
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i) {
        const char* value = get(kPrefillOptions[i].env.data());
        const auto bit = 1U << i;
        if (value && std::string_view(value) == "1" && (kPrefillOptionsCandidates & bit) && !(disabled & bit)) mask |= bit;
        if (value && value[0] == '0') mask &= ~bit;
    }
    return mask;
}

// The active set: environment, then the command line override.
inline unsigned prefill_options_active() {
    unsigned mask = resolve_prefill_options([](const char* name) { return std::getenv(name); });
    const auto& o = detail::prefill_identity_override();
    if (o.set) mask = o.exact ? 0U : mask & ~o.disabled;
    return mask & (kPrefillOptionsReady | kPrefillOptionsCandidates);
}
inline bool prefill_option_enabled(PrefillOption option) { return (prefill_options_active() & prefill_option_bit(option)) != 0U; }

// "fp16-v1" for the plain FP16 identity, else "fp16-v1+<tag>+<tag>..." in enumerator order.
inline std::string prefill_identity_name(unsigned mask) {
    std::string name = "fp16-v1";
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i)
        if (mask & (1U << i)) { name += '+'; name += kPrefillOptions[i].tag; }
    return name;
}
inline std::string prefill_identity_name() { return prefill_identity_name(prefill_options_active()); }
// FNV-1a 64 of the identity name.
inline std::uint64_t prefill_identity_hash(unsigned mask) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char c : prefill_identity_name(mask)) { h ^= static_cast<unsigned char>(c); h *= 1099511628211ULL; }
    return h;
}
inline std::uint64_t prefill_identity_hash() { return prefill_identity_hash(prefill_options_active()); }

// Bind the enabled set into compiled execution, then inherited tiled/lineage identities.
inline std::uint64_t bind_prefill_option_identity(std::uint64_t arithmetic, unsigned mask) {
    for (const std::uint64_t word : {UINT64_C(0x5052464f50545331),
             static_cast<std::uint64_t>(mask), prefill_identity_hash(mask)})
        for (unsigned i = 0; i < 8; ++i) {
            arithmetic ^= (word >> (8U * i)) & 0xffU;
            arithmetic *= UINT64_C(1099511628211);
        }
    return arithmetic;
}
// One arithmetic builder for host compilation and native execution. GSQ is
// supplied from the model contract, never inferred from unrelated process state.
inline std::uint64_t prefill_arithmetic_identity(const NumericalProfilePair& pair, unsigned mask, bool gsq) {
    if (!numerical_profile_fast_prefill(pair.prefill)) return numerical_profile_pair_identity(pair);
    auto identity = bind_prefill_option_identity(numerical_profile_pair_identity(pair), mask);
    if (gsq && (pair.fast_components & kFastPrefillMoe))
        for (const std::uint64_t word : {UINT64_C(0x4753514d4f455046), moe_q4k_a8::gsq_fast_prefill_moe_identity()})
            for (unsigned i = 0; i < 8; ++i) {
                identity ^= (word >> (8U * i)) & 0xffU;
                identity *= UINT64_C(1099511628211);
            }
    return identity;
}
// Legacy selector shares the public arithmetic controls. Explicit fixture kernel IDs are separate.
inline unsigned governed_hc_output_kernel(unsigned requested, unsigned mask) {
    return requested == 2U && !(mask & prefill_option_bit(PrefillOption::hc_output_wmma)) ? 1U : requested;
}

}  // namespace strix

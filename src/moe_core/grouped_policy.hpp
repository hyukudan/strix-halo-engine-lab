#pragma once

// Shared grouped-prefill storage policy. Engines opt into XOR64 explicitly;
// the default leaves their padded/linear storage contract unchanged.
#if defined(__HIPCC__)
#define STRIX_MOE_GROUPED_HD __attribute__((host,device))
#else
#define STRIX_MOE_GROUPED_HD
#endif

namespace strix::moe_core {

template <class Types> struct GroupedF16Fragments;

// PipeTail: partial row blocks of the pipelined MT8 item run on item_pipe with a row guard on the store instead of
// the scalar core (same fragments, K-ascending chains, split ranges and stores: bitwise). The decode form stays the
// engine's Traits::Decode<Bits>::kPacked. Needs inputs padded to the item's MT x 16 rows (unguarded A loads).
template <bool Xor64 = false, bool PipeTail = false>
struct GroupedPrefillPolicy {
    static constexpr bool xor64 = Xor64;
    static constexpr bool kPipeTail = PipeTail;
};

template <class Policy = GroupedPrefillPolicy<>, unsigned PaddedPitch = 72U>
struct GroupedLdsRing {
    static constexpr unsigned pitch = Policy::xor64 ? 64U : PaddedPitch;
    STRIX_MOE_GROUPED_HD static constexpr unsigned offset(unsigned row, unsigned half) {
        if constexpr (Policy::xor64)
            return row*64U+((half/8U)^(row%8U))*8U+half%8U;
        else return row*PaddedPitch+half;
    }
};

// Format identity remains separate from the engine's decoder implementation.
// A declared case does not admit a new format or provide a missing decoder.
enum class GroupedWeightFormat {
    q4_k, q5_k, q2_k, q5_1, q8_0, q2_0,
    iq2_xxs, iq2_s, iq3_xxs, iq3_s, iq4_xs, iq4_nl
};

template <GroupedWeightFormat Format, class Operations>
struct GroupedWeightDecode : Operations {
    static constexpr GroupedWeightFormat format = Format;
};

template <class Decode, class Policy = GroupedPrefillPolicy<>>
struct GroupedPrefillCore : Policy {
    using WeightDecode = Decode;
    template <class Types> using Fragments = GroupedF16Fragments<Types>;
    template <unsigned PaddedPitch = 72U>
    using Ring = GroupedLdsRing<Policy, PaddedPitch>;
};

} // namespace strix::moe_core

#undef STRIX_MOE_GROUPED_HD

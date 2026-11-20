#pragma once
// Shared MLA latent policy for the sparse flash core (attn_core/sparse_flash_wmma.hpp), contract
// include/strix/attn/mla_contract.hpp steps (2)-(4): one cache row per token [kLat latent | kRope rope] (bf16), the
// absorbed query as its bf16 operand image [H][kLat + kRope] (query_op: the absorb GEMM epilogue stores it), FP32
// normalised latent output [H][kLat]. Selection is a trait: List (the indexer's per-row positions) or Window (computed
// by mla::window_count / window_key over a ring of `ring` rows), so the DSA and the SWA layers of a model run the SAME
// kernel family; the learned sink is a trait of the core's normalisation. The gate is NOT here: it acts on the v space
// after the out projection (contract step 6).
//
// Spec S (model data, e.g. a model's binding): static constexpr unsigned kHeads (per launch: 16 | 32), kLat, kRope;
// static constexpr float kScale; static constexpr strix::attn::mla::Select kSelect; static constexpr bool kSink;
// static constexpr unsigned kGroups: head groups per launch (heads of the rank's slice / kHeads; 1 | 2 | 4), one row
// index per (token, group): row = token * kGroups + group (head groups are contiguous in qt / out / sinks).
#include "attn_core/sparse_flash_wmma.hpp"
#include "strix/attn/mla_contract.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::sparse::mla {

namespace contract = strix::attn::mla;

template <class S> struct LatentPolicy {
    struct Args {
        const std::uint16_t* cache;   // rows [kLat + kRope] bf16; List: indexed by position; Window: by position % ring
        const std::uint32_t* lists;   // List: [tokens][list_stride] ascending positions
        const std::uint32_t* list_n;  // List: [tokens]
        unsigned list_stride;         // List
        unsigned t0;                  // Window: position of token 0
        unsigned window;              // Window: keys per token at most (511 earlier + self = 512)
        unsigned ring_mask;           // Window: ring rows - 1 (power of two >= window - 1 + chunk tokens)
        const std::uint16_t* qt;      // [tokens][groups][kHeads][kLat + kRope] bf16 operand image
        const float* sinks;           // [groups][kHeads] (kSink)
        float* out;                   // [tokens][groups][kHeads][kLat]
    };
    static constexpr unsigned kHeads = S::kHeads, kDim = S::kLat, kRope = S::kRope;
    static constexpr tile::Operand kOp = tile::Operand::bf16;
    static constexpr float kScaleLog2 = S::kScale * 1.4426950408889634F;
    static constexpr bool kWindow = S::kSelect == contract::Select::Window;
    static constexpr unsigned kGroups = S::kGroups;
    static_assert(kGroups == 1U || kGroups == 2U || kGroups == 4U, "head groups: a power of two (row / kGroups is a shift)");

    __device__ static unsigned token(unsigned r) { return r / kGroups; }
    __device__ static unsigned count(const Args& a, unsigned r) {
        if constexpr (kWindow) return contract::window_count(a.t0 + token(r), a.window);
        else return a.list_n[token(r)];
    }
    __device__ static unsigned key(const Args& a, unsigned r, unsigned j) {
        if constexpr (kWindow) {
            const unsigned t = a.t0 + token(r);
            return contract::window_key(t, contract::window_count(t, a.window), j);
        } else {
            return a.lists[static_cast<std::size_t>(token(r)) * a.list_stride + j];
        }
    }
    __device__ static const std::uint16_t* key_row(const Args& a, unsigned, unsigned pos) {
        const unsigned slot = kWindow ? (pos & a.ring_mask) : pos;
        return a.cache + static_cast<std::size_t>(slot) * (kDim + kRope);
    }
    __device__ static const std::uint16_t* value_row(const Args& a, unsigned r, unsigned pos) { return key_row(a, r, pos); }
    __device__ static const std::uint16_t* query_op(const Args& a, unsigned r) {
        return a.qt + static_cast<std::size_t>(r) * kHeads * (kDim + kRope);
    }
    __device__ static float* out(const Args& a, unsigned r) { return a.out + static_cast<std::size_t>(r) * kHeads * kDim; }
    __device__ static float sink(const Args& a, unsigned r, unsigned head)
        requires(S::kSink)
    {
        return a.sinks[(r % kGroups) * kHeads + head];
    }
};

}  // namespace strix::attn_core::sparse::mla

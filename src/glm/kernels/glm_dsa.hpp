#pragma once
// GLM-5.3-Flash Tier A device kernels, batch 2: indexer (R4, R11, R12, R5, section 1.4) and absorbed MLA attention
// (R13). Equal to src/glm/host/glm_host.cpp bit for bit given the same inputs (gate K).
// One sequence slot per call row here (decode); caches are per slot with a fixed capacity in tokens.
#include "glm/kernels/glm_tier_a.hpp"

namespace strix::glm::dev {

struct DsaSlot {                 // device cache of one sequence for one DSA layer
    std::uint16_t* latent;       // [cap][512] bf16
    std::uint16_t* ikey;         // [cap][128] bf16
    std::uint16_t* igate;        // [cap][128] bf16
    float* pooled;               // [cap / 4][128] fp32
    unsigned cap;
};
struct IndexerDev {
    const std::uint16_t* wk;            // [128][4096] F16
    const std::uint16_t* gate;          // [128][4096] F16
    const std::uint16_t* weights_proj;  // [32][4096] F16
    const std::uint16_t* k_norm_w;      // [128] F16
    const std::uint16_t* k_norm_b;      // [128] F16
    const float* ape;                   // [4][128]
};
// Appends token position t (the slot's current length) of `rows` slots: latent [rows][512] (normed), x [rows][4096].
// key_scratch / gate_scratch: [rows][128]. Completes pool t / 4 when (t + 1) % 4 == 0. C3/C4 -> flag.
void dsa_append(const DsaSlot* slots, const unsigned* t, const float* latent, const float* x, const IndexerDev& w,
                float* key_scratch, float* gate_scratch, unsigned* flag, unsigned rows, hipStream_t s);
// Canonical token list of the query at position t of one slot (section 1.4): tokens [<= 2051], n.
// qI [32 * 128], x [4096]; scratch: wh [32], scores [cap / 4], keys/flags workspace [cap / 4 * 2] uint32.
void indexer_select(const DsaSlot& slot, unsigned t, const float* qI, const float* x, const IndexerDev& w, float* wh,
                    float* scores, std::uint32_t* work, std::uint32_t* tokens, std::uint32_t* n, hipStream_t s);
// The selection + tail stage alone, on precomputed ordered keys [P] (test hook for P up to 32768).
void select_tokens(const std::uint32_t* keys, unsigned P, unsigned t, std::uint32_t* tokens, std::uint32_t* n, hipStream_t s);
// R13 for `heads` rank heads: q [heads][256]; kv_b rank rows [heads * 512][512] F16; qt scratch [heads][512];
// tokens / n from indexer_select; o [heads][256].
void mla_attention(const DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                   const std::uint16_t* kv_b, unsigned heads, float* qt, float* o, hipStream_t s);

}  // namespace strix::glm::dev

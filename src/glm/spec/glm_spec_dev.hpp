#pragma once
// GLM speculative decoding, device kernels for the verify rows of ONE sequence slot (design notes
// section 3). Bitwise to the host reference: kda_verify == spec::KdaVerify (= the plain kda_step sequence), and
// dsa_append_rows == R sequential host dsa_append. Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero.
#include "glm/kernels/glm_dsa.hpp"
#include "glm/kernels/glm_tier_a.hpp"

namespace strix::glm::dev {

// KDA verify with deferred replay, one slot, `k.heads` rank heads, C = heads * 128.
//   Replay (the rows committed by the previous call, 0 <= n_rep <= 8): qkv_rep [n_rep][3C] (raw pre-conv inputs, for the
//   conv shift) and scr_rep [n_rep][heads][4][128] + beta_rep [n_rep][heads] (that call's prep outputs, reused as is).
//   New rows (0 <= n_new <= 8): qkv [n_new][3C], f / gate [n_new][C], b [n_new][heads].
//   In/out: S [heads][128][128] and conv_state [3C][3] are the COMMITTED state: on return they hold the state after the
//   replay rows (stored once); the new rows run on registers and are never stored.
//   Out: scr_new / beta_new (keep them: next call's replay inputs), o [n_new][heads][128] scratch, y [n_new][C].
//   tree: 0 reads STRIX_GLM_SPEC_KDA_TREE once (only "2" opts in); 1/2 explicitly select a schedule for qualification.
//   Non-finite: replay rows -> flag (committed checkpoint C2); new rows -> row_flag[r] (the caller aborts only if a
//   committed row is flagged; rejected rows' flags are ignored). n_new = 0 is a flush.
void kda_verify(const KdaDev& k, const float* qkv_rep, const float* scr_rep, const float* beta_rep, unsigned n_rep,
                const float* qkv, const float* f, const float* gate, const float* b, unsigned n_new, float* S,
                float* conv_state, float* scr_new, float* beta_new, float* o, float* y, unsigned* flag, unsigned* row_flag,
                hipStream_t s, unsigned tree = 0);

// DSA append of R consecutive positions t0 .. t0+R-1 of ONE slot (R <= 8): stores every row (latent, key, gate), then
// completes the pools closed inside the block (pure functions of their 4 stored tokens: equal to R sequential appends).
// latent [R][512] (normed), x [R][4096]; key_scratch / gate_scratch [R][128]; row_flag[r] per row (C3/C4).
// Rollback needs no device work: the frontier moves back and later appends overwrite the rejected positions; a pool
// closed by a rejected row is recomputed when the pool really closes.
void dsa_append_rows(const DsaSlot& slot, unsigned t0, unsigned R, const float* latent, const float* x,
                     const IndexerDev& w, float* key_scratch, float* gate_scratch, unsigned* row_flag, hipStream_t s);

}  // namespace strix::glm::dev

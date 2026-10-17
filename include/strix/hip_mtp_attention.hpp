#pragma once
#include "strix/hip_mtp_resources.hpp"
namespace strix {
// Complete metadata admission for BOTH stages, before any proposal enqueue.
// Uses the incumbent BF16 indexer/Q8 attention kernels and independent MTP KV
// and raw-index storage; predictor position is hidden-source position.
void preflight_hip_mtp_attention(const MtpNativeContext&);
// Teacher cache reconstruction only: exact incumbent index-key, K/V projection,
// key norm/RoPE and BF16 stores for c.position. No query, index selection,
// attention output, host synchronization or publication. Only that row changes.
void preflight_hip_mtp_cache_rows(const MtpNativeContext&);
void launch_hip_mtp_cache_rows(const MtpNativeContext&,void* stream);
void launch_hip_mtp_indexer(const MtpNativeContext&,void* stream);
void launch_hip_mtp_sparse_attention(const MtpNativeContext&,void* stream);
// Draft-only: same stages with the Q/K/V and output Q8 projections on the
// wave-per-row draft GEMV (different summation order; drafts only).
void launch_hip_mtp_sparse_attention_draft(const MtpNativeContext&,void* stream);
// Raw exact Q8 projection used by the attention stages (benchmark reference).
void enqueue_hip_mtp_attention_q8_linear(const std::uint16_t* input,const void* q8,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream);
void enqueue_hip_mtp_attention_bf16_linear(const std::uint16_t* input,const void* bf16,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream);
} // namespace strix

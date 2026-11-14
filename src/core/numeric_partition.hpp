#pragma once

namespace strix::core {
// Frozen numeric partition policy v1. These reference buckets preserve existing
// S values; physical tile tuning must never modify them.
template<class Weights> inline bool numeric_partition_medium(const Weights& w, int rows) {
    if(rows<32 || w.codebook!=2) return false;
    if(rows>=256) return (w.bits==6 && w.input==5120 && w.output==1280) ||
        (w.bits==5 && w.input==4096 && w.output==1024);
    if(w.bits==6) return w.input==5120 && w.output==1280;
    if(w.bits!=5) return false;
    return (w.input==1280 && w.output==32768) ||
        (w.input==8192 && w.output==5120) || (w.input==5120 && w.output==2304) ||
        (w.input==2304 && w.output==5120) || (w.input==4096 && w.output==1024);
}
template<class Weights> inline int numeric_partition_bucket_rows(const Weights& w, int rows) {
    return numeric_partition_medium(w,rows) || (rows>=256 && w.output>=2304) ||
        (rows>=64 && w.output>=32768 && w.bits==5 && w.codebook==2)?64:16;
}
} // namespace strix::core

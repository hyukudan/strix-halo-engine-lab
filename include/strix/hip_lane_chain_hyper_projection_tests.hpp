#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdint>
namespace strix::hc_projection_diagnostic {
struct Shape {std::uint32_t width=0,streams=0,rank=0,rows=1;};
struct Span {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Diagnostic only. opw=0 selects the unchanged reference (rows8 controls which).
// opw=8/16/32 and prefetch=1/2 select prototype instantiations. Inputs are original
// normalized FP32 and Q8/F32 weights; callers retain all owners through drain.
void enqueue(Shape,Span normalized,Span down,Span inject,Span low,Span injection,
    std::uint32_t opw,std::uint32_t prefetch,bool reference_rows8,hipStream_t);
// Diagnostic reassociated FP32 projection. This is not the ordered reference
// numerical identity and is not selected by any production launcher.
enum class SplitKVariant { p1_r4, p1_r2, p2_r4, p1_r4_fp64 };
// Only p2_r4 uses partial: rows*16*(rank + (inject ? streams : 0))*4 bytes.
// Every owner remains alive until stream completion. The other variants require
// an empty partial span; no implicit allocation or device synchronization occurs.
void enqueue_splitk(Shape,Span normalized,Span down,Span inject,Span low,
    Span injection,Span partial,SplitKVariant,hipStream_t);
}

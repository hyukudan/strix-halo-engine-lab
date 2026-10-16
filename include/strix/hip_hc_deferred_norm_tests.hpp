#pragma once
#include "strix/hip_lane_chain_hyper_projection_tests.hpp"

namespace strix::hc_deferred_norm_diagnostic {
using Span = hc_projection_diagnostic::Span;
// Canonical BF16 raw input, F32 norm, original Q8 down/up and optional F32
// injection. No allocation or synchronization. Every owner must survive drain.
// inverse_audit is optional 81*4 floats (80 down blocks plus one injection);
// without injection it is 80*4 floats. It records each block's four inverses.
// scaled_audit optionally records 320 down and (when present) four injection
// values before their nonlinear epilogues, separating tree checks from expf.
void enqueue(Span raw,Span norm,Span down,Span inject,Span up,Span inverse,
    Span low,Span injection,Span output,Span inverse_audit,Span scaled_audit,float epsilon,hipStream_t);
}

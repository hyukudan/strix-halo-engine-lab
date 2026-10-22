#pragma once

#include "strix/hip_ds41_formats.hpp"

namespace strix::ds41 {

// STRIX_DS41_ROUTES_FAST=1 (default off): enqueue_build_routes takes build_routes_fast_kernel, the same
// integer outputs without the serial union/most loops of one thread.
bool routes_fast_enabled();
// Wider serial builder lives here to preserve the legacy formats GPU function bytes.
void enqueue_build_routes_wide_serial(const BuildRoutesView& view, void* stream);
void enqueue_build_routes_fast(const BuildRoutesView& view, void* stream);

} // namespace strix::ds41

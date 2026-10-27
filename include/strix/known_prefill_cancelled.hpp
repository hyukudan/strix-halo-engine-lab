#pragma once

namespace strix {
// A drained prefill was interrupted before publication; its session requires a fresh reset.
struct KnownPrefillCancelled {};
} // namespace strix

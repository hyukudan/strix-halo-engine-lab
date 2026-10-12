#pragma once

#include <functional>

namespace strix {

// Recovery must observe the original binding both sides of a successful drain.
// A failed API call or matching device index alone is not a completion proof.
// Callbacks throw on failure; this helper never turns an exception into proof.
inline bool prove_native_setup_quiescence(
    const std::function<void()>& require_original_binding,
    const std::function<void()>& synchronize_original_device) noexcept {
    if (!require_original_binding || !synchronize_original_device) return false;
    try {
        require_original_binding();
        synchronize_original_device();
        require_original_binding();
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace strix

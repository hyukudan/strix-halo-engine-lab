#include "strix/prompt_terminal_diagnostics.hpp"
namespace strix {
namespace { bool certification_enabled=false; }
void set_prefix_batch_certification_enabled(bool enabled) noexcept { certification_enabled=enabled; }
bool prefix_batch_certification_enabled() noexcept { return certification_enabled; }
} // namespace strix

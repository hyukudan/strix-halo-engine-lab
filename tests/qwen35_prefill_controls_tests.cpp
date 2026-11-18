#include "strix/qwen35/prefill_controls.hpp"
#include "strix/qwen35/hip_engine.hpp"
#include <stdexcept>
#include <string>
void require(bool value) { if (!value) throw std::runtime_error("prefill control mismatch"); }
struct Options {
    strix::cuda_core::FlashPipe flash_pipe = strix::cuda_core::FlashPipe::pad2;
    strix::cuda_core::FlashTail flash_tail = strix::cuda_core::FlashTail::split;
    bool prefill_attn8 = false;
};
int main() {
    using strix::qwen35::resolve_flash_controls;
    Options o;
    resolve_flash_controls(o, "", "", "");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    resolve_flash_controls(o, "swz3", "off", "on");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::swz3 && o.flash_tail == strix::cuda_core::FlashTail::off && o.prefill_attn8);
    resolve_flash_controls(o, "pad2", "split", "off");
    require(o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    for (int which = 0; which < 3; ++which) {
        bool failed = false;
        try { resolve_flash_controls(o, which == 0 ? "bad" : "swz3", which == 1 ? "bad" : "off", which == 2 ? "bad" : "on"); }
        catch (const std::runtime_error& e) { failed = std::string(e.what()).find("expects") != std::string::npos; }
        require(failed && o.flash_pipe == strix::cuda_core::FlashPipe::pad2 && o.flash_tail == strix::cuda_core::FlashTail::split && !o.prefill_attn8);
    }
    // HIP 27B window rows: EXACT wide windows only over IU4 large dense projections; 16-row windows and the FAST class
    // keep the requested width; a non-IU4 projection resolves the EXACT wide request to the 16-row exact form.
    using strix::qwen35::resolve_prefill_rows;
    static_assert(resolve_prefill_rows(64U, true, true) == 64U && resolve_prefill_rows(64U, true, false) == 16U);
    static_assert(resolve_prefill_rows(512U, false, false) == 512U && resolve_prefill_rows(16U, true, false) == 16U);
    static_assert(resolve_prefill_rows(0U, true, false) == 0U && resolve_prefill_rows(17U, true, false) == 16U);
    require(resolve_prefill_rows(strix::qwen35::HipEngineOptions{}.prefill_rows, strix::qwen35::HipEngineOptions{}.wide_exact, false) == 16U);
    require(resolve_prefill_rows(strix::qwen35::HipEngineOptions{}.prefill_rows, strix::qwen35::HipEngineOptions{}.wide_exact, true) == 64U);
}

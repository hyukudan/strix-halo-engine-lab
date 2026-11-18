#include "strix/cuda/qwen_hc.hpp"

#include <cstdio>
#include <initializer_list>
#include <stdexcept>

namespace hc = strix::cuda::qwen_hc;
static void require(bool condition) {
    if (!condition) throw std::runtime_error("HC workspace admission regression");
}
template <class F> static void rejects(F call) {
    try { call(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid staged workspace accepted");
}
int main() try {
    alignas(16) std::uint16_t input[16]{};
    alignas(16) float norm[8]{};
    hc::HcMixArgs a;
    a.rows = 65; a.mma = true;
    // Existing output-only fixture: normalized is supplied, input/norm are null.
    require(!hc::hc_compact_workspace(a, 4U, true, true));
    require(!hc::hc_compact_workspace(a, 2U, true, true));
    a.gates = norm;
    require(!hc::hc_compact_workspace(a, 12U, true, true));
    a.gates = nullptr; a.input = input; a.norm = norm;
    require(!hc::hc_compact_workspace(a, 15U, true, true));
    a.workspace_layout = hc::HcWorkspaceLayout::automatic;
    require(hc::hc_compact_workspace(a, 15U, true, true));
    require(!hc::hc_compact_workspace(a, 15U, false, true));
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    for (bool mma : {false, true}) {
        a.mma = mma; a.rows = 16;
        a.input = input + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input; a.norm = norm + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input; a.norm = norm;
        require(hc::hc_compact_workspace(a, 15U, true, true));
    }
    a.mma = false; a.rows = 1;
    require(!hc::hc_compact_workspace(a, 15U, true, true));
    require(hc::hc_compact_workspace(a, 15U, true, false));
    // Rows-form cap: an exact automatic call the rows forms can take (rows <= cap) is never compact; above the cap the old rule
    // holds unless exact_full; exact_full never changes the FAST (mma) admission or an explicit layout.
    for (unsigned rows : {9U, 12U, 16U}) {
        a.rows = rows;
        require(hc::hc_compact_workspace(a, 15U, true, true));                // defaults = the admission before the cap
        require(!hc::hc_compact_workspace(a, 15U, true, true, 16U));
        require(hc::hc_compact_workspace(a, 15U, true, false, 16U));          // decode forms off: no rows form to protect
        a.mma = true;
        require(hc::hc_compact_workspace(a, 15U, true, true, 16U, true));
        a.mma = false;
    }
    for (unsigned rows : {17U, 2019U, 8076U}) {
        a.rows = rows;
        require(hc::hc_compact_workspace(a, 15U, true, true, 16U, false));
        require(!hc::hc_compact_workspace(a, 15U, true, true, 16U, true));
        require(!hc::hc_compact_workspace(a, 15U, true, false, 16U, true));
        require(!hc::hc_compact_workspace(a, 15U, false, true, 16U, false));  // compact lever off: full, as before
        a.mma = true;
        require(hc::hc_compact_workspace(a, 15U, true, true, 16U, true));
        require(!hc::hc_compact_workspace(a, 15U, false, true, 16U, true));
        a.mma = false;
        a.workspace_layout = hc::HcWorkspaceLayout::full;
        require(!hc::hc_compact_workspace(a, 15U, true, true, 16U, false));
        a.workspace_layout = hc::HcWorkspaceLayout::compact;
        require(hc::hc_compact_workspace(a, 4U, true, true, 16U, true));     // explicit staged format wins over exact_full
        a.workspace_layout = hc::HcWorkspaceLayout::automatic;
        rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true, 16U, true); });
    }
    a.rows = 1;
    a.workspace_layout = hc::HcWorkspaceLayout::compact;
    for (unsigned stage : {1U, 2U, 4U, 15U})
        require(hc::hc_compact_workspace(a, stage, false, true));
    a.gates = norm;
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    a.gates = nullptr; a.input = input + 1;
    rejects([&] { (void)hc::hc_compact_workspace(a, 2U, true, true); });
    a.input = input; a.norm = nullptr;
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    // Exact fused output epilogue form: default (-1) follows the lever; an explicit per-call choice wins over it.
    {
        hc::HcMixArgs e;
        require(e.output_batched == -1);
        require(hc::hc_output_batched(e, true) && !hc::hc_output_batched(e, false));
        e.output_batched = 0;
        require(!hc::hc_output_batched(e, true) && !hc::hc_output_batched(e, false));
        e.output_batched = 1;
        require(hc::hc_output_batched(e, true) && hc::hc_output_batched(e, false));
    }
    std::puts("HC full/staged/compact/alignment admission + output epilogue form passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}

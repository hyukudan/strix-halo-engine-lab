// Row liveness rule (include/strix/prefill_row_liveness.hpp): which prompt rows each layer must run.
#include "strix/prefill_row_liveness.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

namespace lv = strix::prefill_liveness;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

// Gemma 4 E4B producer map (src/gemma4/model/gguf.cpp): layers 0..23 own K/V, 24..41 read layers 22 / 23.
constexpr unsigned e4b_layers = 42;
constexpr bool e4b_owns(unsigned l) { return l < 24; }
constexpr std::array<std::uint32_t, 5> dflash_taps{5, 11, 17, 23, 35};

void gemma_e4b() {
    // Engine chunk, no drafter: nothing reads a row past layer 23's caches, not even the last row.
    {
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
        expect(p.cut() && p.full_layers == 24 && p.cache_only_last && !p.tail && p.ends_at_state(), "E4B plain chunk stops at layer 23 caches");
        expect(lv::skipped_row_layers(p, 512) == 512ULL * 18, "E4B plain chunk removes 18 layers of every row");
    }
    // A pass whose last row is read (logits now or deferred): the last row runs layers 24..41 alone.
    {
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = {}, .every_row = false, .head_row = true, .tail_row_invariant = true});
        expect(p.cut() && p.full_layers == 24 && p.cache_only_last && p.tail && !p.ends_at_state(), "E4B read chunk keeps a one-row tail");
        expect(lv::skipped_row_layers(p, 512) == 511ULL * 18, "E4B read chunk removes 18 layers of all rows but the last");
    }
    // The same pass on kernels whose row bytes depend on the call's rows: no cut.
    {
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = {}, .every_row = false, .head_row = true, .tail_row_invariant = false});
        expect(!p.narrows() && p.full_layers == e4b_layers, "row-dependent kernels run the full stack");
    }
    // DFlash taps read layer 35 for every row: rows run to layer 35, layers 36..41 are dead; layer 23 stays whole.
    {
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = dflash_taps, .every_row = false, .head_row = false, .tail_row_invariant = false});
        expect(p.cut() && p.full_layers == 36 && !p.cache_only_last && !p.tail, "DFlash taps keep rows to layer 35");
        expect(lv::skipped_row_layers(p, 512) == 512ULL * 6, "DFlash chunk removes 6 layers");
    }
    // A tap on the last owner's output keeps that layer whole.
    {
        constexpr std::array<std::uint32_t, 4> low{5, 11, 17, 23};
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = low, .every_row = false, .head_row = false, .tail_row_invariant = false});
        expect(p.cut() && p.full_layers == 24 && !p.cache_only_last, "a tap at layer 23 keeps its output");
    }
    // Final-norm rows of every row (Eagle / MTP drafters, all-row logits, a pipeline seam): no cut.
    {
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = {}, .every_row = true, .head_row = true, .tail_row_invariant = true});
        expect(!p.narrows() && lv::skipped_row_layers(p, 512) == 0, "every-row readers run the full stack");
    }
    // Tap indices past the stack read nothing (a smaller model than the tap table).
    {
        constexpr std::array<std::uint32_t, 1> past{99};
        const auto p = lv::plan(e4b_layers, e4b_owns, {.taps = past, .every_row = false, .head_row = false, .tail_row_invariant = false});
        expect(p.full_layers == 24, "out-of-stack taps are ignored");
    }
}

// Stacks whose every layer owns state (Qwen hybrids: attention K/V or GDN recurrent state; DS41, GLM, MiniMax): no
// layer sits above the last owner. Only the last layer's post-cache work of unread rows is dead (the headless rule).
void every_layer_owns() {
    constexpr auto all = [](unsigned) { return true; };
    for (const unsigned layers : {40U, 43U, 48U, 62U, 64U}) {
        const auto engine = lv::plan(layers, all, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
        expect(!engine.cut() && engine.cache_only_last && engine.full_layers == layers && engine.ends_at_state(),
               "all-owner stack: last layer cache-only, nothing above");
        expect(lv::skipped_row_layers(engine, 512) == 0, "all-owner stack removes no layer");
        const auto read = lv::plan(layers, all, {.taps = {}, .every_row = false, .head_row = true, .tail_row_invariant = true});
        expect(!read.cut() && read.cache_only_last && read.tail, "all-owner stack: last row finishes the last layer alone");
    }
}

void edges() {
    constexpr auto none = [](unsigned) { return false; };
    expect(!lv::plan(0, none, {}).narrows(), "empty stack");
    expect(!lv::plan(8, none, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false}).narrows(),
           "a stack without state or readers is left alone");
    // A stateful layer only at the bottom.
    const auto p = lv::plan(8, [](unsigned l) { return l == 0; }, {.taps = {}, .every_row = false, .head_row = true, .tail_row_invariant = true});
    expect(p.full_layers == 1 && p.cache_only_last && p.tail && lv::skipped_row_layers(p, 3) == 2ULL * 7, "bottom owner");
    static_assert(lv::plan(42, e4b_owns, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false}).full_layers == 24);
}

}  // namespace

int main() {
    gemma_e4b();
    every_layer_owns();
    edges();
    if (failures) return EXIT_FAILURE;
    std::cout << "prefill row liveness: all checks passed\n";
    return EXIT_SUCCESS;
}

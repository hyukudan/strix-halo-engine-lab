#pragma once
// Stage graph: the one declaration a model writes (host only, backend neutral).
// A declaration is DATA: a table of StageDecl (per layer template, plus the stages before the first and after the last
// layer), the per-layer trait words, and the pack's format rules. From it the shared planners derive, with no model
// code: the launch plan (launch_plan.hpp: today's launch list in check mode, the fused plan) and the rows forms
// (rows_plan.hpp: step, verify(R), window). Nothing here executes; an executor binds Launch entries.
// Home: shared (host shared / plan). src/model_graph keeps the description role and converts into this
// (src/model_graph/stage_bridge.hpp to_stage_graph).
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace strix::graph {

enum class Op : std::uint8_t {
    linear,       // y = W x (one weight operand; stacked items read `read` of `items`: routed experts)
    attend,       // attention over cached keys (operand: heads x (qk + v) dims, read = keys)
    select,       // sparse key selection (score, top-k list)
    route,        // router top-k over the logits of all experts (a reduction over the whole input)
    combine,      // elementwise sum / weighted sum of vectors (residual, expert merge, hyper-connection collapse / expand)
    norm,         // RMS norm (a reduction over the vector)
    mix,          // hyper-connection mix coefficients (a reduction over the streams)
    inject,       // side-input injection into the residual (Engram)
    head,         // vocabulary projection
    elementwise,  // rope, rounding, gates, activations, quantize per block
    store,        // cache row write (state_out)
    copy,         // device-to-device copy
    clear,        // memset
    upload,       // host -> device
    download,     // device -> host
    gather,       // table row gather (embedding, Engram rows)
    argmax        // greedy pick
};
enum class Rows : std::uint8_t { per_row, cross_row };
enum class Split : std::uint8_t { none, replicate, by_heads, by_experts, by_k, by_vocab };
enum class Format : std::uint8_t { none, f32, f16, bf16, q8_0, q4_k, q6_k, iu4 };
// Position regime of a stage (sparse selected attention): always; while the causal prefix is dense (slot < select_after);
// past it; on the slot that closes a pool group ((slot + 1) % group == 0).
enum class Pos : std::uint8_t { any, prefix_dense, prefix_select, group_close };

using Traits = std::uint32_t;
namespace trait {
inline constexpr Traits attn_dense = 1U << 0, attn_gdn = 1U << 1, attn_window = 1U << 2, attn_indexed = 1U << 3,
                        ffn_dense = 1U << 4, ffn_moe = 1U << 5, inject = 1U << 6;
}

// Bytes of one row of `width` elements (throws for formats whose block geometry this layer does not know).
[[nodiscard]] std::uint64_t row_bytes(Format f, std::uint64_t width);

struct Operand {
    std::string_view tensor = {};           // tensor-map symbol (suffix after the layer prefix); empty = no weight read
    std::uint64_t rows = 0, cols = 0;  // per item: output rows x input width (attend: heads x (qk + v) dims)
    std::uint64_t items = 1, read = 1;  // stacked items and items read per token row (experts 512 / 16; attend: keys)
};
[[nodiscard]] constexpr Operand weights(std::string_view tensor, std::uint64_t rows, std::uint64_t cols, std::uint64_t items = 1,
                                        std::uint64_t read = 1) {
    return Operand{tensor, rows, cols, items, read};
}
// Attention work (no weight read): heads x (qk + v dims) per key, `count` keys per row.
[[nodiscard]] constexpr Operand keys(std::uint64_t heads, std::uint64_t dims, std::uint64_t count) {
    return Operand{{}, heads, dims, count, count};
}

struct StageDecl {
    std::string_view id = {};     // unique in its table
    Op op = Op::elementwise;
    std::string_view form = {};   // op form (elementwise / combine / store kind), data for the fusion rules
    std::string_view entry = {};  // shared entry this stage binds today (check mode)
    std::string_view site = {};   // sublayer scope ("attn", "ffn", "seam", "edge", ...)
    std::string_view in[4] = {};
    std::string_view out = {};
    Operand operand{};
    std::string_view numeric = {};  // numeric contract name
    Rows rows = Rows::per_row;
    Split split = Split::none;
    std::uint8_t leaves = 0;     // 0 or 4 (canonical partial tree)
    bool reduce = false;         // needs the whole input vector (norm, weighted merge, top-k): no producer epilogue
    std::uint8_t kernels = 1;    // kernels this stage adds to today's entry call (0 = shares the previous stage's kernel)
    std::string_view call = {};       // stages of one table with the same non-empty call run as ONE entry call today
    Traits when_all = 0, when_none = 0;  // layer guard (trait words)
    Pos pos = Pos::any;
};

struct FormatRule {
    std::string_view tensor = {};  // empty = any tensor (default rule)
    Format format = Format::none;
    std::span<const std::uint32_t> layers = {};  // empty = every layer
};

struct StageGraph {
    std::span<const StageDecl> before = {}, layer = {}, after = {};
    std::vector<Traits> layers = {};              // one trait word per trunk layer
    std::span<const FormatRule> formats = {};     // later rules win
    std::uint32_t select_after = 0, group = 0;  // sparse attention position regimes (0 = none)
};

// One instantiated stage: table, index in it, layer (-1 = before / after), resolved format.
struct Instance {
    const StageDecl* decl = nullptr;
    std::int32_t layer = -1;
    Format format = Format::none;
};

[[nodiscard]] bool active(const StageGraph& g, const StageDecl& s, std::int32_t layer, std::uint32_t slot);
[[nodiscard]] Format format_of(const StageGraph& g, std::string_view tensor, std::int32_t layer);
// The stages one decode step at `slot` executes, in declaration order (before, layers, after).
[[nodiscard]] std::vector<Instance> instances(const StageGraph& g, std::uint32_t slot);
// Weight bytes read by one launch over `rows` token rows sharing it (items read once per distinct item; rows = 1: step).
[[nodiscard]] std::uint64_t bytes(const Instance& s);
[[nodiscard]] std::uint64_t macs(const Instance& s, std::uint32_t rows);
// Layer trait words from flag lists (data in, words out): listed layers (or, with except, the others) get the trait.
struct TraitList {
    Traits trait = 0;
    std::span<const std::uint32_t> layers = {};
    bool except = false;  // true: every layer NOT listed gets the trait
};
[[nodiscard]] std::vector<Traits> layer_traits(std::uint32_t layers, std::span<const TraitList> lists);
// Throws std::invalid_argument on: duplicate ids in a table, empty out on a computing op, a linear / head without operand,
// leaves not 0 / 4, a stage active on no layer, an unknown format for an operand.
void validate(const StageGraph& g);

}  // namespace strix::graph

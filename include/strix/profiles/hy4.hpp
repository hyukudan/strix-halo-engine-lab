#pragma once
// Hy4-flash HIP adapter (strix-hy4-rank, one node or a layer-split pair) product profile: ONE tier word
// '--profile default|fast|quality' resolved by the shared resolver (strix::profile::resolve, strix/engine_profile.hpp)
// per REGIME (device, topology, prompt length), like strix/profiles/qwen_dense_cuda.hpp. This header is data plus the
// regime filter: one row per lever, one `qualified` row per (lever, value, device, topology, gated prompt range) with its
// receipt. The device path has no EXACT class today (A8 routed experts, absorbed MLA: compared with the CPU oracle by
// ids, never called exact), so `default` is the plain invocation (prompt position by position, exports nothing).
// The batched prompt windows (prompt_rows) are FAST: qualified by gate v2 (route-replayed reference, one-ulp null,
// per-window limits) on gfx1151 one node over the prompt range of the receipts below; outside a qualified regime the
// fast cell is pending (runs only with --allow-pending or an explicit --prompt-rows, stamped "unqualified"). The layer
// split (pair cut, windows crossing as FP32 rows) changes no bit and is not a profile lever. No TURBO version: turbo is
// refused. Promoting a regime after its gate is one `qualified` row, never code. Host-only.
#include "strix/engine_profile.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::hy4::profile {
namespace sp = ::strix::profile;

inline constexpr sp::Lever table[] = {
    {.name = "prompt_rows", .env = "", .option = "--prompt-rows", .value = {"", "2048", "", ""}, .builtin = "0",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "2048 (A8 rows prefill windows)", .off = "0",
     .source = "batched prompt: 1K next-32 ids == position by position, window size bitwise (512 = 1024, 2048 = 4096); "
               "FAST gate v2 per regime: see `qualified`",
     .provisional = {false, false, false, false}},
    {.name = "head_q8", .env = "", .option = "--head-q8", .value = {"", "1", "", ""}, .builtin = "0",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, true, true, false},
     .target = "1 (Q8_0 copy of the BF16 head for the decode head, hy4-head-bf16-q8_0-v1)", .off = "0",
     .source = "shared core::convert_bf16_q8_0 (ggml q8_0 rule) at load; FAST gate v2 per regime: see `qualified`",
     .provisional = {false, false, false, false}},
};

// Regime of a run, known before the forward loads. An unknown device resolves every FAST cell to pending.
enum class Device : std::uint8_t { unknown, gfx1151 };
enum class Topology : std::uint8_t { one_node, pair };
struct Regime {
    Device device = Device::unknown;
    Topology topology = Topology::one_node;
    std::uint64_t prompt_tokens = 0;  // the prompt of this run (an entry point that resolves earlier: the longest it admits)
};
[[nodiscard]] inline Device device_of(std::string_view arch) {  // hipDeviceProp_t::gcnArchName; features after ':' ignored
    return arch.substr(0, arch.find(':')) == "gfx1151" ? Device::gfx1151 : Device::unknown;
}
[[nodiscard]] inline const char* device_name(Device d) { return d == Device::gfx1151 ? "gfx1151" : "unknown device"; }
[[nodiscard]] inline const char* topology_name(Topology t) { return t == Topology::pair ? "pair (layer split)" : "one node"; }

// Where a FAST cell is qualified: prompts of at most max_prompt_tokens on (device, topology), with its gate receipt.
struct Qualified {
    std::string_view lever;
    std::string_view value;
    Device device;
    Topology topology;
    std::uint64_t max_prompt_tokens;
    std::string_view receipt;
};
inline constexpr std::uint64_t kNoPromptBound = ~std::uint64_t{0};
inline constexpr Qualified qualified[] = {
    {"prompt_rows", "2048", Device::gfx1151, Topology::one_node, 4096,
     "gate v2 (route replay, one-ulp null): 1,000 held-out rows over en/zh/es/py/rs, KL 0.00189, top-1 0.982, "
     "dNLL -0.00174 +- 0.00304; long4k KL 0.00082"},
    // Long prompts: BASE reference by verify rows R=8 (bitwise step()) carried across tickets by state images, arms with the
    // reference routes replayed, a one-ulp null per cut from the reference image at N-1024 (hy4_fast2, build 8ecdaeabfcab).
    {"prompt_rows", "2048", Device::gfx1151, Topology::one_node, 32568,
     "gate v2 (route replay, one-ulp null): 2,000 held-out rows after prompts of 2,048..32,568 tokens, code 6 windows "
     "(py) + prose 4 windows (licence texts), KL 0.00106, top-1 0.993, dNLL -0.00065 +- 0.00130, all windows PASS or "
     "UNRESOLVED (prose-4096 dNLL), every class final"},
    // The head copy changes only the decode head's weights (no prompt dependence: no bound); the layer split changes no
    // bit, so the pair's head stage computes the same logits as one node.
    {"head_q8", "1", Device::gfx1151, Topology::one_node, kNoPromptBound,
     "gate v2: 5 windows, 1,000 held-out rows, KL 0.00002, top-1 0.999, dNLL +0.00032; decode 57.13 -> 55.79 ms/token"},
    {"head_q8", "1", Device::gfx1151, Topology::pair, kNoPromptBound, "same head numerics as one node (the layer split changes no bit)"},
};

[[nodiscard]] inline const Qualified* qualification(std::string_view lever, std::string_view value, const Regime& regime,
                                                    std::span<const Qualified> rows) {
    const Qualified* best = nullptr;
    for (const auto& q : rows)
        if (q.lever == lever && q.value == value && q.device == regime.device && q.topology == regime.topology &&
            regime.prompt_tokens <= q.max_prompt_tokens && (best == nullptr || q.max_prompt_tokens > best->max_prompt_tokens))
            best = &q;
    return best;
}

// The static engine (rows as declared, every FAST cell pending): option names for take_options; runs resolve below.
[[nodiscard]] inline const sp::Engine& engine() {
    static const sp::Engine e{"hy4-hip", "STRIX_HY4_PROFILE_ALLOW_PENDING", table};
    return e;
}

// Owns the per-regime rows the resolution points into (non-movable: entries hold pointers).
struct Resolved {
    Regime regime;
    std::vector<sp::Lever> rows;
    sp::Engine engine;
    sp::Resolution resolution;
    Resolved() = default;
    Resolved(const Resolved&) = delete;
    Resolved& operator=(const Resolved&) = delete;
    [[nodiscard]] const sp::Entry* find(std::string_view lever) const { return resolution.find(lever); }
};

// The CLI and the server call this one function with the same Inputs and Regime (ONE RESOLVER).
[[nodiscard]] inline std::unique_ptr<Resolved> resolve(const Regime& regime, const sp::Inputs& inputs,
                                                       std::span<const Qualified> rows = qualified) {
    auto out = std::make_unique<Resolved>();
    out->regime = regime;
    constexpr auto fast = static_cast<unsigned>(sp::Name::fast);
    for (const auto& lever : table) {
        sp::Lever row = lever;
        row.ready[fast] = qualification(lever.name, lever.value[fast], regime, rows) != nullptr;
        out->rows.push_back(row);
    }
    out->engine = engine();
    out->engine.table = std::span<const sp::Lever>(out->rows);
    out->resolution = sp::resolve(out->engine, inputs);
    for (auto& entry : out->resolution.entries) {
        if (entry.status != sp::Status::pending && entry.status != sp::Status::unqualified) continue;
        std::string where;
        for (const auto& q : rows)
            if (q.lever == entry.lever->name)
                where += std::string(where.empty() ? "" : "; ") + device_name(q.device) + " / " + topology_name(q.topology) +
                         (q.max_prompt_tokens == kNoPromptBound ? std::string() : " / prompt <= " + std::to_string(q.max_prompt_tokens));
        entry.reason = "not qualified for " + std::string(device_name(regime.device)) + " / " + topology_name(regime.topology) +
                       " / prompt " + std::to_string(regime.prompt_tokens) +
                       (where.empty() ? std::string(": no gate receipt yet") : "; qualified only for " + where);
    }
    return out;
}

}  // namespace strix::hy4::profile

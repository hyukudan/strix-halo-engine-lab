#pragma once

#include "strix/compiled_speculative_verify.hpp"
#include "strix/execution_arena.hpp"
#include "strix/execution_resources.hpp"
#include "strix/execution_runtime.hpp"
#include "strix/serial_session.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace strix {

struct CompiledLaunchPlan;

// The caller retains all row registries/configs and every resource owner until
// completion or quiesce_or_fail_stop. The runtime invokes each selected R1
// launcher's pure preflight on all rows; a missing callback fails admission.
// preflight_plan optionally adds owner policy. preflight_scope proves an owned eager stream and
// deferred-status parity, and null graph-position arguments. complete drains/checks status before publication.
struct KnownWindowExecutionHooks {
    // Tiled prefill only: a block whose logits are never read (more prompt follows) skips the
    // terminal head step entirely; the result then carries no logit rows.
    bool skip_head = false;
    // Admission only: every check and preflight, the memo/slot updates, then return before the
    // lineage, PLE lookup or any launch. Nothing is enqueued and the session is not changed.
    bool preflight_only = false;
    // Metadata/config/resource owners cannot change via aliases during a call.
    std::function<void()> preflight_plan;
    std::function<void()> preflight_scope;
    std::function<void()> complete;
    std::function<void()> quiesce_or_fail_stop;
    // Closed QSA implementation identities admitted by the native owner; a
    // test may inject a recording QSA identity. Runtime position uses its fixed
    // default implementation identity, never the verify-window implementation.
    std::span<const std::uint64_t> position_reader_implementations;
    // Captured native windows opt in explicitly. Records one real whole chunk
    // only after complete succeeds; default legacy prefill labeling is intact.
    bool record_actual_window_lineage = false;
    // Owner-held memo of the launch plan whose per-step metadata preflight already passed.
    // Only positions and tokens change between windows of one owner, and every launcher
    // still validates its own context before enqueueing, so repeat windows skip the
    // ~1,750-context preflight (about 8 ms of host time before the first launch).
    std::vector<const void*>* preflight_memo = nullptr;
    // Lineage recipe identities of a memoized plan (keyed like preflight_memo): the recipe
    // hashes only the plan, the compiled window and the selected entries, all fixed per key.
    std::vector<std::pair<const void*, std::array<std::uint64_t, 2>>>* recipe_memo = nullptr;
    // Owner-held launcher preflights already proven, one slot per (step, row): the preflight
    // function and its user data. The contexts of a (step, row) come from the owner's compiled
    // window, not from its launch plan, so a new plan of the same owner (an MTP capture copy
    // that wraps one step) re-runs only the slots whose launcher differs, plus its own
    // structural plan checks; the owner-level admission already passed with the first plan.
    std::vector<std::array<std::uintptr_t, 2>>* preflight_slots = nullptr;
    // Owner-held index of the compiled window's row arena views (alive and unmodified as long as
    // the owner): their buffer lookups are answered from it instead of being indexed per window.
    const ArenaLookupIndex* arena_index = nullptr;
    // Called just before each step of a launched window is enqueued (never during admission), with all
    // row contexts. Returning true means the hook enqueued that step itself (the loop then only records it).
    std::function<bool(std::size_t, const std::vector<std::vector<ExecutionDispatchContext>>&)> before_step;
};

struct KnownWindowR2Entry {
    KnownWindowR2Variant variant = KnownWindowR2Variant::none;
    void* config = nullptr;
    void (*preflight)(std::span<const ExecutionDispatchContext, 2>, void*) = nullptr;
    void (*launch)(std::span<const ExecutionDispatchContext, 2>, void*) = nullptr;
    // Actual closed native implementation recipe, never a callback address.
    // Required only for selected entries when lineage recording is enabled.
    std::uint64_t implementation_identity = 0;
};

// Separate closed wide entry: R2 callbacks retain their fixed-span contract.
struct KnownWindowRowsEntry {
    KnownWindowR2Variant variant = KnownWindowR2Variant::none;
    void* config = nullptr;
    void (*preflight)(std::span<const ExecutionDispatchContext>, void*) = nullptr;
    void (*launch)(std::span<const ExecutionDispatchContext>, void*) = nullptr;
    std::uint64_t implementation_identity = 0;
};

struct KnownWindowResult {
    EagerExecutionResult execution;
    std::vector<ExecutionBufferAddress> logits;
    std::uint64_t logical_projection_weight_bytes_avoided = 0;
    std::uint32_t projection_launches_saved = 0;
    // Row/operator work represented by actual dispatches, excluding omitted
    // terminal heads. Promoted row groups still represent every source row.
    std::uint32_t logical_row_steps = 0;
    std::uint32_t r2_dispatches = 0;
    std::uint32_t row_loop_dispatches = 0;
    std::uint32_t shared_rows_dispatches = 0;
    std::uint32_t gdn_rows_dispatches = 0;
    std::uint32_t routed_expert_union_dispatches = 0;
    std::uint32_t hyper_mix_rows_dispatches = 0;
    std::uint32_t qsa_rows_dispatches = 0;
    std::uint32_t lm_head_rows_dispatches = 0;
    std::array<std::uint32_t, known_window_r2_variant_slots> r2_variant_dispatches{};
    std::vector<std::uint32_t> logit_row_indices;
    std::uint32_t omitted_lm_heads = 0;
};

// Compatibility entry for two KNOWN input tokens only. This API neither predicts a future greedy token
// nor authorises speculative reuse. Row launch plans must be freshly built from
// window.row_plan[r], preserving each serial incumbent kernel id.
[[nodiscard]] KnownWindowResult execute_known_prefix_window_eager(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& window_scratch, SerialSessionContext& session,
    std::array<std::uint32_t, 2> tokens,
    std::array<const CompiledLaunchPlan*, 2> row_launch,
    std::span<const KnownWindowR2Entry> r2_entries,
    const KnownWindowExecutionHooks& hooks);

// Complete causal known R=2/4/8 window, with explicit output policy and
// wide/shared/GDN entries. All launchers, including omitted LM heads, undergo
// full admission. logits are compact and indexed by logit_row_indices; an
// omitted row never returns a stale scratch address as a produced output.
[[nodiscard]] KnownWindowResult execute_known_prefix_window_eager(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ExecutionRuntimeResources& serial_resources,
    const ExecutionMemoryBinding& window_scratch, SerialSessionContext& session,
    std::span<const std::uint32_t> tokens,
    std::span<const CompiledLaunchPlan* const> row_launch,
    std::span<const KnownWindowR2Entry> r2_entries,
    std::span<const KnownWindowRowsEntry> rows_entries,
    const KnownWindowExecutionHooks& hooks);

struct SpeculativeVerifyExecutionResult {
    EagerExecutionResult execution;
    std::uint64_t base_position = 0;
    std::uint64_t post_position = 0;
    std::uint64_t base_committed_tokens = 0;
    std::uint64_t post_committed_tokens = 0;
};

[[nodiscard]] SpeculativeVerifyExecutionResult
execute_compiled_speculative_verify_eager(
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionRuntimeResources& resources,
    SerialSessionContext& session,
    std::span<const std::uint32_t> candidate_tokens,
    EagerExecutionDispatcher& dispatcher);

}  // namespace strix

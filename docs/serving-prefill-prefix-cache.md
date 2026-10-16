# Serving prefill and prefix-cache composition

This source path composes the existing complete seven-component serving cache with an explicitly configured native prefill producer. It does not change the default serial-only longest-prefix lookup, arithmetic identities, or cache policy defaults.

`SerialPrefixProducerRecipe` is constructed from validated state-only and terminal compiled window/prefill plans for the same original serial execution. It independently checks each recorded producer interval: chunks start at position zero, interior full chunks use the state-only plan, an exactly terminal full chunk uses the terminal plan, and any unpadded suffix uses original serial rows. Token, model-content, runtime, layout and frontier checks remain required.

Producer-aware lookup is exact-length only. It never restores a cached terminal chunk as an interior state-only chunk of a longer request. A request with a valid but different history, such as a small public work budget causing an initial serial row before later chunks, is a cache miss and its capture is skipped without rewriting history. Serial-only consumers continue to reject chunk histories.

The caller sets `max_history_intervals` from `recipe.maximum_history_intervals(max_prefix_tokens)` before computing the shared pool reservation and allocating it. The default remains one interval. Reservation includes declared extra interval payloads for retained entries and the capture candidate; snapshot and capture-peak budgets remain separate explicit limits. This is declared payload accounting, not measured driver or allocator overhead.

Only a successfully selected finite terminal scalar can be captured. The cached token remains pending and is emitted on an exact hit without another target call. State-only requests and zero-generation requests do not create scalar-frontier entries. Restore retains an immutable cache handle and independent allocator owner through complete seven-component transfer and publication; transfer failures retain the existing poison/drain contract.

Source fixtures cover compiled recipe validation, changed producer identities, geometry and policy mismatches, proper-prefix misses, default serial rejection, restored lineage, real native small/tiled prefill capture and restore, complete state and generation comparisons, pending-token consumption, and benign noncanonical-history skips. These fixtures have not been executed as part of this source increment. Native correctness, resource and performance qualification remain separate.

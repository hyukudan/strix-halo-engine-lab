# Prompt lookup

## Explicit occupied-KV snapshots

`--prompt-lookup-snapshot occupied-kv-v1 --prompt-lookup-snapshot-bytes N` selects the bounded compact snapshot format only for the lookup route. N is an explicit positive payload bound checked before snapshot allocation. The default remains full-components with its existing behavior; serving prefix caches keep complete snapshots. A byte budget alone never silently changes the format.

Occupied-KV snapshots preserve the occupied attention interval and every other required state component and lineage. Attention-KV bytes at or beyond the restored next-position frontier are indeterminate; comparing the complete flat physical arena would overstate this contract. The receipt reports the policy, declared bound and only the latest snapshot payload from a completed cycle, not cumulative bytes, peak memory, GPU timing or numerical qualification.

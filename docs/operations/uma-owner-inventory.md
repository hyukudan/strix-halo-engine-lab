# Retained UMA owner inventory

`build_uma_owner_inventory` accepts caller-observed views and retains their actual
allocation leases. Equal nonzero issuer/allocation/generation triples describe
aliases of one backing; copies use distinct triples even when contents match.
Virtual addresses and content hashes never establish shared physical backing.
Inconsistent extents or kinds for the same lease, partial keys, duplicate view
identities, invalid slices and arithmetic overflow are rejected.

The snapshot reports logical view bytes, interval-union coverage, alias bytes,
unique known reservation extents and a conservative declared reservation upper
bound. Any unknown backing makes the complete unique reservation total unknown.
Allocated storage and file mappings remain separate. Retained file extent is
not the number of physically resident pages. An opaque strong reference retains
an owner; the caller still attests the live allocation generation.

System MemTotal/MemAvailable and process RSS, GTT and peak observations remain
separate optional values, tied to an observation identity. RSS and GTT are never
added. No manufacturer capacity or physical residency is inferred. This module
does not admit a GPU allocation or replace the existing memory admission planner.

Status: source implementation and analytical source fixtures only. Compilation,
execution and allocator/runtime inventory integration remain pending. No memory
saving, additional model capacity or performance improvement is established.

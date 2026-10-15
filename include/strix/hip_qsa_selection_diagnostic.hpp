#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <string_view>

namespace strix::qsa_selection_diagnostic {
inline constexpr std::uint32_t block_topk = 512;
inline constexpr std::uint32_t compression = 4;
inline constexpr std::uint32_t nonfinite_score = 1;
inline constexpr std::string_view identity = "qsa-top512-hierarchical-first-index-chronological-v1";
struct Shape { std::uint32_t blocks = 0, tail = 0; };
template<class T> struct View { T* data = nullptr; std::uint64_t count = 0; };
struct alignas(8) Item { float score; std::uint32_t index; };
static_assert(sizeof(Item) == 8);
// Fixed first scope: B<=32768, tail<=3, visible=4B+tail<=131072.
// Each block id denotes its original input position; padding ids remain unique.
[[nodiscard]] std::uint32_t selected_count(Shape);
[[nodiscard]] std::uint64_t scratch_items(Shape); // EACH ping/pong; max16384 items.
// Source-only diagnostic, not decoder integration or a selected default.
// Caller owns an explicit non-default stream and all views until completion,
// including after enqueue exceptions. No allocation, synchronization, status
// clearing or host publication occurs here. Producer must precede on same stream
// or via an already-established event. Status is a sticky caller-owned uint32.
// Exact view extents and disjoint ranges are checked BEFORE any submission.
// For B<=512 the result is all visible tokens: scores may be empty and are never
// read, ping/pong must be empty. Otherwise all scores must be finite; a validation
// kernel precedes every scratch/output write and ORs nonfinite_score on failure.
// Any nonzero status suppresses ALL selection/scratch/output writes. Finite ties
// (+0==-0 included) choose lower original block id. Final consumption order is
// chronological, followed by original tail tokens. Inputs remain read-only.
void enqueue(Shape, View<const float> scores, View<std::uint32_t> selected,
             View<Item> ping, View<Item> pong, View<std::uint32_t> status,
             hipStream_t);
} // namespace strix::qsa_selection_diagnostic

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

// The attention state a deepseek41 sequence leaves after its positions [0, position): everything a decode step reads
// from earlier positions. Per layer the window ring (AttentionHost::ring_capacity entries, slot = position % capacity),
// on the KV sources the compressed KV and index-key entries [0, position / ratio) and the ratio-2 carry, on the index
// sources the last row's top-k; plus the token history (the Engram n-grams of the next positions) and the greedy token
// after the last position. A prefill exports it (DevicePrefill::export_state), a decode forward imports it
// (DeviceForward::import_state) and continues bitwise as if it had run the positions itself; DeviceForward exports it too.
// Saved to a file, one long prefill serves several decode runs and the one-node control.
//
// The per-token size is what the context costs: 720 B of compressed KV (sources 2/8/14 at ratio 2, 20 at ratio 1) and
// 170 B of index keys per position, plus 40 rings of 136 x 528 B (2.9 MB) whatever the length.
namespace strix::deepseek41 {

struct AttentionLayerState {
    std::vector<std::uint8_t> ring;              // ring_capacity x 528 B, or empty (layer outside the range)
    std::vector<std::uint8_t> kv, idx_keys;      // KV sources: position / ratio entries of 288 / 68 B
    std::vector<float> carry;                    // ratio-2 sources: the pending group's (kv, score), 2 x 512 floats
    bool carry_valid = false;
    bool owns_kv = false, owns_index = false;
    std::uint32_t ratio = 0;
    std::vector<std::uint32_t> topk;             // index sources: the last row's selection, ascending
    bool topk_known = true;                      // false: the last row was computed elsewhere (row-split prefill)
};

struct AttentionState {
    std::uint32_t position = 0;
    std::uint32_t next_token = 0;                // greedy token after position - 1
    std::vector<std::uint32_t> tokens;           // tokens of positions [0, position)
    std::vector<AttentionLayerState> layers;

    // FNV of each layer's pieces, labelled and ordered as DeviceForward::state_hashes (ring@l, kv@l, idx_keys@l, carry@l,
    // topk@l); a topk the state does not know hashes as 0.
    [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> hashes() const;
    // FNV of hashes(), the tokens, position and next token; without_topk leaves out the last row's selections (decode
    // never reads them: the next step selects again), so the ranks of a row-split prefill and one node agree.
    [[nodiscard]] std::uint64_t digest(bool without_topk = false) const;
    [[nodiscard]] std::uint64_t bytes() const;   // host bytes of the attention pieces (without the tokens)
};

// Binary file: magic, version, header, then every layer's pieces and the tokens, the digest and an FNV checksum of every
// byte before it; load checks both.
void save_attention_state(const std::filesystem::path& path, const AttentionState& state);
[[nodiscard]] AttentionState load_attention_state(const std::filesystem::path& path);
// The same bytes in memory (a state message of strix/ds41-pipeline-v1).
[[nodiscard]] std::vector<std::uint8_t> serialize_attention_state(const AttentionState& state);
[[nodiscard]] AttentionState deserialize_attention_state(std::span<const std::uint8_t> bytes);

// The same bytes written in place (a producer that already holds the pieces elsewhere, e.g. a device capture into a
// pinned buffer, ds41::cuda::attn::ExportCapture). Fill each layer's flags and piece sizes, plan (offsets of every
// piece and the total), have the pieces written at their offsets, write the frame (magic, header, flags, ratios, sizes,
// tokens: every byte except the pieces, the digest and the checksum), then seal (digest and checksum from the bytes in
// place). The result is byte for byte serialize_attention_state of the same state; a ratio-2 carry that is not valid
// must hold zeros (as export_state leaves it).
struct AttentionFrameLayer {
    bool owns_kv = false, owns_index = false, carry_valid = false, topk_known = true;  // AttentionLayerState's defaults
    std::uint32_t ratio = 0;
    std::uint64_t ring_bytes = 0, kv_bytes = 0, idx_bytes = 0, carry_floats = 0, topk_count = 0;
    std::uint64_t ring_at = 0, kv_at = 0, idx_at = 0, carry_at = 0, topk_at = 0;  // plan_attention_frame
};
struct AttentionFrame {
    std::uint32_t position = 0, next_token = 0;
    std::vector<AttentionFrameLayer> layers;
    std::uint64_t token_count = 0;
    std::uint64_t pieces_end = 0;                       // plan: end of the last layer's pieces (the token count field)
    std::uint64_t tokens_at = 0, digest_at = 0, checksum_at = 0, total = 0;  // plan
};
void plan_attention_frame(AttentionFrame& frame);
// out.size() >= frame.total; tokens.size() == frame.token_count.
void write_attention_frame(const AttentionFrame& frame, std::span<const std::uint32_t> tokens, std::span<std::uint8_t> out);
struct AttentionFrameSeal {
    std::uint64_t digest = 0, checksum = 0;
};
// Digest (the per-piece FNVs on `threads` - 1 threads) and the checksum of every byte before it (one sequential FNV on
// its own thread, overlapped with the digest; it is the long pole, so 0 = the default 2 threads); writes both.
AttentionFrameSeal seal_attention_frame(const AttentionFrame& frame, std::span<std::uint8_t> out, unsigned threads = 0);

// Pipeline stages hold layer ranges of one sequence's state. A layer is present when its ring is not empty; an absent
// layer (empty ring, no caches) is skipped by hashes() and digest(). slice keeps layers [first, end) present (the vector
// keeps its size); merge combines two states of the same positions and tokens whose present layers do not overlap (a
// FRONT's [0, K) and a TAIL's [K, n)) into one: the result's size is the larger, its next token the second's (the
// TAIL's head chose it). Throws std::invalid_argument on positions, tokens or overlapping layers.
[[nodiscard]] AttentionState slice_attention_state(const AttentionState& state, std::uint32_t first, std::uint32_t end);
[[nodiscard]] AttentionState merge_attention_states(const AttentionState& front, const AttentionState& tail);
[[nodiscard]] bool attention_layer_present(const AttentionLayerState& layer) noexcept;

// Device bytes the decode attention state of one sequence of max_positions positions takes (rings, caches, carries,
// top-k), for the released plan: the budget of a KV pool slot.
[[nodiscard]] std::uint64_t attention_state_device_bytes(std::uint32_t max_positions, std::uint32_t layers = 40);

}  // namespace strix::deepseek41

#pragma once

#include <cstdint>
#include <filesystem>
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

// Device bytes the decode attention state of one sequence of max_positions positions takes (rings, caches, carries,
// top-k), for the released plan: the budget of a KV pool slot.
[[nodiscard]] std::uint64_t attention_state_device_bytes(std::uint32_t max_positions, std::uint32_t layers = 40);

}  // namespace strix::deepseek41

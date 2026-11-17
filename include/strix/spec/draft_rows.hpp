#pragma once
// Draft-only output-head rows (shared host layer). A drafted token only steers acceptance: the target verifies every
// position and emits its own greedy token, so the draft head may score a SUBSET of the vocabulary (HIP D-026 / D-043).
// The subset is kept in groups of `group` consecutive token ids, the unit a backend can copy without re-quantising
// (16 = one NVFP4 tile of the CUDA head, include/strix/cuda_core/nvfp4.hpp; 1 = a row-major Q8_0 row). Rows:
//   core     the leading core_groups groups (byte-level BPE ids follow merge rank, i.e. merge frequency);
//   always   given ids (control / chat tokens);
//   packs    id lists (e.g. kana/CJK script tokens) that join only when the prompt holds >= pack_min_hits of their ids
//            outside the core (the HIP pack-by-prompt rule);
//   fallback groups [core, fallback_groups) when no pack joined (a longer prefix of the same ranking for Latin-script
//            prompts; a script prompt spends those rows on its pack instead);
//   context  every prompt id and every emitted id (admitted after it is emitted), so an out-of-rows token costs only its
//            first appearance.
// Slot s of the compact head holds source group groups()[s]; the core occupies slots [0, core) in order, so its compact
// rows are the token ids themselves. A subset head's argmax equals the full head's whenever the full argmax is a row.
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::spec {

struct DraftRowsConfig {
    std::uint32_t vocab = 0;
    std::uint32_t group = 16;
    std::uint32_t core_groups = 2048;
    std::uint32_t fallback_groups = 4096;  // 0 or <= core_groups: no fallback
    std::uint32_t capacity_groups = 0;  // compact head capacity in groups (0 = every group)
    std::uint32_t pack_min_hits = 4;
    std::vector<std::uint32_t> always;
    std::vector<std::vector<std::uint32_t>> packs;
};

class DraftRows {
public:
    explicit DraftRows(DraftRowsConfig config);

    // New request: core + always + packs chosen by `prompt` + the prompt's own ids. Slots of the previous request beyond
    // the core become stale (materialised() drops to the core when it was materialised).
    void begin(std::span<const std::uint32_t> prompt);
    // Context admission of emitted ids (after they are emitted). Returns the number of new groups; ids past the capacity
    // are counted in dropped().
    std::uint32_t admit(std::span<const std::uint32_t> ids);

    [[nodiscard]] const std::vector<std::uint32_t>& groups() const { return slots_; }  // slot -> source group
    [[nodiscard]] std::uint32_t slots() const { return std::uint32_t(slots_.size()); }
    [[nodiscard]] std::uint32_t rows() const { return slots() * config_.group; }
    [[nodiscard]] std::uint32_t group() const { return config_.group; }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }
    // Leading slots whose rows the backend already holds; the backend copies [materialised(), slots()) and marks them.
    [[nodiscard]] std::uint32_t materialised() const { return materialised_; }
    void mark_materialised() { materialised_ = slots(); }
    void invalidate() { materialised_ = 0; }  // the backend lost its copy (new buffer, other subset)
    [[nodiscard]] std::uint32_t vocab() const { return config_.vocab; }
    [[nodiscard]] bool contains(std::uint32_t id) const { return id < config_.vocab && slot_of_[id / config_.group] != kNone; }
    // Compact row -> token id (throws past rows()).
    [[nodiscard]] std::uint32_t token(std::uint32_t row) const;
    [[nodiscard]] std::uint32_t packs_chosen() const { return packs_chosen_; }  // bit i = pack i joined
    [[nodiscard]] std::uint64_t dropped() const { return dropped_; }
    [[nodiscard]] std::string describe() const;  // one line: rows, share of the head, packs, dropped

    // Little-endian uint32 id file (tools/build_draft_vocab.py); ids >= vocab are an error.
    [[nodiscard]] static std::vector<std::uint32_t> read_ids(const std::string& path, std::uint32_t vocab);

private:
    static constexpr std::uint32_t kNone = 0xFFFFFFFFu;
    bool add_group(std::uint32_t g);
    DraftRowsConfig config_;
    std::uint32_t groups_total_ = 0, capacity_ = 0, core_ = 0, materialised_ = 0, packs_chosen_ = 0;
    std::uint64_t dropped_ = 0;
    std::vector<std::uint32_t> slots_, slot_of_;
};

} // namespace strix::spec

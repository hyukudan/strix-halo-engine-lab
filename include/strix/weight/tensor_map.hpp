#pragma once
// Tensor naming map: a family's pack directory described as DATA (name template, layer class, component, symbolic shape,
// admitted wire types), checked by one shared routine: every tensor of the pack is claimed by exactly one entry, every entry
// finds its tensors, and every shape equals the extents resolved from the family's hyper-parameters. The loader, the pack
// verifier and the planner (bytes per component and layer) all read the same table; a family adds a table, never code paths.
#include "strix/gguf_reader.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight::tensor_map {

// Layer classes an entry expands over ("blk.{L}." names); global entries have no layer.
enum class Layers : std::uint8_t { global, all, main, dense_ffn, moe, sparse_index, engram, nextn, count };

// Symbolic extents. A family resolves each symbol (optionally per layer) from its metadata; unused symbols stay 0.
enum class Sym : std::uint8_t {
    none, one, two, hidden, vocab, heads, q_lora, kv_lora, kv_a, qk_nope, qk_head, v_head, dense_ff,
    experts, expert_ff, latent, shared_ff, index_heads, index_key, index_pool, hc, hc_mix, hc_plus1,
    lookup_dim, lookup_heads, lookup_rows,
    kv_b_head,  // MLA kv_b rows per head in the unsplit (safetensors) form: qk_nope + v_head
    count
};

// One extent = product of two symbols (b defaults to one). GGUF order: the contiguous axis first.
struct Extent { Sym a = Sym::none; Sym b = Sym::one; };

// Admitted wire types as a bit set over GGML ids < 64.
constexpr std::uint64_t type_bit(std::uint32_t id) noexcept { return id < 64 ? (std::uint64_t{1} << id) : 0; }
// Matrices: any storage the shared host / CUDA / HIP decoders read (F32 F16 Q4_0 Q5_0 Q5_1 Q8_0 Q2_K..Q6_K IQ* BF16 MXFP4).
inline constexpr std::uint64_t kLinearTypes = type_bit(0) | type_bit(1) | type_bit(2) | type_bit(6) | type_bit(7) |
    type_bit(8) | type_bit(10) | type_bit(11) | type_bit(12) | type_bit(13) | type_bit(14) | type_bit(16) | type_bit(17) |
    type_bit(18) | type_bit(20) | type_bit(21) | type_bit(22) | type_bit(23) | type_bit(30) | type_bit(39);
inline constexpr std::uint64_t kF32 = type_bit(0);
// Lookup tables (row gathers): plain floats, Q8_0 and upstream F8_E4M3 (gguf_reader.hpp kGgmlF8E4M3).
inline constexpr std::uint64_t kLookupTypes = type_bit(0) | type_bit(1) | type_bit(30) | type_bit(8) | type_bit(43);

struct Entry {
    std::string_view name;       // "token_embd.weight" or "blk.{L}.attn_q_a.weight"
    Layers layers;
    std::string_view component;  // planner / byte-summary key, e.g. "experts.gate"
    std::array<Extent, 4> dims;  // trailing Sym::none = rank end
    std::uint64_t types;
};

// Resolved hyper-parameters of one pack: symbol values (per layer when the family has a per-layer array) and the
// layer-class membership of each block.
struct Model {
    std::uint32_t blocks = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Sym::count)> value{};
    std::array<std::vector<std::uint64_t>, static_cast<std::size_t>(Sym::count)> per_layer{};  // empty = use value
    std::vector<std::uint32_t> layer_class;  // bit (1 << Layers) per block
    [[nodiscard]] std::uint64_t resolve(Sym symbol, std::uint32_t layer) const noexcept;
    [[nodiscard]] bool in(Layers set, std::uint32_t layer) const noexcept;
};

struct Claim {
    std::size_t tensor = 0;  // index into the checked tensor list
    std::size_t entry = 0;
    std::int64_t layer = -1;
};

struct Result {
    std::vector<Claim> claims;        // one per tensor when ok
    std::vector<std::string> errors;  // unclaimed, missing, double claim, shape or type mismatch (first 64)
    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Checks a pack directory (all split parts concatenated) against a table. A tensor of wire id 43 is checked as
// F8_E4M3 only when its file is not a Strix tile pack; the caller passes such tensors with their wire id unchanged.
[[nodiscard]] Result check(std::span<const gguf::TensorInfo> tensors, std::span<const Entry> table, const Model& model);

// Expanded name of an entry for a layer ("{L}" replaced).
[[nodiscard]] std::string expand(std::string_view name, std::int64_t layer);

// A family's table plus the resolver of its symbols from the first split's metadata (false + error text when the
// metadata is incomplete or inconsistent). Tables live as data in src/weight/adapters/<family>.cpp.
struct FamilyMap {
    std::string_view architecture;  // general.architecture
    std::span<const Entry> table;
    bool (*model)(const gguf::FileInfo&, Model&, std::string& error);
    std::uint64_t retained_array_elements;  // gguf::Limits::max_retained_array_elements the resolver needs
};
// nullptr when no table describes the architecture.
[[nodiscard]] const FamilyMap* find_family(std::string_view architecture);

}  // namespace strix::weight::tensor_map

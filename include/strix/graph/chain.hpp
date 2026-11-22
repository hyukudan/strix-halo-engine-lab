#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strix::graph {

// Stable IDs. Parameters in the form vocabulary are supplied through Link::arg.
enum class Form : std::uint16_t {
    none = 0,
    store_f32 = 1, store_bf16 = 2, store_f16 = 3, store_a8g32 = 4, store_e4m3 = 5,
    residual_add = 6, residual_add_round_bf16 = 7,
    scale_const = 8, scale_layer = 9, bias_add = 10,
    act_silu_mul = 11, act_swiglu_clamp = 12, act_swiglu_softcap = 13,
    act_gelu_tanh_mul = 14, act_sigmoid = 15, gate_sigmoid_mul = 16,
    softcap = 17, clip_l2 = 18, norm_rms = 19, rope = 20,
    quant_a8g32 = 21, quant_a4had = 22, quant_e4m3 = 23, blocksum = 24,
    kv_append = 25, conv_tap = 26, mix_expand = 27, route_weight_mul = 28,
    finite_status = 29, emit_argmax = 30, emit_top2 = 31, emit_topn = 32,
    norm_layer = 33, norm_group = 34, mod_affine = 35, rope_inverse = 36,
    act_swiglu_oai = 37, rotate_hadamard = 38, load_round_bf16 = 39,
    store_iu4prep = 40, quant_i8rot = 41, emit_lse = 42, bias_by_class = 43,
    xch_publish = 44, xch_join = 45, wait_host = 46,
    recur_commit = 47, index_pool_append = 48
};

[[nodiscard]] std::string_view form_name(Form form) noexcept;
[[nodiscard]] std::optional<Form> parse_form(std::string_view name) noexcept;

struct Link {
    Form form;
    std::uint8_t arg;
    friend bool operator==(const Link&, const Link&) = default;
};
struct Chain {
    std::array<Link, 8> links;
    std::uint8_t count;
};
using ChainId = std::uint16_t;  // 0 is the empty chain

struct ChainArgs {
    float scalar[4];
    const void* aux[4];
    std::uint32_t state[2];
};
static_assert(sizeof(ChainArgs) <= 96);

class ChainTable {
public:
    ChainId intern(const Chain&);
    [[nodiscard]] const Chain& at(ChainId) const;
    [[nodiscard]] std::uint64_t identity(ChainId) const;

private:
    std::vector<Chain> chains_;
    std::unordered_multimap<std::uint64_t, ChainId> ids_;
};

enum class Pack : std::uint8_t { none, wave, sum, pipe };

}  // namespace strix::graph

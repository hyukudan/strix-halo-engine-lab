#include "cuda/core/chain/key.hpp"

#include <cstdio>
#include <utility>

namespace ch = strix::cuda::core::chain;
using namespace strix::graph;
using ch::Form;

constexpr auto ordered = ch::List<ch::Step<Form::residual_add, 1>, ch::Step<Form::norm_rms, 2>,
                                  ch::Step<Form::store_bf16, 3>>::description;
constexpr auto key = ch::kernel_key(ordered);
static_assert(key == ch::static_key<ch::List<ch::Step<Form::residual_add, 1>, ch::Step<Form::norm_rms, 2>,
                                            ch::Step<Form::store_bf16, 3>>>);
static_assert(key && key->scope == ch::Scope::row && key->count == 3);
static_assert(key->links[0] == 0x601 && key->links[1] == 0x1302 && key->links[2] == 0x203);
static_assert(ch::kernel_key(ch::List<>::description) == ch::KernelKey{});

int main() {
    ChainTable table;
    const auto id = table.intern(ordered);
    if (ch::kernel_key(table.at(id)) != key || ch::kernel_key(table.at(0)) != ch::KernelKey{}) return 1;
    auto changed = ordered;
    changed.links[3] = {Form::emit_lse, 255};  // ignored padding
    if (ch::kernel_key(changed) != key) return 2;
    changed.links[0].arg = 2;
    if (!ch::kernel_key(changed) || ch::kernel_key(changed) == key) return 3;
    changed.links[0].arg = 4;
    if (ch::kernel_key(changed)) return 4;
    changed = ordered;
    std::swap(changed.links[0], changed.links[1]);
    if (!ch::kernel_key(changed) || ch::kernel_key(changed) == key) return 5;
    changed.count = 9;
    if (ch::kernel_key(changed)) return 6;

    unsigned supported = 0;
    for (std::uint16_t n = 0; n <= static_cast<std::uint16_t>(Form::index_pool_append) + 1U; ++n) {
        const auto form = static_cast<Form>(n);
        const bool expected = n >= 1U && n <= 48U && (n < 44U || n > 46U);
        const auto bound = ch::kernel_key(Chain{{Link{form, 0}}, 1});
        if (bound.has_value() != expected) return 7;
        if (!bound) continue;
        if (form_name(form).empty() || bound->links[0] != (static_cast<unsigned>(n) << 8U)) return 8;
        const auto info = ch::form_info(form);
        if (bound->scope != info.scope) return 9;
        const std::uint8_t cap = info.arg == ch::Arg::state ? 2 : info.arg == ch::Arg::none ? 1 : 4;
        if (!ch::kernel_key(Chain{{Link{form, static_cast<std::uint8_t>(cap - 1U)}}, 1}) ||
            ch::kernel_key(Chain{{Link{form, cap}}, 1})) return 10;
        ++supported;
    }
    std::printf("K6c key mapping: %u CUDA forms, 3 protocol forms delegated\n", supported);
    return supported == 45 ? 0 : 11;
}

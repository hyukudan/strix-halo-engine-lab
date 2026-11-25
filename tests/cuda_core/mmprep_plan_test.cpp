#include "cuda/core/chain/key.hpp"
#include "ops/chain/prep_plan.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <type_traits>

namespace ch = strix::cuda::core::chain;
namespace prep = strix::ops::chain;

namespace {

constexpr prep::PrepShape shape{7, 64, 4, 4, 1, 32, 20};
constexpr std::array sequences{prep::PrepSequence{0, 3, 2, 9}, prep::PrepSequence{3, 4, 11, 20}};
constexpr std::size_t index_stride = 1973;  // The live index rows end before K/V capacity.
constexpr auto plan = prep::plan_prep(shape, sequences, 2, index_stride);
static_assert(plan && plan->heads == 73);
static_assert(plan->head(63).plane == prep::PrepPlane::query && plan->head(63).index == 63);
static_assert(plan->head(64).plane == prep::PrepPlane::key_value && plan->head(67).index == 3);
static_assert(plan->head(68).plane == prep::PrepPlane::index_query && plan->head(71).index == 3);
static_assert(plan->head(72).plane == prep::PrepPlane::index_key && plan->head(72).index == 0);
static_assert(plan->position(sequences[0], 2) == 4 && plan->position(sequences[1], 3) == 11);
static_assert(plan->rotary_offset(plan->position(sequences[1], 6)) == 896);
static_assert(plan->row_head(6, shape.query_heads, 63) == 57216);
static_assert(plan->cache_head<false>(sequences[0], 4, 3) == 11648);
static_assert(plan->cache_head<false>(sequences[1], 14, 3) == 28032);
static_assert(plan->cache_head<true>(sequences[1], 14, 0) == 5738);

using Form = ch::Form;
using CtaChain = ch::List<ch::Step<Form::norm_rms, 0>, ch::Step<Form::rope, 1>, ch::PrepRawStore>;
constexpr auto raw_key = ch::static_key<ch::PrepRawChain>;
constexpr auto cache_key = ch::static_key<ch::PrepCacheChain>;
constexpr auto value_key = ch::static_key<ch::PrepValueChain>;
constexpr auto cta_key = ch::static_key<CtaChain>;
static_assert(raw_key && raw_key->scope == ch::Scope::row && raw_key->count == 3);
static_assert(ch::PrepRawChain::description.links == CtaChain::description.links);
static_assert(cta_key && raw_key->links[0] != cta_key->links[0] && raw_key->links[1] != cta_key->links[1]);
static_assert(!std::is_base_of_v<ch::Step<Form::norm_rms>, ch::PrepNorm>);
static_assert(cache_key && value_key && raw_key != cache_key && value_key->links[0] == cache_key->links[2]);
static_assert(ch::PrepRawStore::link.form == Form::store_f32 && ch::PrepCacheStore::link.form == Form::kv_append);

constexpr bool invalid_plans_rejected() {
    auto altered = shape;
    altered.rotary_half = 31;
    if (prep::plan_prep(altered, sequences, 2, index_stride)) return false;
    altered = shape;
    altered.rotary_half = 0;
    if (prep::plan_prep(altered, sequences, 2, index_stride)) return false;
    altered = shape;
    altered.rotary_positions = 14;
    if (prep::plan_prep(altered, sequences, 2, index_stride)) return false;
    altered = shape;
    altered.query_heads = 65535;
    if (prep::plan_prep(altered, sequences, 2, index_stride)) return false;
    altered = shape;
    altered.rows = 0x80000000U;
    if (prep::plan_prep(altered, sequences, 2, index_stride)) return false;
    if (prep::plan_prep(shape, sequences, 2, 14U * prep::kPrepWidth)) return false;
    if (prep::plan_prep(shape, sequences, 2, std::numeric_limits<std::size_t>::max())) return false;
    if (prep::plan_prep(shape, std::span{sequences}.first(1), 2, index_stride)) return false;

    auto ranges = sequences;
    ranges[1].first_row = 2;
    if (prep::plan_prep(shape, ranges, 2, index_stride)) return false;
    ranges[1].first_row = 4;
    if (prep::plan_prep(shape, ranges, 2, index_stride)) return false;
    ranges = sequences;
    ranges[1].capacity = 14;
    if (prep::plan_prep(shape, ranges, 2, index_stride)) return false;
    ranges[1].capacity = std::numeric_limits<unsigned>::max();
    return !prep::plan_prep(shape, ranges, std::numeric_limits<unsigned>::max(), index_stride);
}
static_assert(invalid_plans_rejected());

constexpr auto empty = prep::plan_prep({}, {}, 0, 0);
static_assert(empty && empty->heads == 0 && !prep::plan_prep({}, sequences, 0, 0));
constexpr auto basic = prep::plan_prep({7, 8, 2, 0, 0, 64, 20}, sequences, 0, 0);
static_assert(basic && basic->heads == 10 && basic->rotary_offset(2) == 256);

}  // namespace

int main() {
    std::puts("MMPREP plan PASS: plane layout, sequence offsets, numeric keys and limits");
}

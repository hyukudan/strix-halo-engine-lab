#pragma once

#include "core/dense_row_plan.hpp"
#include "strix/attn/group_indexer.hpp"
#include "strix/weight/formats/scalar_float.hpp"

namespace strix::core {

template <RoundStore Policy>
STRIX_GI_HD float round_stored(float value) {
    if constexpr (Policy == RoundStore::separate) return weight::scalar_float::bf16_round(value);
    else return attn::group_indexer::bf16_round(value);
}

}  // namespace strix::core

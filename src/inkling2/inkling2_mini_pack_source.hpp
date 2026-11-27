#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "strix/weight/pack_source.hpp"

namespace strix::inkling2 {

std::string mini_expert_pack_name(unsigned layer, unsigned expert, std::string_view role);

// Bind a named study matrix to the exact reference's admitted source bytes.
weight::Status mini_pack_source(const weight::PackCheckpoint&, const MiniTextWeights&,
                               std::string_view name, weight::PackMatrixSource&);

}  // namespace strix::inkling2

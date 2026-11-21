#pragma once
// MiniMax public names and defaults over the shared TP2 transport core.
#include "minimax_traits.hpp"
#include "../../xport/channel_hip.hpp"
namespace strix::minimax::xfer {
using Memory = strix::xport::Memory;
using Publish = strix::xport::Publish;
using RxView = strix::xport::RxView;
using ProducerTarget = strix::xport::ProducerTarget;
using Stats = strix::xport::Stats;
using Options = strix::xport::ChannelOptions<MinimaxTraits>;
using Exchanger = strix::xport::Channel<MinimaxTraits>;
inline constexpr std::uint32_t kNonFinite = 0xFFFFFFFFU;
void argmax_exchange(Exchanger&, const float*, unsigned, unsigned, unsigned, std::uint32_t, std::uint64_t, std::uint32_t*);
} // namespace strix::minimax::xfer
namespace strix::xport { extern template class Channel<minimax::xfer::MinimaxTraits>; }

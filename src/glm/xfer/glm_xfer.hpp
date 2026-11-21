#pragma once
// GLM public names and defaults over the shared TP2 transport core.
#include "glm_traits.hpp"
#include "glm_seqpar.hpp"
#include "../../xport/channel_hip.hpp"
namespace strix::glm::xfer {
using Memory = strix::xport::Memory;
using Publish = strix::xport::Publish;
using RxView = strix::xport::RxView;
using ProducerTarget = strix::xport::ProducerTarget;
using Stats = strix::xport::Stats;
using Options = strix::xport::ChannelOptions<GlmTraits>;
using Exchanger = strix::xport::Channel<GlmTraits>;
using SeqparChannel = strix::xport::Channel<GlmSeqparTraits>;
inline constexpr std::uint32_t kNonFinite = 0xFFFFFFFFU;
void argmax_exchange(Exchanger&, const float*, unsigned, unsigned, unsigned, std::uint32_t, std::uint64_t, std::uint32_t*);
} // namespace strix::glm::xfer
namespace strix::xport { extern template class Channel<glm::xfer::GlmTraits>; }
namespace strix::xport { extern template class Channel<glm::xfer::GlmSeqparTraits>; }

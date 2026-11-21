#pragma once
#include "strix/dist/communicator_hip.hpp"
#include "deepseek41_prefill_xport_traits.hpp"
#include "xport/channel_hip.hpp"
#include "xport/channel_communicator.hpp"
#include "xport/hip_channel_runtime.hpp"
#include "strix/hip_check.hpp"

namespace strix::deepseek41::prefill_xport {
using Exchange = xport::Channel<Traits>;
// HIP operations of the channel communicator: the shared runtime (one body for every Channel<Traits>).
using Runtime = xport::HipChannelRuntime<Traits>;
using Communicator = xport::ChannelCommunicator<Exchange, dist::Communicator, Runtime>;
} // namespace strix::deepseek41::prefill_xport

namespace strix::xport {
extern template class Channel<deepseek41::prefill_xport::Traits>;
}

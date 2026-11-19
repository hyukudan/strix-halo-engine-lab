// The one instantiation of the shared Channel for the tensor-parallel slot gathers (tp_gather_traits.hpp).
#include "tp_gather_hip.hpp"
#include "channel_impl.hip"

template struct strix::xport::ChannelOptions<strix::xport::tp_gather::Traits>;
template class strix::xport::Channel<strix::xport::tp_gather::Traits>;

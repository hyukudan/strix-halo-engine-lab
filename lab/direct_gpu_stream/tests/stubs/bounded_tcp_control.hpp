#pragma once
#include "strix/direct_gpu_stream_commissioning.hpp"
namespace strix::transport {
struct CommissioningTcpPolicy { bool listener=false;std::uint16_t port=0;std::uint64_t maximum_frame_bytes=0;std::uint32_t maximum_polls=0;std::chrono::milliseconds timeout{10000}; };
class BoundedCommissioningTcpControl : public DistributedControlByteStream {public:explicit BoundedCommissioningTcpControl(CommissioningTcpPolicy);};
}

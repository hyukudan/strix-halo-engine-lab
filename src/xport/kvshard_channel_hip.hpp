#pragma once
#include "channel_hip.hpp"
#include "kvshard_channel.hpp"
#include <cstdlib>
#include <string>

namespace strix::xport::kvshard {
struct HipAttentionRuntime {
    hipStream_t posting_stream;
    hipStream_t stream() const { return posting_stream; }
    Publish publish() const { return Publish::kernel; }
    void synchronize(hipStream_t s) const {
        if (hipStreamSynchronize(s) != hipSuccess)
            throw std::runtime_error("kvshard: stream synchronization failed");
    }
};
// The Channel must be dedicated to this protocol. Its transport remains owned
// by the caller; peer rank / physical epoch / plan are Channel-level contracts.
template<class Traits>
auto glm_attention_channel(Channel<Traits>& channel, HipAttentionRuntime& runtime) {
    using Adapter = AttentionChannel<Channel<Traits>, HipAttentionRuntime>;
    const char* value = std::getenv("STRIX_GLM_KV_SHARD");
    const bool enabled = value != nullptr && std::string(value) == "1";
    if (!enabled) throw std::logic_error("kvshard: STRIX_GLM_KV_SHARD=1 required");
    const auto& options = channel.options();
    if (options.capacity < Adapter::max_frame_bytes || options.tx_memory != Memory::host_coherent ||
        options.rx_memory != Memory::host_coherent || options.stream != runtime.stream() ||
        options.rank > 1 || !options.crc || options.crc_deferred)
        throw std::invalid_argument("kvshard: capacity / coherent memory / stream / rank / immediate CRC contract");
    return Adapter(channel, runtime, enabled);
}
} // namespace strix::xport::kvshard

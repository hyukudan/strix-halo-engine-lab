#pragma once

#include "strix/session_state_memory.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix {

// Only host history is transactional. Device KV/GDN remains in-place and a
// post-dispatch failure poisons the session even when host history is preserved.
class SerialHostStateShadow {
public:
    SerialHostStateShadow(const SerialSessionStateLayout& layout,
                          const SessionStateMemoryBinding& published,
                          std::uint64_t generation)
        : published_(published), shadow_(published), bytes_(layout.host_total_bytes) {
        validate_session_state_memory_binding(layout, published);
        if (!generation || !bytes_ || !layout.host_alignment ||
            (layout.host_alignment & (layout.host_alignment-1U)) != 0 ||
            layout.host_alignment > std::numeric_limits<std::size_t>::max() ||
            bytes_ > std::numeric_limits<std::size_t>::max()-(layout.host_alignment-1U))
            throw std::invalid_argument("host shadow requires bounded aligned state and its own generation");
        storage_.resize(static_cast<std::size_t>(bytes_+layout.host_alignment-1U));
        const auto base = reinterpret_cast<std::uintptr_t>(storage_.data());
        const auto mask = static_cast<std::uintptr_t>(layout.host_alignment-1U);
        if (base > std::numeric_limits<std::uintptr_t>::max()-mask)
            throw std::overflow_error("host shadow alignment overflow");
        shadow_.host_base=(base+mask)&~mask;
        shadow_.host_bytes=bytes_;
        shadow_.host_allocation_generation=generation;
        validate_session_state_memory_binding(layout,shadow_);
    }
    SerialHostStateShadow(const SerialHostStateShadow&)=delete;
    SerialHostStateShadow& operator=(const SerialHostStateShadow&)=delete;
    void begin() {
        if(active_)throw std::logic_error("host shadow transaction already active");
        std::memcpy(reinterpret_cast<void*>(shadow_.host_base),reinterpret_cast<const void*>(published_.host_base),static_cast<std::size_t>(bytes_));
        active_=true;
    }
    void publish() {
        if(!active_)throw std::logic_error("host shadow transaction is not active");
        std::memcpy(reinterpret_cast<void*>(published_.host_base),reinterpret_cast<const void*>(shadow_.host_base),static_cast<std::size_t>(bytes_));
        active_=false;
    }
    void cancel() noexcept { active_=false; }
    [[nodiscard]] const SessionStateMemoryBinding& binding() const noexcept { return shadow_; }
private:
    SessionStateMemoryBinding published_,shadow_;
    std::uint64_t bytes_=0;
    std::vector<std::byte> storage_;
    bool active_=false;
};

} // namespace strix

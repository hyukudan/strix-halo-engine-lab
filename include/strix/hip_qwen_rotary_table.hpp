#pragma once

#include "strix/qwen_yarn_table.hpp"
#include <memory>
#include <atomic>
#include <stdexcept>

namespace strix {

// Default-stream tokens are not stream objects. In particular, some HIP
// runtimes dereference hipStreamLegacy in hipStreamGetDevice. Query only live,
// explicitly owned streams; all default forms belong to the current device.
template<class Stream, class Query>
int qwen_rotary_reader_device(Stream stream, Stream legacy, Stream per_thread,
    int active, Query query) {
    if (stream == Stream{} || stream == legacy || stream == per_thread) return active;
    return query(stream);
}

// Shared host policy, also exercised with failing device/drain operations in CPU tests.
inline void require_qwen_rotary_reader_device(int allocation, int active, int stream) {
    if (allocation < 0 || allocation != active || allocation != stream)
        throw std::invalid_argument("rotary reader requires verified local allocation and stream device");
}
template<class Select, class Drain, class Free>
bool retire_qwen_rotary_allocation(Select select, Drain drain, Free release) {
    if (!select() || !drain()) return false; // quarantine; never free without proven completion
    release();
    return true;
}

// Immutable shared allocation. Model, launcher, runtime and capture copies retain
// this owner. The allocation deleter drains pending readers before freeing storage.
class HipQwenRotaryTable {
public:
    explicit HipQwenRotaryTable(const RotaryDescriptor&);
    // CPU-only adoption. Unverified device metadata (-1) always refuses HIP readers.
    HipQwenRotaryTable(RotaryIdentity identity, std::shared_ptr<const void> allocation,
        std::uint64_t bytes) : descriptor_(identity.descriptor()), allocation_(std::move(allocation)),
        identity_(std::move(identity)) {
        if (!allocation_ || descriptor_.mode() != RotaryMode::yarn || bytes != descriptor_.table_bytes())
            throw std::invalid_argument("rotary allocation extent/owner mismatch");
    }
    ~HipQwenRotaryTable() = default;
    HipQwenRotaryTable(const HipQwenRotaryTable&) = delete;
    HipQwenRotaryTable& operator=(const HipQwenRotaryTable&) = delete;
    const void* data() const noexcept { return allocation_.get(); }
    std::uint32_t extent() const noexcept { return descriptor_.effective_max(); }
    const RotaryDescriptor& descriptor() const noexcept { return descriptor_; }
    const ModelContentIdentity& sha256() const noexcept { return identity_.table_sha256(); }
    int device() const noexcept { return device_; }
    std::uint64_t generation() const noexcept { return generation_; }
    const RotaryIdentity& identity() const noexcept { return identity_; }
    void check_position(std::uint64_t position) const {
        if (position >= extent()) throw std::out_of_range("YaRN position exceeds device table extent");
    }
private:
    const RotaryDescriptor descriptor_;
    std::shared_ptr<const void> allocation_;
    RotaryIdentity identity_;
    int device_ = -1;
    inline static std::atomic<std::uint64_t> next_generation_{1};
    const std::uint64_t generation_ = next_generation_.fetch_add(1);
};

// CPU admission can hold a resolved identity before upload; device readers cannot.
inline void require_qwen_rotary_device_payload(const RotaryIdentity& identity, const HipQwenRotaryTable* table) {
    if (identity.descriptor().mode() == RotaryMode::yarn && !table)
        throw std::invalid_argument("resolved YaRN reader requires uploaded coefficient table");
    if (table && identity != table->identity())
        throw std::invalid_argument("rotary reader payload identity mismatch");
}

// Checks allocation type/extent, allocation device, active device and stream device.
// Must run before the first reader, including capture and partial submissions.
// stream must be null, a HIP default-stream token, or a live stream retained by
// the caller. HIP cannot safely inspect an arbitrary stale or garbage handle.
void validate_qwen_rotary_reader(const HipQwenRotaryTable&, void* stream);

// Retain one submission through stream completion or captured graph destruction.
// Native callers do not enter this path. stream is a hipStream_t cast to void*.
void retain_qwen_rotary_reader(std::shared_ptr<const HipQwenRotaryTable>, void* stream);

} // namespace strix

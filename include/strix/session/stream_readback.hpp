#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace strix::session {

// One serial owner enqueues a reply's field copies on its producer stream and
// consumes a snapshot after one fence. The same storage serves each reply.
template<class Reply, class Backend>
class StreamReadback {
    static_assert(std::is_trivially_copyable_v<Reply> && std::is_nothrow_default_constructible_v<Reply>);

public:
    using Stream = typename Backend::Stream;

    StreamReadback() {
        if (void* storage = Backend::try_pin(sizeof(Reply))) {
            reply_ = std::construct_at(static_cast<Reply*>(storage));
            pinned_ = true;
        } else {
            reply_ = new Reply{};
        }
    }

    ~StreamReadback() {
        // An unretired copy can still write this address, even after an error.
        // The failed owner must retain the buffer instead of returning it to an allocator.
        if (!reply_) return;
        if (pinned_) Backend::unpin(reply_);
        else delete reply_;
    }

    StreamReadback(const StreamReadback&) = delete;
    StreamReadback& operator=(const StreamReadback&) = delete;

    template<class Enqueue>
    [[nodiscard]] Reply receive(Stream stream, Enqueue&& enqueue, const char* operation) {
        if (!reply_) throw std::logic_error("stream readback retains an unretired reply");
        const auto copy = [stream]<class T, std::size_t Extent>(std::span<T, Extent> field, const T* source,
                                                               const char* name) {
            Backend::copy_to_host(field.data(), source, field.size_bytes(), stream, name);
        };
        try {
            enqueue(*reply_, copy);
            Backend::wait(stream, operation);
        } catch (...) {
            if (!Backend::drain(stream)) reply_ = nullptr;
            throw;
        }
        return *reply_;
    }

private:
    Reply* reply_ = nullptr;
    bool pinned_ = false;
};

}  // namespace strix::session

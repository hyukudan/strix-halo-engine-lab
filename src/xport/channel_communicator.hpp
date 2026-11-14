#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::xport {
// Adapt a borrowed Channel to the Communicator byte interface. Framing, wire
// progress, validation and device readiness stay in Channel. Cross-stream
// consumers retain their physical slot until its next producer waits on their
// last-use event. Runtime contains only device API operations; host tests use a
// recorder. No host synchronization occurs in start/finish/release or reuse.
template<class Channel, class Legacy, class Runtime>
class ChannelCommunicator {
    using Stream = typename Runtime::Stream;
    using Event = typename Runtime::Event;
    using Tag = typename Runtime::InputTag;
    Legacy* legacy_;
    Channel* channel_;
    Runtime& runtime_;
    std::vector<Event> consumers_;
    const std::byte* local_ = nullptr;
    std::size_t bytes_ = 0;
    Tag tag_{};
    std::uint64_t held_ = 0;
    bool submitted_ = false;
    bool drained_ = false;

    void before_send(Stream producer) {
        runtime_.order(producer, channel_->options().stream);
        auto& event = consumers_[static_cast<std::size_t>(channel_->sent() % consumers_.size())];
        if (event) runtime_.wait(event, channel_->options().stream);
    }
    const void* receive(Stream consumer) {
        const auto view = channel_->recv_view();
        runtime_.acquire(view, consumer);
        held_ = view.sequence;
        return view.payload;
    }
public:
    ChannelCommunicator(Legacy* legacy, Channel* channel, Runtime& runtime)
        : legacy_(legacy), channel_(channel), runtime_(runtime),
          consumers_(channel ? channel->options().depth : 0U) {}
    ChannelCommunicator(const ChannelCommunicator&) = delete;
    ChannelCommunicator& operator=(const ChannelCommunicator&) = delete;
    ~ChannelCommunicator() {
        // Also retire off-stream consumers on exceptional exits before Channel
        // frees its slots. Failed progress releases RX waits in Channel itself.
        if (channel_) {
            if (!drained_) try { channel_->drain(); } catch (...) {}
            runtime_.synchronize(channel_->options().stream);
            for (auto event : consumers_) if (event) runtime_.synchronize_event(event);
        }
        for (auto event : consumers_) runtime_.destroy(event);
    }
    bool active() const { return channel_ != nullptr; }
    void drain() { if (channel_) { channel_->drain(); drained_ = true; } }
    void* send_staging(std::size_t bytes) {
        if (!channel_) return legacy_->send_staging(bytes);
        if (!bytes || bytes > channel_->options().capacity) return nullptr;
        before_send(channel_->options().stream);
        return channel_->tx();
    }
    void start_staged(Stream stream, std::size_t bytes, const Tag& tag) {
        if (!channel_) return legacy_->start_staged(stream, bytes, tag);
        runtime_.order(stream, channel_->options().stream);
        channel_->send(bytes, Runtime::tag(tag));
    }
    const void* finish_staged(Stream stream) {
        if (!channel_) return legacy_->finish_staged(stream);
        return receive(stream);
    }
    void release_staged(Stream stream) {
        if (!channel_) return legacy_->release_staged(stream);
        auto& event = consumers_[static_cast<std::size_t>((held_ - 1U) % consumers_.size())];
        runtime_.record(event, stream);
    }
    // Ready-view vocabulary is required by the existing DS41 facade's type
    // contract. Those calls remain on the legacy decode route.
    auto finish_staged_ready() { return legacy_->finish_staged_ready(); }
    void start_staged_ready(Stream stream, std::size_t bytes, const Tag& tag) {
        legacy_->start_staged_ready(stream, bytes, tag);
    }
    void release_staged_ready(Stream stream) { legacy_->release_staged_ready(stream); }
    void start(Stream stream, const void* local, std::size_t bytes, const Tag& tag) {
        if (!channel_) return legacy_->start(stream, local, bytes, tag);
        local_ = static_cast<const std::byte*>(local);
        bytes_ = bytes;
        tag_ = tag;
        submitted_ = bytes && bytes <= channel_->options().capacity;
        if (submitted_) {
            before_send(stream);
            channel_->send_copy(local, bytes, Runtime::tag(tag));
        }
    }
    void finish(Stream stream, void* remote) {
        if (!channel_) return legacy_->finish(stream, remote);
        // Large legacy messages use the same bounded slot allocation. Both
        // ranks split identical byte counts and preserve the caller's tag.
        for (std::size_t at = 0; at < bytes_;) {
            const auto n = std::min(bytes_ - at, channel_->options().capacity);
            if (!submitted_) {
                before_send(stream);
                channel_->send_copy(local_ + at, n, Runtime::tag(tag_));
            }
            const auto* peer = receive(stream);
            if (remote) runtime_.copy(static_cast<std::byte*>(remote) + at, peer, n, stream);
            release_staged(stream);
            submitted_ = false;
            at += n;
        }
    }
    void exchange(Stream stream, const void* local, void* remote, std::size_t bytes, const Tag& tag) {
        start(stream, local, bytes, tag);
        finish(stream, remote);
    }
};
} // namespace strix::xport

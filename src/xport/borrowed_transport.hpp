#pragma once
#include "strix/dist/transport.hpp"

namespace strix::xport {

// Preserve each borrower's inherited defaults independently of forwarding.
enum class BorrowedAbort { forward, base_default };
enum class BorrowedState { forward, base_default };

// Non-owning: the transport must outlive this adapter and all of its calls.
// Framing, progress, CRCs and state remain owned by the underlying transport.
template<BorrowedAbort Abort = BorrowedAbort::forward,
         BorrowedState State = BorrowedState::forward>
class BorrowedTransport final : public dist::Transport {
public:
    explicit BorrowedTransport(dist::Transport& t) : t_(t) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override { t_.exchange(o, i); }
    // Inherit exchange-then-callback, even when the backend supports overlap.
    [[nodiscard]] std::string name() const override { return t_.name(); }
    [[nodiscard]] bool verifies_delivery() const noexcept override { return t_.verifies_delivery(); }
    void abort(const std::string& why) override {
        if constexpr (Abort == BorrowedAbort::forward) t_.abort(why);
        else dist::Transport::abort(why);
    }
    [[nodiscard]] std::string state() const override {
        if constexpr (State == BorrowedState::forward) return t_.state();
        else return dist::Transport::state();
    }

private:
    dist::Transport& t_;
};

} // namespace strix::xport

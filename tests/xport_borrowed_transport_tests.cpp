#include "xport/borrowed_transport.hpp"
#include "xport/wire.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value) {
    if (!value) throw std::runtime_error("BorrowedTransport fixture");
}
enum class Call { send, receive, exchange, overlap_begin, callback, overlap_end, name, abort, state };
struct Message {
    Call call;
    const std::byte* out;
    std::size_t out_size;
    std::byte* in;
    std::size_t in_size;
};
struct Fake final : strix::dist::Transport {
    mutable std::vector<Call> calls;
    std::vector<Message> messages;
    void (*callback)(void*) = nullptr;
    void* context = nullptr;
    const std::string* reason = nullptr;
    bool fail_exchange = false;
    mutable bool fail_state = false;
    unsigned& destroyed;

    explicit Fake(unsigned& count) : destroyed(count) {}
    ~Fake() override { ++destroyed; }
    void send(std::span<const std::byte> m) override {
        calls.push_back(Call::send);
        messages.push_back({Call::send, m.data(), m.size(), nullptr, 0});
    }
    void receive(std::span<std::byte> m) override {
        calls.push_back(Call::receive);
        messages.push_back({Call::receive, nullptr, 0, m.data(), m.size()});
        for (auto& b : m) b = std::byte{0x42};
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        calls.push_back(Call::exchange);
        messages.push_back({Call::exchange, out.data(), out.size(), in.data(), in.size()});
        if (fail_exchange) throw std::runtime_error("exchange failure");
        for (std::size_t k = 0; k < in.size(); ++k) in[k] = out[k];
    }
    void exchange_overlapped(std::span<const std::byte> out, std::span<std::byte> in,
                             void (*on_wire)(void*), void* ctx) override {
        calls.push_back(Call::overlap_begin);
        messages.push_back({Call::overlap_begin, out.data(), out.size(), in.data(), in.size()});
        callback = on_wire;
        context = ctx;
        on_wire(ctx); // Deliberately before completion: the adapter must bypass this override.
        for (std::size_t k = 0; k < in.size(); ++k) in[k] = out[k];
        calls.push_back(Call::overlap_end);
    }
    std::string name() const override {
        calls.push_back(Call::name);
        return "recording";
    }
    void abort(const std::string& why) override {
        calls.push_back(Call::abort);
        reason = &why;
    }
    std::string state() const override {
        calls.push_back(Call::state);
        if (fail_state) throw std::runtime_error("state failure");
        return "progress 7/11";
    }
};
struct Callback {
    Fake& fake;
    std::span<const std::byte> out;
    std::span<const std::byte> in;
    unsigned count = 0;
    static void run(void* ctx) {
        auto& self = *static_cast<Callback*>(ctx);
        require(self.fake.calls.back() == Call::exchange);
        for (std::size_t k = 0; k < self.in.size(); ++k) require(self.in[k] == self.out[k]);
        ++self.count;
        self.fake.calls.push_back(Call::callback);
    }
};

template<strix::xport::BorrowedAbort Abort, strix::xport::BorrowedState State>
void forwarding() {
    unsigned destroyed = 0;
    {
        Fake fake(destroyed);
        {
            strix::xport::BorrowedTransport<Abort, State> borrowed(fake);
            strix::dist::Transport& transport = borrowed;
            const strix::dist::Transport& inspect = borrowed;
            const std::array out{std::byte{1}, std::byte{0x80}, std::byte{0xff}};
            std::array<std::byte, 3> in{};
            transport.send(out);
            transport.receive(in);
            for (auto b : in) require(b == std::byte{0x42});
            transport.exchange(out, in);
            require(in == out);
            in.fill(std::byte{0});
            Callback callback{fake, out, in};
            transport.exchange_overlapped(out, in, Callback::run, &callback);
            require(in == out && callback.count == 1);
            require(fake.callback == nullptr && fake.context == nullptr);
            require(inspect.name() == "recording");
            const std::string why = "stop";
            transport.abort(why);
            if constexpr (State == strix::xport::BorrowedState::forward)
                require(inspect.state() == "progress 7/11");
            else
                require(inspect.state().empty());
            std::vector<Call> expected{Call::send, Call::receive, Call::exchange,
                Call::exchange, Call::callback, Call::name};
            if constexpr (Abort == strix::xport::BorrowedAbort::forward) {
                expected.push_back(Call::abort);
                require(fake.reason == &why);
            } else {
                require(fake.reason == nullptr);
            }
            if constexpr (State == strix::xport::BorrowedState::forward)
                expected.push_back(Call::state);
            require(fake.calls == expected);
            require(fake.messages.size() == 4);
            for (const auto& m : fake.messages) {
                if (m.call != Call::receive)
                    require(m.out == out.data() && m.out_size == out.size());
                if (m.call != Call::send)
                    require(m.in == in.data() && m.in_size == in.size());
            }
            fake.fail_exchange = true;
            bool threw = false;
            try { transport.exchange_overlapped(out, in, Callback::run, &callback); }
            catch (const std::runtime_error& e) { threw = std::string(e.what()) == "exchange failure"; }
            require(threw && callback.count == 1);
            fake.fail_state = true;
            const auto calls_before_state = fake.calls.size();
            threw = false;
            try {
                const auto state = inspect.state();
                if constexpr (State == strix::xport::BorrowedState::base_default) require(state.empty());
            }
            catch (const std::runtime_error& e) {
                threw = true;
                require(std::string(e.what()) == "state failure");
            }
            if constexpr (State == strix::xport::BorrowedState::forward) {
                require(threw && fake.calls.size() == calls_before_state + 1);
                require(fake.calls.back() == Call::state);
            } else {
                require(!threw && fake.calls.size() == calls_before_state);
            }
        }
        require(destroyed == 0); // The adapter never destroys the borrowed transport.
    }
    require(destroyed == 1);
}

// A backend without an overlap override keeps Transport's serial fallback.
struct Serial final : strix::dist::Transport {
    unsigned phase = 0;
    void send(std::span<const std::byte>) override {}
    void receive(std::span<std::byte>) override {}
    void exchange(std::span<const std::byte>, std::span<std::byte>) override { phase = 1; }
    std::string name() const override { return "serial"; }
};
void fallback() {
    Serial serial;
    strix::xport::BorrowedTransport<> borrowed(serial);
    borrowed.exchange_overlapped({}, {}, [](void* ctx) {
        auto& phase = static_cast<Serial*>(ctx)->phase;
        require(phase == 1);
        phase = 2;
    }, &serial);
    require(serial.phase == 2);
    require(borrowed.state().empty());
    borrowed.abort("default no-op");
    require(!borrowed.verifies_delivery());  // the base default: nothing proven
}

// The borrower reports exactly what the backend proves about delivered bytes, and the channel's deferred payload CRC
// follows it: deferred only over a verifying transport, checked before publication otherwise.
struct Verified final : strix::dist::Transport {
    bool verified = true;
    void send(std::span<const std::byte>) override {}
    void receive(std::span<std::byte>) override {}
    void exchange(std::span<const std::byte>, std::span<std::byte>) override {}
    std::string name() const override { return "verified"; }
    bool verifies_delivery() const noexcept override { return verified; }
};
void delivery() {
    Verified backend;
    strix::xport::BorrowedTransport<> borrowed(backend);
    require(borrowed.verifies_delivery());
    backend.verified = false;
    require(!borrowed.verifies_delivery());
    using strix::xport::payload_crc_after_publish;
    static_assert(payload_crc_after_publish(true, true, true));
    static_assert(!payload_crc_after_publish(true, true, false));   // raw stream / device path: check before publish
    static_assert(!payload_crc_after_publish(true, false, true));
    static_assert(!payload_crc_after_publish(false, true, true));
    for (const bool v : {true, false}) {
        backend.verified = v;
        require(payload_crc_after_publish(true, true, borrowed.verifies_delivery()) == v);
    }
}
} // namespace

int main() {
    using strix::xport::BorrowedAbort;
    using strix::xport::BorrowedState;
    forwarding<BorrowedAbort::base_default, BorrowedState::base_default>(); // Probe.
    forwarding<BorrowedAbort::forward, BorrowedState::base_default>(); // MiniMax rank.
    forwarding<BorrowedAbort::forward, BorrowedState::base_default>(); // MiniMax serve.
    forwarding<BorrowedAbort::forward, BorrowedState::forward>(); // GLM.
    fallback();
    delivery();
}

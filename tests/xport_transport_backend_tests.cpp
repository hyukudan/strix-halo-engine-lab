#include <cstdint>
// Channel's existing publication enum must coexist with the shared Backend API.
namespace strix::xport { enum class Publish : std::uint32_t; }
#include "xport/transport_backend.hpp"
#include "xport/failure_log.hpp"
#include <memory>
#include <string>
#include <array>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value) { if (!value) throw std::runtime_error("TransportBackend fixture"); }
struct Runtime {
    struct Tag { unsigned kind, layer, step; };
    struct Options { int stream = 7; } options_;
    struct View { const void* data; unsigned flag; };
    struct Message { const void* source; std::size_t bytes; Tag tag; };
    std::vector<Message> messages;
    std::vector<char> calls;
    unsigned received = 0;
    bool fail_send = false, fail_drain = false;
    const Options& options() const { return options_; }
    std::uint64_t sent() const { return messages.size(); }
    void send_copy(const void* source, std::size_t bytes, Tag tag) {
        if (fail_send) throw std::runtime_error("send");
        calls.push_back('p');
        messages.push_back({source, bytes, tag});
    }
    const void* recv() { calls.push_back('w'); return messages.at(received++).source; }
    View recv_view() { calls.push_back('v'); return {messages.at(received++).source, 42}; }
    void drain() {
        calls.push_back('d');
        if (fail_drain) throw std::runtime_error("deferred CRC");
    }
};
using B = strix::xport::TransportBackend<Runtime>;
static_assert(strix::xport::Backend<B>);
template<class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::runtime_error&) { caught = true; }
    catch (const std::invalid_argument&) { caught = true; }
    require(caught);
}
}

static void minimax_publication_modes() {
    struct PublishedRuntime {
        using Tag = int;
        enum class Mode { kernel, producer };
        struct Options { int stream = 7; } opts;
        std::uint64_t sequence = 0;
        int tag = 0, waits = 0, views = 0;
        std::size_t bytes = 0;
        Mode mode = Mode::kernel;
        const Options& options() const { return opts; }
        std::uint64_t sent() const { return sequence; }
        void send(std::size_t n, Tag t, Mode m) {
            bytes = n; tag = t; mode = m; ++sequence;
        }
        const void* recv() { ++waits; return this; }
        int recv_view() { ++views; return 42; }
    } runtime;
    strix::xport::TransportBackend<PublishedRuntime> backend(runtime);
    for (auto mode : {PublishedRuntime::Mode::kernel, PublishedRuntime::Mode::producer}) {
        const auto ticket = backend.post_publish(23, 24576, mode);
        if (runtime.mode != mode || runtime.bytes != 24576 || runtime.tag != 23 ||
            ticket.sequence != runtime.sent())
            throw std::runtime_error("MiniMax publication forwarding");
        if (mode == PublishedRuntime::Mode::kernel) {
            if (backend.wait(ticket, 7).remote != &runtime)
                throw std::runtime_error("MiniMax stream wait forwarding");
        } else if (backend.wait_view(ticket, 7) != 42) {
            throw std::runtime_error("MiniMax consumer wait forwarding");
        }
        backend.release(ticket, 7);
    }
    if (runtime.waits != 1 || runtime.views != 1 || runtime.sent() != 2)
        throw std::runtime_error("MiniMax extra publication/wait");
}

static void first_failure_log() {
    // No GPU or peer is needed to prove the original cause is emitted once,
    // before the owning thread calls check(), even if another error follows.
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    require(file != nullptr);
    std::string failure;
    strix::xport::report_first_failure(failure, "producer of sequence 42 not published within the timeout", 1, file.get());
    strix::xport::report_first_failure(failure, "peer closed", 1, file.get());
    require(failure == "producer of sequence 42 not published within the timeout");
    std::rewind(file.get());
    char line[256]{};
    require(std::fgets(line, sizeof(line), file.get()) != nullptr);
    require(std::string(line) == "xport: rank=1 exchange failed: producer of sequence 42 not published within the timeout\n");
    require(std::fgets(line, sizeof(line), file.get()) == nullptr);
}

int main() {
    first_failure_log();
    minimax_publication_modes();
    Runtime r;
    // Simulate earlier control traffic on the same Channel before this adapter.
    int prior = 0;
    r.send_copy(&prior, sizeof(prior), {0, 0, 0});
    (void)r.recv();
    B b(r);
    std::array<int, 8> payload{};
    std::array<B::Ticket, 8> tickets{};
    for (unsigned i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<int>(i);
        tickets[i] = b.post_copy({i % 2, i / 2, 19}, &payload[i], sizeof(int));
        const auto& m = r.messages.back();
        require(m.source == &payload[i] && m.bytes == sizeof(int));
        require(m.tag.kind == i % 2 && m.tag.layer == i / 2 && m.tag.step == 19);
        require(tickets[i].sequence == i + 2);
    }
    rejects([&] { (void)b.wait(tickets[1], 7); });
    rejects([&] { (void)b.wait(tickets[0], 8); });
    for (unsigned i = 0; i < payload.size(); ++i) {
        if (i % 2 == 0) require(b.wait(tickets[i], 7).remote == &payload[i]);
        else {
            const auto view = b.wait_view(tickets[i], 7);
            require(view.data == &payload[i] && view.flag == 42);
        }
        const auto count = r.calls.size();
        b.release(tickets[i], 7);
        require(r.calls.size() == count); // release must introduce no operation
    }
    rejects([&] { (void)b.wait(tickets[0], 7); });
    require(r.received == 9);
    require(r.calls == std::vector<char>({'p', 'w', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p',
                                          'w', 'v', 'w', 'v', 'w', 'v', 'w', 'v'}));
    // Recreate the short-lived adapter across slot reuse / intervening traffic.
    B reused(r);
    const auto ticket = reused.post_copy({1, 5, 20}, &payload[0], sizeof(int));
    require(reused.wait(ticket, 7).remote == &payload[0]);
    reused.release(ticket, 7);
    reused.drain();
    require(r.calls.back() == 'd');
    r.fail_drain = true;
    rejects([&] { reused.drain(); }); // deferred-tail failure remains visible
    r.fail_send = true;
    rejects([&] { (void)reused.post_copy({0, 6, 20}, &prior, sizeof(prior)); });
    require(r.sent() == 10);
    rejects([&] { (void)reused.post({0, 6, 20}, {}, sizeof(int), {false, true}); });
}

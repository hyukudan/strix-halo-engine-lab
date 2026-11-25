// Loopback test of the replica front end: two fake replicas and the router, all on HttpServer, driven by the
// blocking HTTP client. Checks SSE pass-through, cancel propagation, and that a request is never sent twice.

#include "strix/serve/http_client.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/replica_proxy.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <net/if.h>
#endif
#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace {
enum class ProxyPausePoint { before_placement, before_connect_failure };
struct ProxyPause {
    ProxyPausePoint point;
    std::barrier<> reached{2}, resume{2};
};
std::atomic<ProxyPause*> active_pause{nullptr};

void pause_forward(ProxyPausePoint point) {
    auto* pause = active_pause.load();
    if (pause && point == pause->point && active_pause.compare_exchange_strong(pause, nullptr)) {
        pause->reached.arrive_and_wait();
        pause->resume.arrive_and_wait();
    }
}
} // namespace

// Build the shared proxy here so the barriers don't need hooks in the serving binary.
#define STRIX_REPLICA_PROXY_TEST_PAUSE(point) pause_forward(ProxyPausePoint::point)
#include "../src/serve/replica_proxy.cpp"
#undef STRIX_REPLICA_PROXY_TEST_PAUSE

namespace {
using namespace strix::serve;
using namespace std::chrono_literals;

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void rejects_alias(std::string_view first, std::string_view second) {
    const auto a = parse_replica_endpoint(first), b = parse_replica_endpoint(second);
    require(a && b, "alias fixture parses both endpoints");
    try { require_distinct_replicas({*a, *b}); }
    catch (const std::invalid_argument& e) {
        require(std::string_view(e.what()).find("distinct worker") != std::string_view::npos, e.what());
        return;
    }
    throw std::runtime_error("duplicate socket accepted: " + std::string(first) + " / " + std::string(second));
}

void scoped_alias() {
#ifdef __linux__
    const auto index = if_nametoindex("lo");
    require(index != 0, "loopback interface has an index");
    rejects_alias("[fe80::1%lo]:18196", "[fe80::1%" + std::to_string(index) + "]:18196");
#else
    rejects_alias("[fe80::1%1]:18196", "[fe80::1%01]:18196");
#endif
}

void scoped_uri() {
    const auto uri = parse_replica_endpoint("http://[fe80::1%25eth%30]:18196/");
    const auto bare = parse_replica_endpoint("[fe80::1%251]:18196");
    const auto uri_numeric = parse_replica_endpoint("http://[fe80::1%251]:18196");
    require(uri && uri->host == "fe80::1%eth0", "URI zone is decoded in the client dial host");
    require(bare && bare->host == "fe80::1%251" && uri_numeric && uri_numeric->host == "fe80::1%1",
        "bare numeric scope and escaped URI scope have different dial hosts");
    require(!parse_replica_endpoint("http://[fe80::1%25]:18196") &&
            !parse_replica_endpoint("http://[fe80::1%25eth%00]:18196"), "malformed URI zones are refused");
}

bool wait_until(const std::atomic<bool>& flag, std::chrono::milliseconds limit = 10s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!flag.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
    return flag.load();
}

void distinct_numeric_scopes() {
    const auto bare = parse_replica_endpoint("[fe80::1%251]:18196");
    const auto uri = parse_replica_endpoint("http://[fe80::1%251]:18196");
    require(bare && uri, "numeric scope endpoints parse");
    require_distinct_replicas({*bare, *uri});
}

const std::string kEventA = "data: {\"t\":\"a\"}\n\n";
const std::string kEventB = "data: {\"t\":\"b\"}\n\n";
const std::string kDone = "data: [DONE]\n\n";

// A running HttpServer on a free loopback port.
struct Served {
    explicit Served(HttpHandler handler, std::uint16_t listen_port = 0) {
        HttpServerConfig config;
        config.port = listen_port;
        server = std::make_unique<HttpServer>(config, std::move(handler));
        server->listen();
        port = server->port();
        thread = std::thread([this] { server->serve(); });
    }
    ~Served() { stop(); }
    void stop() {
        if (!server) return;
        server->stop();
        if (thread.joinable()) thread.join();
        server.reset();
    }
    std::unique_ptr<HttpServer> server;
    std::thread thread;
    std::uint16_t port = 0;
};

struct FakeReplica {
    std::string name;
    std::atomic<int> generations{0};
    std::atomic<int> busy{0};
    std::atomic<bool> release_second{false};  // set by the test once the first event reached the client
    std::atomic<bool> saw_client_gone{false};
    std::atomic<bool> draining{false};
    std::atomic<const char*> worker_id{nullptr};
    std::string last_auth;
    Served served;

    explicit FakeReplica(std::string id) : name(std::move(id)), served([this](const HttpRequest& q, HttpResponseWriter& r) { handle(q, r); }) {}

    void handle(const HttpRequest& q, HttpResponseWriter& r) {
        if (q.method == "GET" && q.path == "/api/status") {
            const auto* id = worker_id.load();
            r.send(200, "application/json", std::string("{\"state\":\"") + (draining ? "draining" : "ready")
                + (id ? std::string("\",\"worker_instance_id\":\"") + id : "")
                + "\",\"capacity\":{\"slots_total\":4,\"slots_busy\":" + std::to_string(busy.load())
                + ",\"kv_bytes_free\":null,\"queue_depth\":0}}");
            return;
        }
        if (q.method == "GET" && q.path == "/v1/models") { r.send(200, "application/json", "{\"data\":[\"" + name + "\"]}"); return; }
        ++generations;
        last_auth = q.header("authorization");
        if (q.path == "/v1/chat/completions") {
            if (q.body.find("\"stream\":true") == std::string::npos) {
                r.send(200, "application/json", "{\"replica\":\"" + name + "\",\"echo\":" + std::to_string(q.body.size()) + "}",
                    {{"X-Request-Id", "req-" + name}});
                return;
            }
            if (!r.begin_stream(200, "text/event-stream")) return;
            (void)r.write(kEventA);
            // The second event is held back until the client has the first: a proxy that buffered would deadlock here.
            (void)wait_until(release_second);
            (void)r.write(kEventB);
            (void)r.write(kDone);
            r.end_stream();
            return;
        }
        if (q.path == "/endless") {
            if (!r.begin_stream(200, "text/event-stream")) return;
            const auto deadline = std::chrono::steady_clock::now() + 10s;
            while (std::chrono::steady_clock::now() < deadline) {
                if (!r.write(kEventA)) { saw_client_gone = true; return; }
                std::this_thread::sleep_for(10ms);
            }
            r.end_stream();
            return;
        }
        if (q.path == "/cut") {
            // A stream that dies after its first event: no terminator, no end of the chunked body.
            if (!r.begin_stream(200, "text/event-stream")) return;
            (void)r.write(kEventA);
            throw std::runtime_error("replica died mid-stream");
        }
        if (q.path == "/fail") { r.send(500, "application/json", "{\"error\":\"" + name + "\"}"); return; }
        r.send(404, "application/json", "{}");
    }
};

HttpClientRequest to(std::uint16_t port, std::string method, std::string target, std::string_view body = {}) {
    HttpClientRequest q;
    q.port = port;
    q.method = std::move(method);
    q.target = std::move(target);
    q.body = body;
    q.idle_timeout_ms = 20000;
    return q;
}

template<class Action>
HttpClientResult forward_while_paused(const HttpClientRequest& request, ProxyPausePoint point, Action action) {
    ProxyPause pause{point};
    active_pause.store(&pause);
    auto forwarded = std::async(std::launch::async, [&] {
        std::string body;
        return http_fetch(request, body);
    });
    pause.reached.arrive_and_wait();
    try { action(); }
    catch (...) { pause.resume.arrive_and_wait(); throw; }
    pause.resume.arrive_and_wait();
    return forwarded.get();
}

void quarantine_during_placement() {
    FakeReplica a("A"), b("B");
    a.worker_id = "first";
    b.worker_id = "second";
    ReplicaProxyConfig config;
    config.replicas = {{"127.0.0.1", a.served.port}, {"127.0.0.1", b.served.port}};
    config.block_bytes = 64;
    ReplicaProxy proxy(config);
    proxy.admit_distinct_workers();
    Served front([&](const HttpRequest& q, HttpResponseWriter& r) { proxy.handle(q, r); });

    const std::string prefix(128, 'p');
    auto q = to(front.port, "POST", "/v1/chat/completions", prefix);
    q.headers = {{"X-Session-Id", "quarantine-session"}};
    std::string body;
    const auto seed = http_fetch(q, body);
    require(seed.ok && seed.head.header("x-strix-replica") == "0", "seed both affinities on A");

    const auto blocked = forward_while_paused(q, ProxyPausePoint::before_placement, [&] {
        b.worker_id = a.worker_id.load();
        proxy.refresh();
        const auto states = proxy.states();
        require(!states[0].healthy && !states[1].healthy, "poll published duplicate quarantine");
    });
    const auto deliveries = a.generations.load() + b.generations.load();

    b.worker_id = "second";
    a.busy = 1;
    proxy.refresh();
    const auto prefix_route = http_fetch(to(front.port, "POST", q.target, prefix), body);
    q.body = "{}";
    const auto session_route = http_fetch(q, body);
    const auto prefix_replica = prefix_route.head.header("x-strix-replica");
    const auto session_replica = session_route.head.header("x-strix-replica");
    if (!blocked.ok || blocked.head.status != 503 || deliveries != 1 || !prefix_route.ok || !session_route.ok
        || prefix_replica != "1" || session_replica != "1")
        throw std::runtime_error("quarantine placement: status=" + std::to_string(blocked.head.status)
            + " deliveries=" + std::to_string(deliveries) + " recovered prefix replica=" + prefix_replica
            + " session replica=" + session_replica);
}

void late_connect_failure() {
    FakeReplica a("A"), b("B");
    a.worker_id = "first";
    b.worker_id = "second";
    ReplicaProxyConfig config;
    config.replicas = {{"127.0.0.1", a.served.port}, {"127.0.0.1", b.served.port}};
    ReplicaProxy proxy(config);
    proxy.admit_distinct_workers();
    std::unique_ptr<Served> recovered;
    Served front([&](const HttpRequest& q, HttpResponseWriter& r) { proxy.handle(q, r); });
    a.served.stop();

    const auto q = to(front.port, "POST", "/v1/chat/completions", "{}");
    const auto retried = forward_while_paused(q, ProxyPausePoint::before_connect_failure, [&] {
        recovered = std::make_unique<Served>([&](const HttpRequest& request, HttpResponseWriter& response) {
            a.handle(request, response);
        }, a.served.port);
        // A traverses quarantine and recovery with the same ID. Comparing IDs alone would miss this.
        b.worker_id = a.worker_id.load();
        proxy.refresh();
        require(!proxy.states()[0].healthy, "original worker entered quarantine");
        b.worker_id = "second";
        proxy.refresh();
        require(proxy.states()[0].healthy, "original worker recovered before failure publication");
    });
    require(retried.ok && retried.head.header("x-strix-replica") == "1", "unsent request retries on B");
    require(a.generations == 0 && b.generations == 1, "only the retry reached a worker");
    require(proxy.states()[0].healthy, "stale connect failure marked recovered worker unhealthy");
    const auto status = proxy.status();
    const auto& a_status = status.find("replicas")->as_array()[0];
    require(a_status.find("failed")->as_double() == 0, "stale connect failure counted against recovered worker");
}

enum class AffinityFailure { connect, response, replacement };

void unpinned_affinity_invalidation(AffinityFailure failure) {
    FakeReplica a("A"), b("B");
    if (failure == AffinityFailure::replacement) a.worker_id = "first";
    ReplicaProxyConfig config;
    config.replicas = {{"127.0.0.1", a.served.port}, {"127.0.0.1", b.served.port}};
    config.block_bytes = 64;
    ReplicaProxy proxy(config);
    proxy.refresh();
    std::unique_ptr<Served> recovered;
    Served front([&](const HttpRequest& q, HttpResponseWriter& r) { proxy.handle(q, r); });

    const std::string prefix(128, 'p');
    auto q = to(front.port, "POST", "/v1/chat/completions", prefix);
    q.headers = {{"X-Session-Id", "legacy-session"}};
    std::string body;
    const auto seed = http_fetch(q, body);
    require(seed.ok && seed.head.status == 200 && seed.head.header("x-strix-replica") == "0",
        "seed both unpinned affinities on A");

    if (failure == AffinityFailure::connect) {
        a.served.stop();
        const auto retried = http_fetch(to(front.port, "POST", q.target, "{}"), body);
        require(retried.ok && retried.head.status == 200 && retried.head.header("x-strix-replica") == "1"
            && !proxy.states()[0].healthy && a.generations == 1 && b.generations == 1,
            "current connect failure marks A unhealthy and retries an unrelated request on B");
        recovered = std::make_unique<Served>([&](const HttpRequest& request, HttpResponseWriter& response) {
            a.handle(request, response);
        }, a.served.port);
    } else if (failure == AffinityFailure::response) {
        const auto cut = http_fetch(to(front.port, "POST", "/cut", "{}"), body);
        require(!cut.ok && cut.head.status == 200 && cut.head.header("x-strix-replica") == "0"
            && body == kEventA && !proxy.states()[0].healthy && a.generations == 2 && b.generations == 0,
            "current response failure is truncated and marks A unhealthy without replaying the request");
        a.served.stop();
        recovered = std::make_unique<Served>([&](const HttpRequest& request, HttpResponseWriter& response) {
            a.handle(request, response);
        }, a.served.port);
    } else {
        a.worker_id = "replacement";
    }
    a.busy = 1;
    proxy.refresh();
    const auto states = proxy.states();
    require(states[0].healthy && states[1].healthy && states[0].capacity.slots_busy == 1
        && states[1].capacity.slots_busy == 0, "both replicas healthy and B has lower reported load");

    // A one-slot difference still allows a prefix hit. Probe each affinity independently.
    const auto prefix_route = http_fetch(to(front.port, "POST", q.target, prefix), body);
    q.body = "{}";
    const auto session_route = http_fetch(q, body);
    const auto prefix_replica = prefix_route.head.header("x-strix-replica");
    const auto session_replica = session_route.head.header("x-strix-replica");
    if (!prefix_route.ok || prefix_route.head.status != 200 || !session_route.ok || session_route.head.status != 200
        || prefix_replica != "1" || session_replica != "1")
        throw std::runtime_error(std::string("unpinned affinity survived worker failure or replacement")
            + ": prefix replica=" + prefix_replica + " session replica=" + session_replica);
}

void client_basics(FakeReplica& a) {
    std::string body;
    auto r = http_fetch(to(a.served.port, "GET", "/api/status"), body);
    require(r.ok && r.head.status == 200 && r.request_sent && r.stage == HttpClientStage::done, "client: plain GET");
    require(body.find("\"slots_total\":4") != std::string::npos && r.body_bytes == body.size(), "client: Content-Length body");
    require(r.head.header("content-type") == "application/json", "client: header lookup");

    // Nothing listens on a port that was just released: the failure is reported before any byte is sent.
    std::uint16_t dead = 0;
    { Served gone([](const HttpRequest&, HttpResponseWriter& w) { w.send(200, "text/plain", "x"); }); dead = gone.port; }
    auto q = to(dead, "POST", "/x", "payload");
    q.connect_timeout_ms = 1500;
    r = http_fetch(q, body);
    require(!r.ok && !r.request_sent && r.stage == HttpClientStage::connect && !r.error.empty(), "client: refused connect");

    // A sink abort closes the connection and is reported as cancelled.
    HttpClientSink sink;
    sink.on_data = [](std::string_view) { return false; };
    r = http_exchange(to(a.served.port, "GET", "/v1/models"), sink);
    require(!r.ok && r.cancelled && r.request_sent && r.stage == HttpClientStage::body, "client: sink abort");
}

void run() {
    FakeReplica a("A"), b("B");
    client_basics(a);

    ReplicaProxyConfig config;
    config.replicas = {{"127.0.0.1", a.served.port}, {"127.0.0.1", b.served.port}};
    config.block_bytes = 64;
    config.connect_timeout_ms = 1500;
    ReplicaProxy proxy(config);
    require(proxy.states()[0].healthy == false, "replicas are unhealthy until the first status read");
    proxy.refresh();
    require(proxy.states()[0].healthy && proxy.states()[1].healthy, "both replicas healthy");
    Served front([&](const HttpRequest& q, HttpResponseWriter& r) { proxy.handle(q, r); });
    std::string body;

    // Router-owned routes.
    auto r = http_fetch(to(front.port, "GET", "/health"), body);
    require(r.ok && r.head.status == 200 && body.find("\"replicas_healthy\":2") != std::string::npos, "router health");

    // Non-stream request: same status, type, body and headers as the replica sends; credentials pass through.
    const std::string plain = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    auto q = to(front.port, "POST", "/v1/chat/completions", plain);
    q.headers = {{"Content-Type", "application/json"}, {"Authorization", "Bearer k"}};
    r = http_fetch(q, body);
    require(r.ok && r.head.status == 200 && body == "{\"replica\":\"A\",\"echo\":" + std::to_string(plain.size()) + "}", "plain body passes through");
    require(r.head.header("content-type") == "application/json" && r.head.header("x-request-id") == "req-A"
        && r.head.header("x-strix-replica") == "0" && r.head.header("content-length") == std::to_string(body.size()),
        "plain headers pass through");
    require(a.last_auth == "Bearer k" && a.generations == 1 && b.generations == 0, "request reached exactly one replica");

    // SSE: events arrive one by one, bytes unchanged.
    const std::string stream = "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    std::string events;
    std::vector<std::size_t> pieces;
    HttpClientSink sink;
    std::string seen_type;
    sink.on_head = [&](const HttpClientHead& head) { seen_type = head.header("content-type"); return true; };
    sink.on_data = [&](std::string_view data) {
        events.append(data);
        pieces.push_back(data.size());
        if (events == kEventA) { a.release_second = true; b.release_second = true; }
        return true;
    };
    r = http_exchange(to(front.port, "POST", "/v1/chat/completions", stream), sink);
    require(r.ok && r.head.status == 200 && seen_type == "text/event-stream", "SSE head");
    require(events == kEventA + kEventB + kDone, "SSE bytes unchanged");
    require(pieces.size() >= 2 && pieces.front() == kEventA.size(), "first SSE event was delivered before the second was produced");
    require(a.generations + b.generations == 2, "one generation per request");

    // Cancel: the client leaves mid-stream, the replica sees its own client go away.
    int taken = 0;
    HttpClientSink leaving;
    leaving.on_data = [&](std::string_view) { return ++taken < 3; };
    r = http_exchange(to(front.port, "POST", "/endless", "{}"), leaving);
    require(r.cancelled && !r.ok, "client aborted the stream");
    require(wait_until(a.saw_client_gone) || wait_until(b.saw_client_gone, 1ms), "cancel reached the replica");

    // No duplicate after a failure once the request was sent: a cut stream and a 500 are both final.
    int before_a = a.generations, before_b = b.generations;
    events.clear();
    HttpClientSink collect;
    collect.on_data = [&](std::string_view data) { events.append(data); return true; };
    r = http_exchange(to(front.port, "POST", "/cut", "{}"), collect);
    require(!r.ok && r.head.status == 200 && events == kEventA, "cut stream delivers what the replica sent and nothing else");
    require((a.generations - before_a) + (b.generations - before_b) == 1, "cut stream is not replayed on the other replica");
    before_a = a.generations; before_b = b.generations;
    r = http_fetch(to(front.port, "POST", "/fail", "{}"), body);
    require(r.ok && r.head.status == 500 && (body == "{\"error\":\"A\"}" || body == "{\"error\":\"B\"}"), "replica error passes through");
    require((a.generations - before_a) + (b.generations - before_b) == 1, "a failed request is not retried");

    // Session stickiness through the front end: the same session returns to its replica whatever the load.
    b.busy = 0; a.busy = 2;
    proxy.refresh();
    q = to(front.port, "POST", "/v1/chat/completions", plain);
    q.headers = {{"X-Session-Id", "conv-7"}};
    r = http_fetch(q, body);
    require(r.ok && r.head.header("x-strix-replica") == "1", "new session placed on the less busy replica");
    a.busy = 0; b.busy = 3;
    proxy.refresh();
    r = http_fetch(q, body);
    require(r.ok && r.head.header("x-strix-replica") == "1", "session stays on its replica");
    r = http_fetch(to(front.port, "GET", "/v1/models"), body);
    require(r.ok && body == "{\"data\":[\"A\"]}", "unkeyed request goes to the less busy replica");

    // Draining replica takes nothing new.
    b.busy = 0; a.busy = 0; a.draining = true;
    proxy.refresh();
    require(!proxy.states()[0].healthy && proxy.states()[1].healthy, "draining replica is skipped");
    r = http_fetch(to(front.port, "GET", "/v1/models"), body);
    require(r.ok && body == "{\"data\":[\"B\"]}", "request avoids the draining replica");
    a.draining = false;
    proxy.refresh();

    // Replica gone between two status reads: nothing was sent to it, so the request moves to the other one.
    b.busy = 3;
    proxy.refresh();
    before_b = b.generations;
    a.served.stop();
    r = http_fetch(to(front.port, "POST", "/v1/chat/completions", plain), body);
    require(r.ok && r.head.status == 200 && r.head.header("x-strix-replica") == "1", "refused connection moves to the other replica");
    require(b.generations - before_b == 1, "moved request ran once");
    require(!proxy.states()[0].healthy, "refusing replica is marked unhealthy at once");
    r = http_fetch(to(front.port, "GET", "/router/status"), body);
    require(r.ok && body.find("\"retried_before_send\":1") != std::string::npos, "router status counts the move");

    // No replica left: 503, nothing invented.
    b.served.stop();
    r = http_fetch(to(front.port, "POST", "/v1/chat/completions", plain), body);
    require(r.ok && r.head.status == 503 && body.find("no healthy replica") != std::string::npos, "no replica: 503");
    proxy.refresh();
    r = http_fetch(to(front.port, "GET", "/health"), body);
    require(r.ok && r.head.status == 503, "router health reports no replica");

    require(parse_replica_endpoint("http://node-b:8080/") && parse_replica_endpoint("http://node-b:8080/")->port == 8080
        && parse_replica_endpoint("10.0.0.2:9000")->host == "10.0.0.2", "endpoint parsing");
    require(!parse_replica_endpoint("node-b") && !parse_replica_endpoint("node-b:0") && !parse_replica_endpoint("node-b:99999")
        && !parse_replica_endpoint(":80") && !parse_replica_endpoint("http://a/b:80"), "malformed endpoints rejected");
    require_distinct_replicas({{"127.0.0.1", 18196}, {"127.0.0.1", 18197}, {"127.0.0.2", 18196}});
    for (const auto& [first, second] : {
            std::pair{"127.0.0.1:18196", "http://127.0.0.1:18196/"},
            std::pair{"http://[::1]:18196", "http://[0:0:0:0:0:0:0:1]:18196"},
            std::pair{"[::1]:18196", "http://[0000:0::0001]:18196/"},
            std::pair{"[2001:DB8::A]:18196", "[2001:db8:0:0:0:0:0:a]:18196"},
            std::pair{"127.0.0.1:18196", "http://[::ffff:127.0.0.1]:18196"},
            std::pair{"http://[::ffff:7f00:1]:18196", "127.0.0.1:18196"}})
        rejects_alias(first, second);
#ifndef _WIN32
    rejects_alias("127.0.0.1:18196", "127.1:18196");
    rejects_alias("10.0.0.2:18196", "0x0a.0.0.02:18196");
#endif
}

} // namespace

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        const auto selected = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
        if (argc > 2 || (argc == 2 && selected != "scoped-alias" && selected != "scoped-uri" &&
            selected != "named-alias" && selected != "numeric-scopes" && selected != "quarantine-placement" &&
            selected != "late-connect-failure" && selected != "unpinned-recovery" && selected != "unpinned-replacement" &&
            selected != "unpinned-response"))
            throw std::invalid_argument("unknown replica proxy test selector");
        if (selected.empty() || selected == "scoped-alias") scoped_alias();
        if (selected.empty() || selected == "scoped-uri") scoped_uri();
        if (selected.empty() || selected == "named-alias") rejects_alias("localhost:18196", "127.0.0.1:18196");
        if (selected.empty() || selected == "numeric-scopes") distinct_numeric_scopes();
        if (selected.empty() || selected == "quarantine-placement") quarantine_during_placement();
        if (selected.empty() || selected == "late-connect-failure") late_connect_failure();
        if (selected.empty() || selected == "unpinned-recovery") unpinned_affinity_invalidation(AffinityFailure::connect);
        if (selected.empty() || selected == "unpinned-replacement") unpinned_affinity_invalidation(AffinityFailure::replacement);
        if (selected.empty() || selected == "unpinned-response") unpinned_affinity_invalidation(AffinityFailure::response);
        if (selected.empty()) run();
    } catch (const std::exception& e) {
        std::cerr << "replica_proxy_tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "replica_proxy_tests: ok\n";
    return 0;
}

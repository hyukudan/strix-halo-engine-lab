#include "strix/serve/replica_proxy.hpp"

#include "strix/serve/http_client.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <set>
#include <stdexcept>

namespace strix::serve {
namespace {

// Headers that describe one connection, not the message: never copied across the proxy.
bool hop_by_hop(std::string_view name) {
    static constexpr std::array<std::string_view, 10> names{"host", "connection", "content-length",
        "transfer-encoding", "keep-alive", "proxy-connection", "te", "upgrade", "expect", "trailer"};
    for (const auto n : names) if (n == name) return true;
    return false;
}

std::string error_body(std::string_view type, std::string_view message) {
    Json detail = Json::object();
    detail.set("type", Json(type));
    detail.set("message", Json(message));
    Json out = Json::object();
    out.set("error", std::move(detail));
    return out.dump();
}

// Decode the URI spelling once, before the socket host is used for either resolution or dialing.
std::optional<std::string> decode_uri_zone(std::string_view zone) {
    if (zone.empty()) return std::nullopt;
    constexpr std::string_view unreserved =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    std::string out;
    for (std::size_t i = 0; i < zone.size(); ++i) {
        if (zone[i] != '%') {
            if (unreserved.find(zone[i]) == std::string_view::npos) return std::nullopt;
            out += zone[i];
            continue;
        }
        if (zone.size() - i < 3) return std::nullopt;
        unsigned value = 0;
        const auto* first = zone.data() + i + 1;
        const auto parsed = std::from_chars(first, first + 2, value, 16);
        if (parsed.ec != std::errc{} || parsed.ptr != first + 2 || value <= 32 || value == 127)
            return std::nullopt;
        const auto ch = static_cast<char>(value);
        if (std::string_view("/:[]").find(ch) != std::string_view::npos) return std::nullopt;
        out += ch;
        i += 2;
    }
    return out;
}

std::string replica_label(const ReplicaEndpoint& replica) {
    return "--replica " + replica.host + ":" + std::to_string(replica.port);
}

} // namespace

std::optional<ReplicaEndpoint> parse_replica_endpoint(std::string_view text) {
    constexpr std::string_view scheme = "http://";
    const bool uri = text.starts_with(scheme);
    if (uri) text.remove_prefix(scheme.size());
    while (!text.empty() && text.back() == '/') text.remove_suffix(1);
    const auto colon = text.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == text.size()) return std::nullopt;
    const auto digits = text.substr(colon + 1);
    unsigned port = 0;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), port);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || port == 0 || port > 65535)
        return std::nullopt;
    auto host = text.substr(0, colon);
    if (host.find_first_of("/ \t\r\n") != std::string_view::npos || host.find('\0') != std::string_view::npos)
        return std::nullopt;
    const bool bracketed = host.size() > 2 && host.front() == '[' && host.back() == ']';
    if (bracketed) host = host.substr(1, host.size() - 2);
    if (host.find_first_of("[]") != std::string_view::npos) return std::nullopt;
    if (const auto zone = host.find('%'); uri && zone != std::string_view::npos) {
        if (!bracketed || host.find(':') >= zone || !host.substr(zone).starts_with("%25")) return std::nullopt;
        const auto decoded = decode_uri_zone(host.substr(zone + 3));
        if (!decoded) return std::nullopt;
        return ReplicaEndpoint{std::string(host.substr(0, zone + 1)) + *decoded, static_cast<std::uint16_t>(port)};
    }
    return ReplicaEndpoint{std::string(host), static_cast<std::uint16_t>(port)};
}

namespace {

std::vector<std::vector<std::string>> distinct_replica_hosts(const std::vector<ReplicaEndpoint>& replicas) {
    std::set<std::pair<std::string, std::uint16_t>> seen;
    std::vector<std::vector<std::string>> resolved;
    for (const auto& replica : replicas) {
        auto hosts = resolve_http_hosts(replica.host, replica.port);
        if (hosts.empty()) throw std::invalid_argument(replica_label(replica) + " cannot be resolved");
        for (const auto& host : hosts)
            if (!seen.emplace(host, replica.port).second)
                throw std::invalid_argument(replica_label(replica) +
                    " resolves to a repeated socket address; each replica must be a distinct worker");
        resolved.push_back(std::move(hosts));
    }
    return resolved;
}

} // namespace

void require_distinct_replicas(const std::vector<ReplicaEndpoint>& replicas) {
    (void)distinct_replica_hosts(replicas);
}

ReplicaProxy::ReplicaProxy(ReplicaProxyConfig config)
    : config_(std::move(config)),
      router_(config_.replicas.size(), config_.affinity_entries, config_.block_bytes),
      states_(config_.replicas.size()), generations_(config_.replicas.size()), counters_(config_.replicas.size()) {}

ReplicaState ReplicaProxy::probe(std::size_t replica, const std::string& connect_host) const {
    HttpClientRequest request;
    request.host = config_.replicas[replica].host;
    request.port = config_.replicas[replica].port;
    request.connect_host = connect_host;
    request.target = config_.status_target;
    request.connect_timeout_ms = config_.connect_timeout_ms;
    request.idle_timeout_ms = config_.status_timeout_ms;
    std::string body;
    const auto result = http_fetch(request, body);
    if (result.ok && result.head.status == 200)
        if (auto parsed = replica_state_from_status(body)) return std::move(*parsed);
    return {};
}

void ReplicaProxy::admit_distinct_workers() {
    const auto resolved = distinct_replica_hosts(config_.replicas);
    std::set<std::string> instances;
    std::vector<std::string> connect_hosts;
    std::vector<ReplicaState> states;
    for (std::size_t r = 0; r < resolved.size(); ++r) {
        for (const auto& host : resolved[r]) {
            auto state = probe(r, host);
            if (state.worker_instance_id.empty()) continue;
            if (!instances.insert(state.worker_instance_id).second)
                throw std::invalid_argument(replica_label(config_.replicas[r]) + " worker instance is already configured");
            connect_hosts.push_back(host);
            states.push_back(std::move(state));
            break;
        }
        if (states.size() != r + 1)
            throw std::invalid_argument(replica_label(config_.replicas[r]) + " did not report worker_instance_id");
    }
    connect_hosts_ = std::move(connect_hosts);
    states_ = std::move(states);
}

void ReplicaProxy::refresh() {
    const bool pinned = !connect_hosts_.empty();
    if (!pinned) {
        for (std::size_t r = 0; r < config_.replicas.size(); ++r) {
            auto next = probe(r, {});
            const std::lock_guard lock(mutex_);
            const bool replaced = states_[r].worker_instance_id != next.worker_instance_id;
            if ((states_[r].healthy && !next.healthy) || replaced) router_.forget(r);
            if (states_[r].healthy != next.healthy || replaced)
                ++generations_[r];
            next.ewma_ms_per_token = states_[r].ewma_ms_per_token;
            states_[r] = std::move(next);
        }
        return;
    }

    std::vector<ReplicaState> next;
    for (std::size_t r = 0; r < connect_hosts_.size(); ++r) next.push_back(probe(r, connect_hosts_[r]));

    // Publish one sweep together so a changed identity cannot temporarily add a duplicate load slot.
    const std::lock_guard lock(mutex_);
    for (std::size_t r = 0; r < next.size(); ++r) {
        auto& state = next[r];
        const auto& id = state.worker_instance_id;
        if (id.empty() || std::count_if(next.begin(), next.end(), [&](const ReplicaState& peer) {
                return peer.worker_instance_id == id;
            }) > 1)
            state.healthy = false;
        const bool replaced = states_[r].worker_instance_id != id;
        if ((states_[r].healthy && !state.healthy) || replaced) router_.forget(r);
        if (states_[r].healthy != state.healthy || replaced) ++generations_[r];
        state.ewma_ms_per_token = states_[r].ewma_ms_per_token;
    }
    states_ = std::move(next);
}

std::vector<ReplicaState> ReplicaProxy::states() const {
    const std::lock_guard lock(mutex_);
    return states_;
}

Json ReplicaProxy::status() const {
    const std::lock_guard lock(mutex_);
    Json replicas = Json::array();
    for (std::size_t r = 0; r < states_.size(); ++r) {
        const auto& s = states_[r];
        Json capacity = Json::object();
        capacity.set("slots_total", Json(s.capacity.slots_total));
        capacity.set("slots_busy", Json(s.capacity.slots_busy));
        capacity.set("kv_bytes_free", s.capacity.kv_bytes_free ? Json(*s.capacity.kv_bytes_free) : Json());
        capacity.set("queue_depth", Json(s.capacity.queue_depth));
        Json row = Json::object();
        row.set("index", Json(static_cast<std::uint64_t>(r)));
        row.set("endpoint", Json(config_.replicas[r].host + ":" + std::to_string(config_.replicas[r].port)));
        row.set("healthy", Json(s.healthy));
        row.set("capacity", std::move(capacity));
        row.set("inflight", Json(router_.inflight(r)));
        row.set("forwarded", Json(counters_[r].forwarded));
        row.set("failed", Json(counters_[r].failed));
        row.set("by_session", Json(counters_[r].by_session));
        row.set("by_prefix", Json(counters_[r].by_prefix));
        row.set("by_load", Json(counters_[r].by_load));
        replicas.as_array().push_back(std::move(row));
    }
    Json out = Json::object();
    out.set("replicas", std::move(replicas));
    out.set("retried_before_send", Json(retried_.load()));
    out.set("unplaced", Json(unplaced_.load()));
    return out;
}

void ReplicaProxy::handle(const HttpRequest& request, HttpResponseWriter& response) {
    if (request.method == "GET" && request.path == "/router/status") {
        response.send(200, "application/json", status().dump());
        return;
    }
    if (request.method == "GET" && (request.path == "/health" || request.path == "/v1/health")) {
        std::uint64_t healthy = 0;
        for (const auto& s : states()) healthy += s.healthy ? 1U : 0U;
        Json out = Json::object();
        out.set("status", Json(healthy ? "ok" : "unavailable"));
        out.set("replicas_healthy", Json(healthy));
        out.set("replicas_total", Json(static_cast<std::uint64_t>(config_.replicas.size())));
        response.send(healthy ? 200 : 503, "application/json", out.dump());
        return;
    }
    forward(request, response);
}

void ReplicaProxy::forward(const HttpRequest& request, HttpResponseWriter& response) {
    RouteKeyStorage storage;
    const auto route = route_request_from_http(request.path, request.header("x-session-id"), request.body, storage);
    std::uint64_t excluded = 0;
    for (;;) {
#ifdef STRIX_REPLICA_PROXY_TEST_PAUSE
        STRIX_REPLICA_PROXY_TEST_PAUSE(before_placement);
#endif
        std::size_t replica;
        std::uint64_t generation = 0;
        RouteReason reason = RouteReason::none;
        {
            // Selection also publishes affinity. Order both with refresh's health and affinity updates.
            const std::lock_guard lock(mutex_);
            replica = router_.place(route, states_, excluded);
            if (replica != kNoReplica) {
                generation = generations_[replica];
                reason = router_.last_reason();
            }
        }
        if (replica == kNoReplica) {
            ++unplaced_;
            response.send(503, "application/json", error_body("overloaded_error", "no healthy replica"),
                {{"Retry-After", "1"}});
            return;
        }
        HttpClientRequest out;
        out.host = config_.replicas[replica].host;
        out.port = config_.replicas[replica].port;
        if (!connect_hosts_.empty()) out.connect_host = connect_hosts_[replica];
        out.method = request.method;
        out.target = request.target.empty() ? request.path : request.target;
        out.body = request.body;
        out.connect_timeout_ms = config_.connect_timeout_ms;
        out.idle_timeout_ms = config_.idle_timeout_ms;
        for (const auto& h : request.headers) if (!hop_by_hop(h.first)) out.headers.push_back(h);

        // Streams (SSE, or any body without a length) pass through as they arrive; a body with a Content-Length is
        // delivered whole, so the client sees the same bytes and the same framing a replica would send.
        bool started = false, streaming = false, client_lost = false;
        int status = 0;
        std::string content_type, buffered;
        std::vector<std::pair<std::string, std::string>> headers;
        HttpClientSink sink;
        sink.cancelled = [&] { return response.client_gone(); };
        sink.on_head = [&](const HttpClientHead& head) {
            status = head.status;
            content_type = head.header("content-type");
            for (const auto& h : head.headers)
                if (!hop_by_hop(h.first) && h.first != "content-type" && h.first != "cache-control"
                    && h.first != "x-accel-buffering")
                    headers.push_back(h);
            headers.emplace_back("X-Strix-Replica", std::to_string(replica));
            streaming = head.header("content-length").empty() && head.status != 204 && head.status != 304
                && request.method != "HEAD";
            if (!streaming) {
                for (const auto& h : head.headers) if (h.first == "cache-control") headers.push_back(h);
                return true;
            }
            started = true;
            if (!response.begin_stream(status, content_type, headers)) { client_lost = true; return false; }
            return true;
        };
        sink.on_data = [&](std::string_view data) {
            if (!streaming) { buffered.append(data); return true; }
            if (!response.write(data)) { client_lost = true; return false; }
            return true;
        };
        const auto result = http_exchange(out, sink);
        router_.finished(replica, route);

        if (!result.ok && !result.cancelled) {
#ifdef STRIX_REPLICA_PROXY_TEST_PAUSE
            if (!result.request_sent) STRIX_REPLICA_PROXY_TEST_PAUSE(before_connect_failure);
#endif
            {
                const std::lock_guard lock(mutex_);
                if (generations_[replica] == generation) {
                    states_[replica].healthy = false;
                    ++generations_[replica];
                    router_.forget(replica);
                    if (!result.request_sent) ++counters_[replica].failed;
                }
            }
            // A different replica can take the request only if no bytes reached this one.
            if (!result.request_sent) {
                excluded |= 1ULL << replica;
                ++retried_;
                continue;
            }
        }
        {
            const std::lock_guard lock(mutex_);
            auto& c = counters_[replica];
            ++c.forwarded;
            if (reason == RouteReason::session) ++c.by_session;
            else if (reason == RouteReason::prefix) ++c.by_prefix;
            else ++c.by_load;
            if (!result.ok && !result.cancelled) ++c.failed;
        }
        if (streaming) {
            // A stream cut by the replica ends here; the missing terminator tells the client it is incomplete.
            if (started && !client_lost) {
                if (!result.ok) throw std::runtime_error("replica stream ended before completion");
                response.end_stream();
            }
            return;
        }
        if (result.ok) response.send(status, content_type.empty() ? "application/octet-stream" : content_type, buffered, headers);
        else if (!result.cancelled && !response.client_gone())
            response.send(502, "application/json",
                error_body("api_error", "replica " + std::to_string(replica) + " failed: " + result.error),
                {{"X-Strix-Replica", std::to_string(replica)}});
        return;
    }
}

} // namespace strix::serve

#pragma once

// Front end glue of the two-replica router: reads each replica's status route, asks ReplicaRouter where a request
// goes and passes the request and its response through unchanged. Model-agnostic: it never parses a response body.

#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/replica_router.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

struct ReplicaEndpoint {
    std::string host;
    std::uint16_t port = 0;
};
// "host:port" or "http://host:port[/]". URI IPv6 zones use "%25"; bare socket hosts use "%".
[[nodiscard]] std::optional<ReplicaEndpoint> parse_replica_endpoint(std::string_view text);
// Reject unresolved endpoints and overlaps between their resolved TCP destinations.
void require_distinct_replicas(const std::vector<ReplicaEndpoint>& replicas);

struct ReplicaProxyConfig {
    std::vector<ReplicaEndpoint> replicas;
    std::string status_target = "/api/status";
    int connect_timeout_ms = 2000;
    int status_timeout_ms = 2000;
    // Longest silence accepted from a replica on a forwarded request. 0 = no limit.
    int idle_timeout_ms = 0;
    std::size_t affinity_entries = 4096;
    std::size_t block_bytes = 1024;
};

class ReplicaProxy {
public:
    explicit ReplicaProxy(ReplicaProxyConfig config);

    // Before serving a policy-validated placement, verify a distinct worker instance per endpoint and
    // pin the socket address that reported it. Throws if any endpoint cannot establish its own identity.
    void admit_distinct_workers();

    // Reads every replica's status route once (blocking, at most status_timeout_ms each). A replica that stops
    // answering, or reports failed/draining, is skipped until a later refresh sees it healthy again; a replica that
    // was healthy and is not any more loses its recorded sessions and prefixes (it may have restarted).
    void refresh();

    // HttpHandler body. Routes owned by the router: GET /router/status, GET /health and /v1/health.
    // Everything else is forwarded to one replica. A request is sent again to another replica only when no byte of
    // it left this host (connection refused); after that a failure is answered with 502, never with a second attempt.
    void handle(const HttpRequest& request, HttpResponseWriter& response);

    [[nodiscard]] std::vector<ReplicaState> states() const;
    [[nodiscard]] Json status() const;

private:
    struct Counters { std::uint64_t forwarded = 0, failed = 0, by_session = 0, by_prefix = 0, by_load = 0; };
    void forward(const HttpRequest& request, HttpResponseWriter& response);
    ReplicaState probe(std::size_t replica, const std::string& connect_host) const;

    ReplicaProxyConfig config_;
    ReplicaRouter router_;
    mutable std::mutex mutex_;
    std::vector<std::string> connect_hosts_; // populated once by policy admission, before serving
    std::vector<ReplicaState> states_;
    std::vector<std::uint64_t> generations_; // advances when a slot's worker or availability changes
    std::vector<Counters> counters_;
    std::atomic<std::uint64_t> retried_{0}, unplaced_{0};
};

} // namespace strix::serve

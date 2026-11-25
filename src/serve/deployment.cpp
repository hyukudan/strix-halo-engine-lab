#include "strix/serve/deployment.hpp"

#include <cstdio>
#include <random>

namespace strix::serve {

void deployment_health(Json& health, const DeploymentCallbacks& callbacks) {
    if (callbacks.deployment_id) health.set("deployment_id", Json(callbacks.deployment_id()));
}

void deployment_status(Json& status, const DeploymentCallbacks& callbacks,
    const GenerationQueue& queue, std::uint64_t slots_total) {
    const auto q = queue.snapshot();
    const auto c = callbacks.capacity_provider ? callbacks.capacity_provider()
        : ServeCapacity{slots_total, q.running, std::nullopt, q.queued};
    Json capacity = Json::object();
    capacity.set("slots_total", Json(c.slots_total));
    capacity.set("slots_busy", Json(c.slots_busy));
    capacity.set("kv_bytes_free", c.kv_bytes_free ? Json(*c.kv_bytes_free) : Json());
    capacity.set("queue_depth", Json(c.queue_depth));
    status.set("capacity", std::move(capacity));
    // Shared by every listener in this process, including custom API status providers.
    static const std::string instance = [] {
        std::random_device entropy;
        char id[33]{};
        std::snprintf(id, sizeof(id), "%08x%08x%08x%08x", static_cast<unsigned>(entropy()),
            static_cast<unsigned>(entropy()), static_cast<unsigned>(entropy()), static_cast<unsigned>(entropy()));
        return std::string(id);
    }();
    status.set("worker_instance_id", Json(instance));
    if (queue.draining()) {
        const auto* state = status.find("state");
        if (!state || !state->is_string() || state->as_string() != "failed") status.set("state", Json("draining"));
        status.set("drained", Json(queue.drained()));
    }
}

bool handle_deployment_route(const HttpRequest& request, HttpResponseWriter& response,
    GenerationQueue& queue, const DeploymentCallbacks& callbacks, const std::function<void()>& on_drain) {
    if (request.method == "GET" && request.path == "/v1/deployment") {
        const auto body = callbacks.deployment_json ? callbacks.deployment_json() : std::string{};
        response.send(body.empty() ? 404 : 200, "application/json",
            body.empty() ? "{\"error\":{\"type\":\"not_found_error\",\"message\":\"no active deployment\"}}" : body);
        return true;
    }
    if (request.method == "POST" && request.path == "/admin/drain") {
        // Linearizes against submit under the queue lock; repeat requests are harmless.
        if (queue.begin_drain() && on_drain) on_drain();
        Json out = Json::object(); out.set("status", Json("draining"));
        out.set("drained", Json(queue.drained()));
        response.send(200, "application/json", out.dump());
        return true;
    }
    return false;
}

} // namespace strix::serve

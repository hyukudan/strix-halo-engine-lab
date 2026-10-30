#pragma once
#include "strix/serve/json.hpp"
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
namespace strix::serve {
struct SessionApiError : std::runtime_error {
    int status;
    std::string code;
    SessionApiError(int s, std::string c, const std::string& message)
        : std::runtime_error(message), status(s), code(std::move(c)) {}
};
inline Json session_error(const std::string& code, const std::string& message, int status = 0) {
    auto detail = Json::object(); detail.set("code", Json(code)); detail.set("message", Json(message));
    auto out = Json::object(); out.set("error", std::move(detail));
    if (status) out.set("_status", Json(std::int64_t(status)));
    return out;
}
// Translate the public routes into engine-owned session operations.
inline Json session_command(std::string_view method, std::string_view path, std::string_view body) {
    auto out = Json::object();
    std::string op, id;
    if (method == "GET" && path == "/v1/sessions") op = "list";
    else if (method == "GET" && path == "/v1/cache/stats") op = "stats";
    else if (path.starts_with("/v1/sessions/")) {
        auto tail = path.substr(13); const auto slash = tail.find('/');
        id = std::string(tail.substr(0, slash));
        if (method == "DELETE" && slash == std::string_view::npos) op = "delete";
        else if (method == "POST" && slash != std::string_view::npos) {
            op = std::string(tail.substr(slash + 1));
            if (op != "checkpoint" && op != "park" && op != "resume") op.clear();
        }
        if (!op.empty() && (id.empty() || id.size() > 128 || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        }))) throw SessionApiError(400, "invalid_request", "invalid session id");
    }
    if (op.empty()) throw SessionApiError(404, "not_found", "unknown session route");
    if (op == "checkpoint") {
        Json input;
        try { input = parse_json(body); } catch (const std::exception& e) { throw SessionApiError(400, "invalid_request", e.what()); }
        const auto* d = input.find("durability");
        if (!input.is_object() || !d || !d->is_string() || (d->as_string() != "ram" && d->as_string() != "disk"))
            throw SessionApiError(400, "invalid_request", "durability must be ram or disk");
        out.set("durability", Json(d->as_string()));
        if (d->as_string() == "ram") op = "ram_checkpoint";
    }
    out.set("op", Json(op)); out.set("id", Json(id)); return out;
}
// Live scheduler metadata supersedes retained generations. No host checkpoint or
// durable generation represents a running frontier; unknown byte counts are null.
inline void session_inventory_live(std::map<std::string, Json>& rows, const std::string& id,
                                   std::uint64_t consumed, std::optional<std::uint32_t> pending,
                                   std::uint64_t emitted, bool active) {
    if (id.empty()) return;
    auto row = Json::object();
    row.set("id", Json(id)); row.set("position", Json(consumed)); row.set("consumed", Json(consumed));
    row.set("pending", pending ? Json(static_cast<std::uint64_t>(*pending)) : Json());
    row.set("emitted", Json(emitted)); row.set("bytes", Json());
    row.set("state", Json(active ? "active" : "parked"));
    row.set("durability", Json("none"));
    auto engine = Json::object(); engine.set("generation", Json()); row.set("engine", std::move(engine));
    rows.insert_or_assign(id, std::move(row));
}
inline SessionApiError session_failure(const std::string& message) {
    for (const auto& item : {std::pair{"session_not_found", 404}, {"session_budget", 507}, {"session_incompatible", 409}, {"session_corrupt", 422}})
        if (message.find(item.first) != std::string::npos) return {item.second, item.first, message};
    if (message.find("budget") != std::string::npos || message.find("allocation") != std::string::npos) return {507, "session_budget", message};
    if (message.find("checksum") != std::string::npos || message.find("truncated") != std::string::npos) return {422, "session_corrupt", message};
    return {409, "session_incompatible", message};
}
} // namespace strix::serve

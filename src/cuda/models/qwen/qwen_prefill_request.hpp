// Request body of the CUDA Qwen prefill service (apps/cuda_qwen_prefill_service_main.cu, POST /v1/prefill), parsed
// on the host so the service's refusals are testable without a device. Every refusal is an exception (the service
// answers 400): a body that is not an object, a missing `tokens` / `handoff`, a wrong type, a value out of range.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/serve/json.hpp"

namespace strix::qwen_http {

struct PrefillRequest {
    std::vector<std::uint32_t> tokens;
    std::string handoff, api_key, codec;
    bool has_codec = false;
    bool trace = false, tail_large = false, has_schedule = false;
    std::uint32_t final_rows = 0;
    bool has_final_rows = false;
    std::uint64_t local_rows = 0;
    std::string node_completions;
    serve::Json session;
};

// Fills `r` from a parsed body; `max_block_rows` bounds final_block_rows. Throws std::invalid_argument (or the JSON
// reader's type error) on a malformed body; never dereferences an absent key.
inline void parse_prefill_request(const serve::Json& j, std::uint32_t max_block_rows, PrefillRequest& r) {
    if (!j.is_object()) throw std::invalid_argument("body is not an object");
    const serve::Json* tokens = j.find("tokens");
    const serve::Json* handoff = j.find("handoff");
    if (tokens == nullptr) throw std::invalid_argument("tokens");
    if (handoff == nullptr) throw std::invalid_argument("handoff");
    for (const auto& t : tokens->as_array()) {
        const std::int64_t id = t.as_int();
        if (id < 0 || id > static_cast<std::int64_t>(UINT32_MAX)) throw std::invalid_argument("tokens");
        r.tokens.push_back(static_cast<std::uint32_t>(id));
    }
    r.handoff = handoff->as_string();
    if (const auto* x = j.find("session")) r.session = *x;
    if (const auto* x = j.find("handoff_api_key")) r.api_key = x->as_string();
    if (const auto* x = j.find("handoff_codec")) { r.codec = x->as_string(); r.has_codec = true; }
    if (const auto* x = j.find("local_rows")) {
        const std::int64_t rows = x->as_int();
        if (rows < 0) throw std::invalid_argument("local_rows");
        r.local_rows = static_cast<std::uint64_t>(rows);
    }
    if (const auto* x = j.find("trace")) r.trace = x->as_bool();
    if (const auto* x = j.find("block_schedule")) {
        const auto& schedule = x->as_string();
        if (schedule != "default" && schedule != "tail-large") throw std::invalid_argument("block_schedule");
        r.has_schedule = true;
        r.tail_large = schedule == "tail-large";
    }
    if (const auto* x = j.find("final_block_rows")) {
        const std::int64_t n = x->as_int();
        if (n < 0 || n > static_cast<std::int64_t>(max_block_rows)) throw std::invalid_argument("final_block_rows");
        r.has_final_rows = true;
        r.final_rows = static_cast<std::uint32_t>(n);
    }
    if (const auto* x = j.find("node_completions")) r.node_completions = x->as_string();
}

}  // namespace strix::qwen_http

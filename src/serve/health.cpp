#include "strix/serve/health.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <thread>

namespace strix::serve {
namespace {
Json object(std::initializer_list<Json::Member> fields) { return Json(Json::Object(fields)); }
template<class T> Json number(const std::optional<T>& value) { return value ? Json(*value) : Json(); }
Json age(const std::optional<double>& value) {
    return value && std::isfinite(*value) && *value >= 0 ? Json(*value) : Json();
}
bool hex(std::string_view s, std::size_t length) {
    return s.size() == length && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
Json digest(const Json* v, std::size_t length = 64) {
    return v && v->is_string() && hex(v->as_string(), length) ? *v : Json();
}
Json choice(const Json* v, std::initializer_list<std::string_view> choices) {
    if (v && v->is_string()) for (auto s : choices) if (v->as_string() == s) return *v;
    return {};
}
Json version(const Json* v) {
    if (!v || !v->is_string()) return {};
    const auto& s = v->as_string();
    return !s.empty() && s.size() <= 32 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || c == '.';
    }) ? *v : Json();
}
Json build(const Json& input) {
    if (!input.is_object()) return {};
    auto out = object({{"format", "strix-build"}, {"revision", 1},
        {"status", choice(input.find("status"), {"available", "unavailable"})},
        {"version", version(input.find("version"))}, {"git_sha", digest(input.find("git_sha"), 40)},
        {"flags_evidence", choice(input.find("flags_evidence"), {"configured", "compiled"})}});
    const auto* dirty = input.find("dirty");
    out.set("dirty", dirty && dirty->is_bool() ? *dirty : Json());
    for (const char* key : {"compiler_id", "hip_compiler_id"})
        out.set(key, choice(input.find(key), {"GNU", "Clang", "AppleClang", "MSVC", "AMDClang"}));
    for (const char* key : {"compiler_version", "hip_compiler_version", "cmake_version"})
        out.set(key, version(input.find(key)));
    Json targets;
    if (const auto* list = input.find("target_isa"); list && list->is_array()) {
        targets = Json::array();
        for (const auto& item : list->as_array()) {
            const auto target = choice(&item, {"gfx1151", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx1201", "gfx942", "gfx90a"});
            if (!target.is_null()) targets.push(target);
        }
    }
    out.set("target_isa", targets);
    out.set("git_sha_source", choice(input.find("git_sha_source"), {"git", "supplied"}));
    Json options;
    if (const auto* source = input.find("options"); source && source->is_object()) {
        options = Json::object();
        for (const auto* key : {"STRIX_EMBED_BUILD_INFO", "STRIX_ENABLE_HIP_RUNTIME", "STRIX_WARNINGS_AS_ERRORS",
             "STRIX_FAMILY_DEEPSEEK41", "STRIX_FAMILY_MINIMAX", "STRIX_ENABLE_CUDA_DS41", "STRIX_ENABLE_CUDA_TRELLIS",
             "STRIX_USE_OPENSSL_SHA256", "STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME", "STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE",
             "STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS", "STRIX_CHECK_KERNEL_SCRATCH", "STRIX_CUDA_DS41_ONLY"})
            if (const auto* value = source->find(key); value && value->is_bool()) options.set(key, *value);
    }
    out.set("options", options);
    return out;
}
Json assets(const Json& input) {
    auto roles = Json::object();
    const bool full = choice(input.find("verification"), {"full"}).is_string() &&
        !digest(input.find("manifest_digest")).is_null();
    // A plain total + validity flag (not std::optional) keeps GCC 13 -O2 free of maybe-uninitialized.
    std::uint64_t total = 0;
    bool valid = full;
    if (full) {
        if (const auto* list = input.find("roles"); list && list->is_object()) {
            for (const auto* role : {"target", "target_shard", "draft", "vocabulary", "tokenizer", "profile", "calibration"}) {
                if (const auto* entry = list->find(role)) {
                    roles.set(role, digest(entry->find("digest")));
                    const auto* n = entry->find("count");
                    if (!n || !n->is_number() || n->as_int() < 0) { valid = false; break; }
                    total += static_cast<std::uint64_t>(n->as_int());
                }
            }
        } else valid = false;
    }
    return object({{"manifest_sha256", full ? digest(input.find("manifest_digest")) : Json()},
        {"verification", full ? Json("full") : Json()}, {"coverage", full ? "complete" : "unbound"},
        {"file_count", valid ? Json(total) : Json()}, {"roles", full ? roles : Json()}});
}
const char* state_name(BackendState state) {
    switch (state) {
    case BackendState::starting: return "starting";
    case BackendState::responsive: return "responsive";
    case BackendState::probe_pending: return "probe_pending";
    case BackendState::unresponsive: return "unresponsive";
    case BackendState::failed: return "failed";
    case BackendState::draining: return "draining";
    case BackendState::unchecked: return "unchecked";
    }
    return "unchecked";
}
const char* proof_name(BackendProof proof) {
    switch (proof) {
    case BackendProof::control_only: return "control-only";
    case BackendProof::work_completion: return "completion-boundary";
    case BackendProof::idle_scratch_event: return "idle-scratch-event";
    case BackendProof::none: return nullptr;
    }
    return nullptr;
}
} // namespace
BackendProbeSlot::BackendProbeSlot(std::function<void()> wake, Clock clock, Pause pause)
    : wake_(std::move(wake)), clock_(std::move(clock)), pause_(std::move(pause)) {
    if (!clock_) clock_ = [] { return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    if (!pause_) pause_ = [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); };
}
void BackendProbeSlot::acknowledge(std::uint64_t generation, BackendProof proof) noexcept {
    // Only requested boundaries read a clock. Low two bits hold the proof kind.
    const auto now = static_cast<std::uint64_t>(clock_());
    const std::uint64_t kind = proof == BackendProof::work_completion ? 2U :
        proof == BackendProof::idle_scratch_event ? 3U : 1U;
    stamp_.store((now & ~std::uint64_t{3}) | kind, std::memory_order_release);
    owner_generation_ = generation;
    acknowledged_.store(generation, std::memory_order_release);
}
BackendHealth BackendProbeSlot::cached(std::int64_t now) const {
    BackendHealth out;
    out.state = BackendState::probe_pending;
    out.required_ranks = 1;
    out.responsive_ranks = 0;
    const auto stamp = stamp_.load(std::memory_order_acquire);
    if (stamp) {
        const auto time = static_cast<std::int64_t>(stamp & ~std::uint64_t{3});
        const double age_seconds = now >= time ? static_cast<double>(now - time) / 1e9 : 2.0;
        out.proof = (stamp & 3U) == 2U ? BackendProof::work_completion :
            (stamp & 3U) == 3U ? BackendProof::idle_scratch_event : BackendProof::control_only;
        out.probe_age_seconds = age_seconds;
        if (out.proof != BackendProof::control_only) {
            out.last_completion_age_seconds = age_seconds;
            if (age_seconds <= 1.0) {
                out.state = BackendState::responsive;
                out.responsive_ranks = 1;
            }
        }
    }
    if (failed_.load(std::memory_order_acquire)) out.state = BackendState::failed;
    return out;
}
BackendHealth BackendProbeSlot::snapshot() const {
    const auto start = clock_();
    auto out = cached(start);
    if (out.state == BackendState::failed ||
        (out.probe_age_seconds && *out.probe_age_seconds <= 1.0)) return out;
    auto generation = acknowledged_.load(std::memory_order_acquire);
    // Zero is reserved: generations never wrap or reset during this slot's lifetime.
    if (generation == std::numeric_limits<std::uint64_t>::max()) {
        failed_.store(true, std::memory_order_release);
        return cached(start);
    }
    // A racing reader joins the same generation; it never appends another request.
    if (requested_.compare_exchange_strong(generation, generation + 1U,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
        ++generation;
        if (wake_) wake_();
    }
    for (;;) {
        const auto acknowledged = acknowledged_.load(std::memory_order_acquire);
        const auto now = clock_();
        out = cached(now);
        if (out.state == BackendState::failed || acknowledged >= generation || now - start >= 100000000) return out;
        pause_();
    }
}
bool serve_health_record_enabled() noexcept {
    const auto* value = std::getenv("STRIX_SERVE_HEALTH_RECORD");
    return !value || std::string_view(value) != "0";
}
HealthResponse render_health(const ServeHealthRecord& c, const BackendHealth& evidence,
    std::string_view service_state, std::string_view model, const QueueSnapshot& load, std::string_view capabilities_id) {
    auto backend = c.backend_probe_enabled ? evidence : BackendHealth{};
    if (service_state == "failed") backend.state = BackendState::failed;
    else if (service_state == "draining") backend.state = BackendState::draining;
    const bool failed = backend.state == BackendState::failed, draining = backend.state == BackendState::draining;
    const bool ready = service_state == "ready" && backend.state == BackendState::responsive &&
        (backend.proof == BackendProof::work_completion || backend.proof == BackendProof::idle_scratch_event) &&
        backend.probe_age_seconds && *backend.probe_age_seconds >= 0 && *backend.probe_age_seconds <= 1.0 &&
        backend.last_completion_age_seconds && *backend.last_completion_age_seconds >= 0 &&
        *backend.last_completion_age_seconds <= 1.0 && backend.required_ranks && *backend.required_ranks > 0 &&
        backend.responsive_ranks == backend.required_ranks;
    const bool interim = c.backend_probe_enabled && c.interim_unchecked_http_success &&
        (backend.state == BackendState::unchecked || backend.state == BackendState::probe_pending) && !failed && !draining;
    auto preflight = object({{"state", "unavailable"}, {"warnings", Json::array()}, {"advisories", Json::array()},
        {"sample_age_seconds", Json()}});
    if (c.preflight && c.preflight_assessment) {
        const auto startup = parse_json(render_preflight(*c.preflight, *c.preflight_assessment));
        preflight.set("state", c.preflight_assessment->warnings.empty() ?
            (c.preflight_assessment->advisories.empty() ? "ok" : "advisory") : "warning");
        // Same split as the startup record: advisories are facts, never alert triggers.
        preflight.set("warnings", *startup.find("warnings"));
        preflight.set("advisories", *startup.find("advisories"));
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (c.preflight->end_ns && now >= 0 && static_cast<std::uint64_t>(now) >= c.preflight->end_ns)
            preflight.set("sample_age_seconds", Json(static_cast<double>(static_cast<std::uint64_t>(now) - c.preflight->end_ns) / 1e9));
    }
    const auto info = build(c.build_info);
    const auto* proof = proof_name(backend.proof);
    auto out = object({{"format", "strix-service-health"}, {"revision", 1},
        {"status", failed ? "failed" : draining ? "draining" : "ok"}, {"model", Json(model)}, {"ready", ready},
        {"release", object({{"version", info.find("version") ? *info.find("version") : Json()},
            {"git_sha", info.find("git_sha") ? *info.find("git_sha") : Json()},
            {"build_info_sha256", hex(c.build_info_sha256, 64) ? Json(c.build_info_sha256) : Json()}, {"build_info", info}})},
        {"identity", object({{"model_key", Json()}, {"content_identity", Json()}, {"assets", assets(c.asset_verification)}})},
        {"capacity", object({{"context_tokens", number(c.context_tokens)}, {"slots", number(c.slots)}})},
        {"capabilities_id", capabilities_id.empty() ? Json() : Json(capabilities_id)},
        {"capabilities_url", capabilities_id.empty() ? Json() : Json("/v1/capabilities")},
        {"execution", object({{"decode_profile", c.decode_profile ? Json(numerical_profile_name(*c.decode_profile)) : Json()},
            {"prefill_profile", c.prefill_profile ? Json(numerical_profile_name(*c.prefill_profile)) : Json()},
            {"prefill_tile_rows", number(c.prefill_tile_rows)},
            {"draft_profile", (c.draft_profile == "base" || c.draft_profile == "distill-d" || c.draft_profile == "distill-g") ? Json(c.draft_profile) : Json()}})},
        {"storage", object({{"prefix", object({{"enabled", c.prefix_enabled}, {"host_limit_bytes", number(c.prefix_host_bytes)},
            {"disk_enabled", c.prefix_disk_enabled}, {"disk_limit_bytes", number(c.prefix_disk_bytes)}})},
            {"sessions", object({{"enabled", false}})}, {"resident_pool", object({{"supported", false}})}})},
        {"load", object({{"queued", load.queued}, {"running", load.running}, {"sample_age_seconds", Json()}})},
        {"backend", object({{"state", state_name(backend.state)}, {"proof", proof ? Json(proof) : Json()},
            {"probe_age_seconds", age(backend.probe_age_seconds)}, {"last_completion_age_seconds", age(backend.last_completion_age_seconds)},
            {"required_ranks", number(backend.required_ranks)}, {"responsive_ranks", number(backend.responsive_ranks)}})},
        {"preflight", preflight}});
    return {ready || interim ? 200 : 503, std::move(out)};
}
} // namespace strix::serve

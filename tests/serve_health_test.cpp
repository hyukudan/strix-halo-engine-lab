#include "strix/serve/api.hpp"
#include "qwen_service_health.hpp"
#include "rtx_state_import_routes.hpp"

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <fstream>
#include <random>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <limits>
#include <set>

namespace {
using namespace strix::serve;
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
const Json& at(const Json& j, const char* key) {
    const auto* v = j.find(key);
    if (!v) throw std::runtime_error(std::string("missing ") + key);
    return *v;
}
struct Writer final : HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int code, std::string_view, std::string_view text,
        const std::vector<std::pair<std::string, std::string>>&) override { status = code; body = text; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};
void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
}
Writer get(ApiService& api, const char* path = "/health", const char* token = nullptr) {
    HttpRequest request; request.method = "GET"; request.path = path;
    if (token) request.headers.emplace_back("authorization", std::string("Bearer ") + token);
    Writer out; api.handle(request, out); return out;
}
class ScriptedProbe final : public BackendProbe {
public:
    BackendHealth value;
    mutable unsigned reads = 0;
    BackendHealth snapshot() const override { ++reads; return value; }
};
// Private-member access is confined to this test TU. Explicit instantiation permits
// naming private members without a production seeding API or altered class definition.
template<class Tag, typename Tag::Type Member> struct TestMember {
    friend typename Tag::Type member(Tag) { return Member; }
};
struct RequestedTag {
    using Type = std::atomic<std::uint64_t> BackendProbeSlot::*;
    friend Type member(RequestedTag);
};
struct AcknowledgedTag {
    using Type = std::atomic<std::uint64_t> BackendProbeSlot::*;
    friend Type member(AcknowledgedTag);
};
struct OwnerTag {
    using Type = std::uint64_t BackendProbeSlot::*;
    friend Type member(OwnerTag);
};
template struct TestMember<RequestedTag, &BackendProbeSlot::requested_>;
template struct TestMember<AcknowledgedTag, &BackendProbeSlot::acknowledged_>;
template struct TestMember<OwnerTag, &BackendProbeSlot::owner_generation_>;

// Deadlines are hang safeguards, never latency/performance assertions. On ordinary
// failures callers release their rendezvous and shut down before joining. If an
// internal queue deadlock prevents cleanup, the watchdog terminates this test process.
class Watchdog {
    std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false;
    std::thread thread_;
public:
    Watchdog() : thread_([this] {
        std::unique_lock lock(mutex_);
        if (!changed_.wait_for(lock, std::chrono::seconds(20), [&] { return done_; })) {
            std::cerr << "FAIL: concurrency watchdog expired\n";
            std::_Exit(1);
        }
    }) {}
    ~Watchdog() {
        { std::lock_guard lock(mutex_); done_ = true; }
        changed_.notify_all();
        thread_.join();
    }
};
struct Rendezvous {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0;
    bool released = false;
    void arrive() {
        std::lock_guard lock(mutex);
        ++arrived;
        changed.notify_all();
    }
    bool wait(unsigned count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] { return arrived >= count; });
    }
    void park() {
        std::unique_lock lock(mutex);
        ++arrived;
        changed.notify_all();
        if (!changed.wait_for(lock, std::chrono::seconds(5), [&] { return released; })) {
            released = true;
            changed.notify_all();
        }
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};

void probe_slot_tests() {
    struct CounterSpy {
        mutable unsigned loads = 0;
        std::uint64_t load(std::memory_order order) const {
            check(order == std::memory_order_relaxed, "owner load is relaxed");
            ++loads; return 0;
        }
    } counter;
    unsigned callbacks = 0;
    if (detail::probe_generation(counter, 0)) ++callbacks;
    check(counter.loads == 1 && callbacks == 0, "no-probe path exactly one load and no callback");
    std::int64_t now = 4000000000;
    unsigned clocks = 0, pauses = 0, wakes = 0;
    BackendProbeSlot* slot_ptr = nullptr;
    bool acknowledge = false;
    BackendProbeSlot slot([&] { ++wakes; }, [&] { ++clocks; return now; }, [&] {
        ++pauses; now += 10000000;
        if (acknowledge) slot_ptr->boundary(BackendProof::work_completion);
    });
    slot_ptr = &slot;
    slot.boundary(BackendProof::work_completion);
    check(clocks == 0 && wakes == 0 && pauses == 0, "unrequested boundary has no clock/wake/wait");
    auto health = slot.snapshot();
    check(health.state == BackendState::probe_pending && now == 4100000000 && pauses == 10 && wakes == 1,
        "no ACK has injectable 100 ms wait bound");
    health = slot.snapshot();
    check(wakes == 1 && now == 4200000000, "timeouts retain one pending generation");
    acknowledge = true;
    health = slot.snapshot();
    check(health.state == BackendState::responsive && wakes == 1, "late ACK services original generation");
    const auto paused = pauses;
    health = slot.snapshot();
    check(health.state == BackendState::responsive && pauses == paused && wakes == 1, "fresh proof cached without request");
    now += 1000000001;
    acknowledge = false;
    health = slot.snapshot();
    check(health.state == BackendState::probe_pending && wakes == 2 && *health.probe_age_seconds > 1.0,
        "stale proof cannot establish readiness");
    slot.boundary(BackendProof::control_only);
    health = slot.snapshot();
    check(health.state == BackendState::probe_pending && health.proof == BackendProof::control_only &&
        !health.last_completion_age_seconds, "idle control ACK is explicitly not device proof");
    slot.fail();
    check(slot.snapshot().state == BackendState::failed, "failure latched independently of cached proof");

    std::int64_t failure_time = 8000000000;
    BackendProbeSlot* failing_ptr = nullptr;
    BackendProbeSlot failing({}, [&] { return failure_time; }, [&] {
        failure_time += 1000000;
        failing_ptr->fail();
    });
    failing_ptr = &failing;
    check(failing.snapshot().state == BackendState::failed && failure_time == 8001000000,
        "failure during bounded wait returns without waiting for timeout");

    unsigned exhausted_wakes = 0;
    BackendProbeSlot exhausted([&] { ++exhausted_wakes; }, [] { return 4000000000LL; });
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    (exhausted.*member(RequestedTag{})).store(maximum);
    (exhausted.*member(AcknowledgedTag{})).store(maximum);
    exhausted.*member(OwnerTag{}) = maximum;
    for (unsigned i = 0; i < 3; ++i) {
        const auto evidence = exhausted.snapshot();
        const auto response = render_health({}, evidence, "ready", "model", {});
        check(evidence.state == BackendState::failed && response.status == 503,
            "exhaustion permanently fails closed");
        check((exhausted.*member(RequestedTag{})).load() == maximum &&
            (exhausted.*member(AcknowledgedTag{})).load() == maximum &&
            exhausted.pending_generation() == 0 && exhausted_wakes == 0,
            "exhaustion never publishes zero or wakes owner");
        exhausted.boundary(BackendProof::work_completion);
    }

    Watchdog watchdog;
    // Concurrent real HTTP readers wait on one owner, with no generated jobs or observation arming.
    GenerationQueue queue;
    std::atomic<unsigned> wake_count{0}, clock_count{0};
    BackendProbeSlot concurrent([&] { ++wake_count; }, [&] {
        ++clock_count;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    });
    auto record = std::make_shared<ServeHealthRecord>();
    record->interim_unchecked_http_success = true;
    ApiConfig config; config.health_record = record;
    config.health_provider = [&] { return concurrent.snapshot(); };
    ApiService api(config, {}, queue, 8192, false); api.set_state("ready");
    constexpr unsigned readers = 24;
    Rendezvous start;
    std::atomic<unsigned> pending{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < readers; ++i) threads.emplace_back([&] {
        start.park();
        const auto response = get(api);
        const auto body = parse_json(response.body);
        if (response.status == 200 && !at(body, "ready").as_bool() &&
            at(at(body, "backend"), "state").as_string() == "probe_pending") ++pending;
    });
    observe_disarm();
    check(start.wait(readers), "all pending readers rendezvous");
    start.release();
    for (auto& reader : threads) reader.join();
    check(pending == readers && wake_count == 1, "many HTTP readers coalesce into one pending request");
    const auto before = clock_count.load();
    const auto generation = concurrent.pending_generation();
    check(generation != 0, "batch owner captures pending generation before step");
    concurrent.complete(generation, BackendProof::work_completion);
    check(concurrent.pending_generation() == 0, "batch ACK clears captured generation");
    check(clock_count == before + 1, "owner ACK uses one clock and never waits for readers");
    check(at(parse_json(get(api).body), "ready").as_bool(), "owner completion makes API ready");
    check(queue.snapshot().queued == 0 && queue.snapshot().running == 0 && !observe_armed(),
        "health consumes no jobs and never arms observation");
    concurrent.fail(); check(get(api).status == 503, "latched owner failure overrides interim 200");

    Rendezvous rendezvous;
    std::atomic<unsigned> ack_wakes{0}, responsive{0};
    std::atomic<std::int64_t> controlled_now{4000000000};
    BackendProbeSlot live_probe([&] { ++ack_wakes; }, [&] { return controlled_now.load(); }, [&] {
        thread_local bool first_pause = true;
        if (first_pause) { first_pause = false; rendezvous.park(); }
        else std::this_thread::yield();
    });
    config.health_provider = [&] { return live_probe.snapshot(); };
    ApiService live_api(config, {}, queue, 8192, false); live_api.set_state("ready");
    threads.clear();
    for (unsigned i = 0; i < readers; ++i) threads.emplace_back([&] {
        const auto response = get(live_api);
        if (response.status == 200 && at(parse_json(response.body), "ready").as_bool()) ++responsive;
    });
    const bool all_waiting = rendezvous.wait(readers);
    check(all_waiting, "all ACK readers rendezvous under controlled clock");
    if (all_waiting) live_probe.boundary(BackendProof::work_completion);
    else live_probe.fail(); // release all readers even if the rendezvous failed
    rendezvous.release();
    for (auto& reader : threads) reader.join();
    check(ack_wakes == 1 && responsive == readers, "concurrent HTTP readers share one owner ACK");
}
void queue_control_tests() {
    Watchdog watchdog;
    struct RecordingEngine final : GenerationEngine {
        std::vector<std::uint32_t> seen;
        GenerationStats generate(GenerationJob& job) override {
            seen.push_back(job.prompt.at(0));
            GenerationStats stats; stats.finish = "stop"; return stats;
        }
        std::uint64_t context_tokens() const override { return 8192; }
    };
    for (const bool batch : {false, true}) {
        // Use the same engine and completion path for each owner API.
        const auto own = [batch](GenerationQueue& queue, RecordingEngine& engine) {
            if (!batch) { queue.run(engine); return; }
            while (!queue.stopping()) {
                if (auto job = queue.take(true)) queue.finish(*job, engine.generate(job->job));
            }
        };
        // Retain the queue/probe integration check with a controlled clock.
        GenerationQueue idle;
        RecordingEngine idle_engine;
        BackendProbeSlot idle_probe([&] { idle.wake_control(); }, [] { return 4000000000LL; },
            [] { std::this_thread::yield(); });
        idle.set_idle_control([&] { idle_probe.boundary(BackendProof::control_only); });
        std::thread idle_owner([&] { own(idle, idle_engine); });
        const auto idle_health = idle_probe.snapshot();
        idle.shutdown();
        idle_owner.join();
        check(idle_health.state == BackendState::probe_pending && idle_health.proof == BackendProof::control_only &&
            idle_engine.seen.empty(), "idle owner wake returns control-only proof without generation");

        GenerationQueue fifo(128);
        RecordingEngine engine;
        std::vector<std::shared_ptr<GenerationTicket>> tickets;
        std::vector<std::uint32_t> expected;
        unsigned cancelled = 0;
        for (std::uint32_t id = 0; id < 100; ++id) {
            GenerationJob job; job.prompt = {id};
            auto ticket = fifo.submit(std::move(job));
            if (id % 7 == 0) { ticket->cancel(); ++cancelled; }
            else expected.push_back(id);
            tickets.push_back(std::move(ticket));
        }
        bool control_after_work = false;
        unsigned controls = 0;
        fifo.set_idle_control([&] {
            ++controls;
            const auto load = fifo.snapshot(); // callback must run outside queue mutex
            control_after_work = engine.seen == expected && load.queued == 0 && load.running == 0 &&
                load.completed == expected.size() && load.cancelled == cancelled;
            fifo.shutdown(); // callback re-entry must not deadlock
        });
        for (unsigned i = 0; i < 100; ++i) fifo.wake_control();
        std::thread owner([&] { own(fifo, engine); });
        owner.join();
        check(controls == 1 && control_after_work, "FIFO work and cancelled jobs precede coalesced control");
        for (std::size_t i = 0; i < tickets.size(); ++i) {
            bool done = false;
            (void)tickets[i]->wait(std::chrono::milliseconds(0), done);
            check(done && tickets[i]->stats().finish == (i % 7 == 0 ? "cancelled" : "stop"),
                "every queued ticket completes with expected cancellation status");
        }
        check(!fifo.submit({}), "submission rejected after callback shutdown");

        // First control is pending before the wait. Every subsequent wake follows
        // an individually acknowledged callback and races its return to the wait.
        GenerationQueue racing;
        RecordingEngine no_work;
        Rendezvous callbacks;
        constexpr unsigned iterations = 128;
        racing.set_idle_control([&] {
            (void)racing.snapshot();
            callbacks.arrive();
        });
        racing.wake_control();
        std::thread waiter([&] { own(racing, no_work); });
        bool all_wakes = true;
        for (unsigned i = 1; i <= iterations; ++i) {
            if (!callbacks.wait(i)) { all_wakes = false; break; }
            if (i < iterations) racing.wake_control();
        }
        // Always clean up, including a failed handshake. No thread is detached.
        racing.shutdown();
        waiter.join();
        check(all_wakes && callbacks.arrived == iterations && no_work.seen.empty(),
            "bounded acknowledged wake/wait races lose no control and create no work");
        check(!racing.submit({}), "submission rejected after wake/wait shutdown");

        // Shutdown suppresses a pending control and cancels queued work, even
        // when both predicates were established before the owner begins waiting.
        GenerationQueue stopped;
        RecordingEngine stopped_engine;
        unsigned stopped_controls = 0;
        stopped.set_idle_control([&] { ++stopped_controls; (void)stopped.snapshot(); });
        auto dropped = stopped.submit({});
        stopped.wake_control();
        stopped.shutdown();
        std::thread stopped_owner([&] {
            if (batch) (void)stopped.take(true);
            else stopped.run(stopped_engine);
        });
        stopped_owner.join();
        bool done = false;
        (void)dropped->wait(std::chrono::milliseconds(0), done);
        check(stopped_controls == 0 && stopped_engine.seen.empty() && done &&
            dropped->stats().finish == "cancelled" && stopped.snapshot().queued == 0,
            "pending-control shutdown suppresses callback and cancels queued ticket");
        check(!stopped.submit({}), "submission rejected after pending-control shutdown");
    }
}

} // namespace
int main(int argc, char** argv) {
    // CPU startup harness: the production helper and the decoder's generic outer-handler contract.
    if (argc > 1 && std::string_view(argv[1]).starts_with("--startup-")) {
        try {
            if (std::string_view(argv[1]) != "--startup-check") {
                (void)strix::decode_cli::release_asset_boundary([&]() -> Json {
                    if (std::string_view(argv[1]) == "--startup-denied")
                        throw std::filesystem::filesystem_error("secret-token\nprivate-host", "/secret/account",
                            std::make_error_code(std::errc::permission_denied));
                    throw std::runtime_error("secret-token\nprivate-host /secret/account");
                });
            } else {
                strix::decode_cli::Options options;
                options.model_path = argv[2]; options.release_manifest = argv[3];
                options.mtp.draft_vocab = argv[4];
                (void)strix::decode_cli::verify_http_release_assets(options);
            }
            std::cerr << "model/device admission\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "strix-hip-decode-token: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc > 1) {
        auto fixtures = strix::serve::Json::array();
        fixtures.push(strix::serve::render_health({}, {}, "ready", "public-model", {}).body);
        strix::decode_cli::Options options;
        strix::StartupPreflight startup;
        const auto assets = strix::serve::parse_json(
            R"({"verification":"full","manifest_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","roles":{"target":{"count":1,"digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}})");
        const auto record = strix::decode_cli::http_health_record(options, 1, startup, assets);
        fixtures.push(strix::serve::render_health(*record,
            {strix::serve::BackendState::responsive, strix::serve::BackendProof::work_completion, 0.1, 0.1, 1, 1},
            "ready", "public-model", {}).body);
        GenerationQueue fixture_queue;
        for (const bool enabled : {false, true}) {
            ApiConfig config; config.model_id = "public-model"; config.capabilities = enabled;
            config.health_record = record;
            config.capabilities_extra = [](Json& document) { document.set("engine_feature", Json(true)); };
            ApiService api(config, {}, fixture_queue, 8192, false); api.set_state("ready");
            fixtures.push(parse_json(get(api).body));
        }
        std::cout << fixtures.dump();
        return 0;
    }
    using namespace strix::serve;
    env("STRIX_SERVE_HEALTH_RECORD", nullptr);
    probe_slot_tests();
    queue_control_tests();
    GenerationQueue queue;
    ApiConfig legacy;
    legacy.model_id = "public-model-\"seed";
    ApiService old(legacy, {}, queue, 8192, false);
    const auto cap_document = parse_json(get(old, "/v1/capabilities").body);
    const auto cap_id = at(cap_document, "capabilities_id").as_string();
    const auto old_body = std::string(R"({"status":"ok","model":"public-model-\"seed","capabilities_id":")") + cap_id + "\"}";
    check(get(old).body == old_body && get(old, "/v1/health").body == old_body, "legacy aliases byte identical");
    auto c = std::make_shared<ServeHealthRecord>();
    c->context_tokens = 8192; c->slots = 1;
    c->prefix_host_bytes = 0;
    c->build_info = parse_json(R"({"version":"0.1.0","git_sha":"/secret/account","compiler_id":"secret-host","argv":"secret-token","hostname":"private-host"})");
    c->asset_verification = parse_json(R"({"model_key":"private-host","member":"/secret/account"})");
    c->draft_profile = "/secret/account";
    ScriptedProbe probe;
    ApiConfig config = legacy;
    config.health_record = c;
    config.health_provider = [&] { return probe.snapshot(); };
    config.api_key = "secret-token";
    config.engine_description = "private-host /secret/account secret-token";
    ApiService api(config, {}, queue, 8192, false);
    api.set_state("ready");
    observe_disarm(); const auto period = observe_period();
    const auto result = get(api);
    check(result.status == 503, "unchecked is not device ready");
    check(!observe_armed() && observe_period() == period && probe.reads == 1, "health never arms");
    const auto j = parse_json(result.body);
    std::set<std::string> keys;
    for (const auto& [key, value] : j.as_object()) { (void)value; keys.insert(key); }
    check(keys == std::set<std::string>{"format", "revision", "status", "model", "ready", "release", "identity", "capacity",
        "capabilities_id", "capabilities_url", "execution", "storage", "load", "backend", "preflight"}, "exact envelope keys");
    check(at(j, "format").as_string() == "strix-service-health" && at(j, "revision").as_int() == 1, "schema identity");
    check(at(at(j, "backend"), "probe_age_seconds").is_null(), "unknown age null");
    check(at(at(j, "backend"), "responsive_ranks").is_null(), "unknown ranks null");
    check(at(at(j, "load"), "queued").as_int() == 0, "measured zero retained");
    check(at(at(j, "execution"), "prefill_tile_rows").is_null(), "unknown capacity null");
    check(!at(at(at(j, "storage"), "sessions"), "enabled").as_bool(), "unsupported storage false");
    check(at(at(at(j, "storage"), "prefix"), "host_limit_bytes").as_int() == 0, "disabled budget zero");
    const auto document = parse_json(get(api, "/v1/capabilities", "secret-token").body);
    Json canonical = Json::object();
    for (const auto& [key, value] : document.as_object()) if (key != "capabilities_id") canonical.set(key, value);
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : canonical.dump()) { hash ^= byte; hash *= 1099511628211ULL; }
    char hex[17]; std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
    check(at(j, "capabilities_id").as_string() == hex && at(document, "capabilities_id").as_string() == hex,
        "health reuses capabilities document FNV-1a identity");
    check(at(j, "capabilities_url").as_string() == "/v1/capabilities", "capabilities reference on");
    ApiConfig hidden = config; hidden.capabilities = false;
    ApiService hidden_api(hidden, {}, queue, 8192, false);
    const auto hidden_health = parse_json(get(hidden_api).body);
    check(at(hidden_health, "capabilities_id").is_null() && at(hidden_health, "capabilities_url").is_null(),
        "disabled capabilities references null");
    check(get(hidden_api, "/v1/capabilities", "secret-token").status == 404, "disabled capabilities route remains absent");
    // Main's legacy wire contract, for both aliases, all service states and both capability settings.
    for (const bool enabled : {false, true}) {
        ApiConfig baseline = legacy; baseline.capabilities = enabled;
        ApiConfig rollback = config; rollback.capabilities = enabled;
        ApiService baseline_api(baseline, {}, queue, 8192, false);
        ApiService rollback_api(rollback, {}, queue, 8192, false);
        env("STRIX_SERVE_HEALTH_RECORD", "0");
        const auto reads_before = probe.reads;
        for (const char* state : {"ready", "failed", "draining"}) {
            baseline_api.set_state(state); rollback_api.set_state(state);
            const std::string status = std::string_view(state) == "ready" ? "ok" : state;
            const auto expected = std::string("{\"status\":\"") + status + R"(","model":"public-model-\"seed")" +
                (enabled ? std::string(",\"capabilities_id\":\"") + cap_id + "\"" : "") + "}";
            for (const char* path : {"/health", "/v1/health"}) {
                const auto original = get(baseline_api, path), restored = get(rollback_api, path);
                check(original.body == expected && restored.body == expected &&
                    original.status == (status == "ok" ? 200 : 503) && restored.status == original.status,
                    "main legacy bytes and status preserved with capabilities on/off");
            }
        }
        check(probe.reads == reads_before, "legacy never reads provider");
        env("STRIX_SERVE_HEALTH_RECORD", nullptr);
    }
    for (const char* seed : {"/secret/account", "private-host", "secret-token", "secret-host"})
        check(result.body.find(seed) == std::string::npos, "privacy seed absent");
    check(at(j, "model").as_string() == legacy.model_id, "configured public model preserved and escaped");
    for (const auto state : {BackendState::starting, BackendState::probe_pending, BackendState::unresponsive, BackendState::failed, BackendState::draining}) {
        probe.value.state = state; check(get(api).status == 503, "nonresponsive state 503");
    }
    probe.value = {BackendState::responsive, BackendProof::work_completion, 0.1, 0.1, 1, 1};
    check(get(api).status == 200 && at(parse_json(get(api).body), "ready").as_bool(), "scripted fresh device evidence");
    probe.value.proof = BackendProof::control_only;
    check(get(api).status == 503, "owner ACK insufficient");
    probe.value.proof = BackendProof::idle_scratch_event;
    probe.value.probe_age_seconds = 2.0;
    check(get(api).status == 503, "stale evidence");
    probe.value.probe_age_seconds = 0.1; probe.value.required_ranks = 2;
    check(get(api).status == 503, "peer missing");
    probe.value.responsive_ranks = 2;
    check(get(api).status == 200, "both ranks fresh");
    probe.value.last_completion_age_seconds = std::numeric_limits<double>::quiet_NaN();
    check(get(api).status == 503 && at(at(parse_json(get(api).body), "backend"), "last_completion_age_seconds").is_null(), "invalid age null");
    probe.value = {};
    c->interim_unchecked_http_success = true;
    check(get(api).status == 200 && !at(parse_json(get(api).body), "ready").as_bool(), "interim compatibility without liveness claim");
    c->backend_probe_enabled = false;
    check(get(api).status == 503, "explicit disabled probe");
    for (const char* state : {"failed", "draining"}) {
        api.set_state(state); check(get(api).status == 503, "service state wins");
    }
    api.set_state("ready");
    env("STRIX_SERVE_HEALTH_RECORD", "0");
    const auto reads = probe.reads;
    check(get(api).body == old_body && get(api, "/v1/health").body == old_body && probe.reads == reads, "off switch legacy bytes and no provider");
    api.set_state("failed"); check(get(api).status == 503, "legacy failure 503");
    env("STRIX_SERVE_HEALTH_RECORD", nullptr);
    ApiConfig throwing = config;
    throwing.health_provider = []() -> BackendHealth { throw std::runtime_error("secret-token /secret/account private-host"); };
    ApiService throwing_api(throwing, {}, queue, 8192, false);
    c->backend_probe_enabled = true;
    const auto provider_error = get(throwing_api);
    check(provider_error.status == 503 && provider_error.body.find("secret-token") == std::string::npos, "provider errors are bounded");
    // All fixed routes are accepted by the dispatcher; optional drain is installed explicitly.
    config.on_drain = [] {};
    config.api_key.clear();
    const std::vector<std::string> fixed_routes{"GET /health", "GET /v1/health", "GET /metrics", "GET /api/status",
        "GET /v1/models", "GET /v1/capabilities", "POST /v1/completions", "POST /v1/chat/completions",
        "POST /v1/responses", "POST /v1/messages", "POST /tokenize", "POST /detokenize", "POST /admin/drain"};
    ApiService routed(config, {}, queue, 8192, false);
    for (const auto& route : fixed_routes) {
        HttpRequest request; request.method = route.substr(0, route.find(' ')); request.path = route.substr(route.find(' ') + 1);
        // Valid JSON reaches path dispatch; invalid route-specific data avoids tokenizer/device work.
        request.body = "{}";
        Writer out; routed.handle(request, out); check(out.status != 404, "advertised route registered");
    }
    HttpRequest absent; absent.method = "POST"; absent.path = "/not-installed"; absent.body = "{}";
    Writer missing_route; routed.handle(absent, missing_route);
    check(missing_route.status == 404, "unregistered POST valid JSON is 404");
    // Compare the advertised optional catalog with independent fixtures through the actual registry dispatch.
    strix::RtxStateImportRegistry registry([](std::uint64_t bytes) -> strix::PrefixCacheBuffer {
        std::shared_ptr<std::byte> data(new std::byte[static_cast<std::size_t>(bytes ? bytes : 1)],
            std::default_delete<std::byte[]>());
        return {std::move(data), bytes};
    }, 2);
    config.engine_routes = [&](const HttpRequest& request, HttpResponseWriter& out) {
        return strix::decode_cli::handle_rtx_state_import_route(registry, request, out);
    };
    ApiService imported(config, {}, queue, 8192, false);
    const std::vector<std::pair<std::string, int>> import_fixtures{
        {"GET /admin/rtx-import/layout", 503}, {"GET /admin/rtx-import/status", 200},
        {"POST /admin/rtx-import/begin", 400}, {"POST /admin/rtx-import/chunk", 400},
        {"POST /admin/rtx-import/chunks", 400}, {"POST /admin/rtx-import/commit", 400},
        {"POST /admin/rtx-import/abort", 400}};
    for (const auto& [route, status] : import_fixtures) {
        HttpRequest request; request.method = route.substr(0, route.find(' '));
        request.path = route.substr(route.find(' ') + 1); request.body = "{}";
        Writer out; imported.handle(request, out);
        check(out.status == status, "advertised import route reaches actual registry handler");
    }
    absent.path = "/admin/rtx-import/not-installed";
    Writer missing_import; imported.handle(absent, missing_import);
    check(missing_import.status == 404, "unregistered import POST is 404");
    check(!observe_armed() || observe_period() != period, "route scan reaches metrics arming");
    observe_disarm();
    // Exercise the same host-only startup builder used by both Qwen serve bridges.
    strix::decode_cli::Options options;
    options.http_bind = "private-host"; options.http_api_key_file = "/secret/account";
    options.prefix_cache_dir = "/secret/account";
    options.prefix_cache_host_bytes = 1024; options.prefix_cache_disk_bytes = 2048;
    options.context_tokens = 4096; options.known_prefix_prefill_rows = 256;
    options.mtp.draft_profile = "distill-g";
    strix::StartupPreflight startup;
    startup.observation.end_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    startup.assessment.advisories.push_back(strix::PreflightWarning::preexisting_gtt_use);
    env("STRIX_SERVE_BACKEND_PROBE", nullptr);
    const auto single = strix::decode_cli::http_health_record(options, 1, startup, {});
    const auto batch = strix::decode_cli::http_health_record(options, 4, startup, {});
    check(*single->slots == 1 && *batch->slots == 4, "Qwen actual slots");
    const auto wired = render_health(*single, {}, "ready", "public-model", {});
    check(wired.status == 200 && !at(wired.body, "ready").as_bool(), "Qwen interim startup default");
    check(at(at(wired.body, "preflight"), "state").as_string() == "advisory" &&
        at(at(wired.body, "preflight"), "warnings").as_array().empty() &&
        at(at(wired.body, "preflight"), "advisories").as_array().size() == 1 &&
        at(at(wired.body, "preflight"), "sample_age_seconds").is_number(), "cached startup evidence and age");
    check(wired.body.dump().find("/secret/account") == std::string::npos &&
        wired.body.dump().find("private-host") == std::string::npos, "Qwen raw options stay private");
    check(at(at(at(wired.body, "storage"), "prefix"), "host_limit_bytes").as_int() == 1024 &&
        at(at(at(wired.body, "storage"), "prefix"), "disk_limit_bytes").as_int() == 2048, "Qwen byte budgets");
    env("STRIX_SERVE_BACKEND_PROBE", "0");
    check(!strix::decode_cli::http_health_record(options, 1, {}, {})->backend_probe_enabled, "Qwen probe off switch");
    env("STRIX_SERVE_BACKEND_PROBE", nullptr);
    std::random_device random;
    const auto root = std::filesystem::temp_directory_path() / ("strix-health-" + std::to_string(random()));
    check(std::filesystem::create_directory(root), "create private synthetic asset fixture");
    std::ofstream(root / "target", std::ios::binary) << "abc";
    std::filesystem::create_directory(root / "vocab");
    std::ofstream(root / "vocab/a.bin", std::ios::binary) << "a";
    std::ofstream(root / "vocab/b.bin", std::ios::binary) << "b";
    const auto spec = R"({"format":"strix-assets","revision":1,"engine":"qwen","model_key":"secret-host","profile_pair":"test","required":{"target":["model"],"vocabulary":["a","b"]},"files":[{"role":"target","key":"model","member":"target","source_revision":"test"},{"role":"vocabulary","key":"a","member":"vocab/a.bin","source_revision":"test"},{"role":"vocabulary","key":"b","member":"vocab/b.bin","source_revision":"test"}],"provenance":[],"licenses":[]})";
    const auto manifest = strix::create_asset_manifest(spec, root);
    std::ofstream(root / "assets.json", std::ios::binary) << manifest;
    options.http_api_key_file.clear();
    options.model_path = (root / "target").string();
    options.release_manifest = (root / "assets.json").string();
    options.mtp.draft_vocab = (root / "vocab/a.bin").string() + "," + (root / "vocab/b.bin").string();
    const auto verified = strix::decode_cli::verify_http_release_assets(options);
    options.mtp.draft_vocab = "auto:" + (root / "vocab").string();
    check(!strix::decode_cli::verify_http_release_assets(options).is_null(), "automatic vocabulary closure bound at startup");
    std::ofstream(root / "vocab/extra.bin", std::ios::binary) << "c";
    bool extra_rejected = false;
    try { (void)strix::decode_cli::verify_http_release_assets(options); } catch (const strix::decode_cli::ReleaseAssetFailure&) { extra_rejected = true; }
    check(extra_rejected, "automatic vocabulary cannot read unlisted files");
    std::filesystem::remove(root / "vocab/extra.bin");
    const auto verified_caps = strix::decode_cli::http_health_record(options, 1, {}, verified);
    const auto bound = render_health(*verified_caps, {}, "ready", "public-model", {}).body;
    check(at(at(at(bound, "identity"), "assets"), "coverage").as_string() == "complete", "verified asset coverage");
    check(at(at(at(bound, "identity"), "assets"), "file_count").as_int() == 3, "verified file count");
    check(bound.dump().find(root.string()) == std::string::npos && bound.dump().find("secret-host") == std::string::npos, "verified private names excluded");
    std::ofstream(root / "other", std::ios::binary) << "abc";
    options.model_path = (root / "other").string();
    bool rejected = false;
    try { (void)strix::decode_cli::verify_http_release_assets(options); } catch (const strix::decode_cli::ReleaseAssetFailure&) { rejected = true; }
    check(rejected, "manifest cannot bind unrelated loader path");
    options.model_path = (root / "target").string();
    // The API-key file is an operator secret: an off-package key never fails verification.
    options.http_api_key_file = (root / "other").string();
    check(!strix::decode_cli::verify_http_release_assets(options).is_null(), "operator API key is outside the release inventory");
    options.http_api_key_file.clear();
    options.http_serve = true;
    strix::decode_cli::validate_release_manifest(options);
    const auto release_options = options;
    const auto reject_mode = [&](const strix::decode_cli::Options& candidate) {
        bool invalid = false;
        try { strix::decode_cli::validate_release_manifest(candidate); }
        catch (const std::invalid_argument& error) {
            invalid = std::string_view(error.what()).starts_with("--release-manifest requires --http-serve without cohort");
        }
        check(invalid, "unsupported release mode rejected before startup");
    };
    options.http_serve = false; reject_mode(options); options = release_options;
    options.mtp.cohort.push_back({{}, "secret-token", {}}); reject_mode(options); options = release_options;
    options.mtp.sample_repeats = 2; reject_mode(options); options = release_options;
    options.supplied_options.insert("--mtp-sample-repeats"); reject_mode(options); options = release_options;
    options.serving_prefix_files.push_back("secret-token"); reject_mode(options); options = release_options;
    options.prefix_cache_then_prompts.push_back("secret-token"); reject_mode(options); options = release_options;
    options.offline_compile_provenance_path = "secret-token"; reject_mode(options); options = release_options;
    for (const auto* flag : {"--mtp-cohort-request", "--mtp-cohort-mode", "--mtp-sample-repeats",
         "--serve-prefix-u32le", "--prefix-cache-then-prompt", "--offline-compile-provenance"}) {
        for (const bool manifest_first : {false, true}) {
            std::vector<std::string> arguments{"decode", "missing-model", "--http-serve"};
            if (manifest_first) { arguments.emplace_back("--release-manifest"); arguments.emplace_back("missing-manifest"); }
            arguments.emplace_back(flag); arguments.emplace_back("secret-token\nprivate-host");
            if (!manifest_first) { arguments.emplace_back("--release-manifest"); arguments.emplace_back("missing-manifest"); }
            std::vector<char*> pointers;
            for (auto& argument : arguments) pointers.push_back(argument.data());
            bool invalid = false;
            try { (void)strix::decode_cli::parse_options(static_cast<int>(pointers.size()), pointers.data(), true); }
            catch (const std::invalid_argument& error) {
                invalid = std::string_view(error.what()) == strix::decode_cli::release_manifest_mode_error;
            }
            check(invalid, "parser rejects unsupported release mode before reading extra inputs in either order");
        }
    }
    options.http_batch = 4; strix::decode_cli::validate_release_manifest(options); options = release_options;
#ifndef _WIN32
    const auto startup_failure = [&](const char* mode, const std::string& model, const std::string& manifest_path,
                                     const std::string& vocab, const std::string& expected) {
        int descriptors[2];
        if (pipe(descriptors) != 0) { check(false, "startup stderr pipe"); return; }
        const auto pid = fork();
        if (pid == 0) {
            close(descriptors[0]);
            if (dup2(descriptors[1], STDERR_FILENO) < 0) _exit(126);
            close(descriptors[1]);
            execl(argv[0], argv[0], mode, model.c_str(), manifest_path.c_str(), vocab.c_str(), nullptr);
            _exit(127);
        }
        close(descriptors[1]);
        std::string output;
        char buffer[256];
        ssize_t count;
        while ((count = read(descriptors[0], buffer, sizeof(buffer))) > 0) output.append(buffer, static_cast<std::size_t>(count));
        close(descriptors[0]);
        int status = 0;
        check(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 1,
            "startup failure exits nonzero");
        check(output == std::string("strix-hip-decode-token: ") + expected + "\n" && output.size() < 256,
            "outer startup message fixed and bounded");
        for (const auto* seed : {"secret-token", "private-host", "/secret/account", "model/device admission"})
            check(output.find(seed) == std::string::npos, "startup seed and device admission absent");
    };
    startup_failure("--startup-check", (root / "other").string(), options.release_manifest, "",
        "release manifest loader binding failure");
    const auto inventory = root / "inventory";
    std::filesystem::create_directory(inventory);
    for (unsigned i = 0; i < 1025; ++i) std::ofstream(inventory / (std::to_string(i) + ".txt"));
    startup_failure("--startup-check", options.model_path, options.release_manifest, "auto:" + inventory.string(),
        "release manifest inventory limit");
    std::filesystem::remove_all(inventory);
    const auto missing = (root / "secret-token\nprivate-host").string();
    startup_failure("--startup-check", missing, options.release_manifest, "", "release manifest I/O failure");
    startup_failure("--startup-check", options.model_path, missing, "", "release manifest I/O failure");
    startup_failure("--startup-check", options.model_path, options.release_manifest, "auto:" + missing,
        "release manifest I/O failure");
    startup_failure("--startup-denied", "", "", "", "release manifest I/O failure");
    startup_failure("--startup-unexpected", "", "", "", "release manifest I/O failure");
    std::ofstream(root / "invalid.json", std::ios::binary) << "secret-token\nprivate-host";
    startup_failure("--startup-check", options.model_path, (root / "invalid.json").string(), "",
        "release manifest invalid");
#endif
    std::ofstream(root / "target", std::ios::binary) << "abd";
    rejected = false;
    try { (void)strix::decode_cli::verify_http_release_assets(options); } catch (const strix::decode_cli::ReleaseAssetFailure&) { rejected = true; }
    check(rejected, "manifest verification failure stops startup");
#ifndef _WIN32
    const std::string key = "model";
    const auto key_digest = strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(key.data(), key.size()))));
    startup_failure("--startup-check", options.model_path, options.release_manifest, "",
        "release asset mismatch role=target index=0 key_digest=" + key_digest.substr(0, 16));
#endif
    options.print_effective_config = true;
    check(strix::decode_cli::verify_http_release_assets(options).is_null(), "inspection skips verification");
    std::filesystem::remove_all(root);
    std::cout << "health failures: " << failures << '\n';
    return failures ? 1 : 0;
}

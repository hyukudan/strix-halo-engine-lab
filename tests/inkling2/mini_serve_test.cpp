#include "inkling2/inkling2_mini_serve.hpp"
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/http_client.hpp"
#include "strix/serve/replica_router.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace sv = strix::serve;
namespace mini = strix::inkling2;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class FixtureDecoder {
public:
    explicit FixtureDecoder(bool final_logits = true) : final_logits_(final_logits) {}
    void allocate_slots(unsigned slots) { frontiers_.resize(slots); }
    void prefill_slot(std::uint32_t lane, std::span<const std::uint32_t> tokens, bool first) {
        if (first) {
            if (frontiers_.at(lane) != 0U) ++reused;
            frontiers_.at(lane) = 0;
            ++started;
        }
        frontiers_.at(lane) += tokens.size();
    }
    void decode_slots(std::span<const std::uint32_t> slots, std::span<const std::uint32_t> tokens,
                      std::span<std::uint32_t> top) {
        require(slots.size() == tokens.size() && top.size() == tokens.size(), "group geometry");
        peak = std::max(peak, slots.size());
        last_.clear();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            require(frontiers_.at(slots[i]) != 0, "invalid live row");
            last_.emplace_back(tokens[i], frontiers_[slots[i]]);
            ++frontiers_[slots[i]];
            top[i] = (tokens[i] + 1U) % 2U;
        }
    }

    // Final scores of the last decode_slots() call. A row depends only on its fed token and its own frontier.
    static constexpr std::uint32_t kWidth = 8U;
    bool final_logits_supported() const { return final_logits_; }
    std::uint32_t logit_width() const { return kWidth; }
    void read_slot_logits(std::uint32_t row, std::span<float> output) {
        require(row < last_.size() && output.size() == kWidth, "final logits row geometry");
        ++reads;
        const auto [token, frontier] = last_[row];
        for (std::uint32_t v = 0; v < kWidth; ++v)
            output[v] = static_cast<float>((token * 5U + static_cast<std::uint32_t>(frontier) * 3U + v * 7U) % 11U) / 4.0F;
    }

    std::size_t peak = 0;
    unsigned started = 0, reused = 0, reads = 0;
private:
    bool final_logits_;
    std::vector<std::size_t> frontiers_;
    std::vector<std::pair<std::uint32_t, std::size_t>> last_;
};

using FixtureCohort = sv::FixedKvCohort<FixtureDecoder>;

// A resident decoder for the order of the served commitments: device bytes come from one free pool.
struct FakeScreenDecoder {
    std::uint64_t free;
    std::uint64_t plane;
    std::uint64_t built = 0;
    std::vector<std::string> log{};
    struct Memory { std::uint64_t free, total; };
    [[nodiscard]] Memory memory() const { return {free, 1000}; }
    [[nodiscard]] std::uint64_t screen_device_bytes() const { return built; }
    std::string_view build_screen() {
        log.emplace_back("plane");
        if (plane > free) return "plane-unavailable";
        free -= (built = plane);
        return {};
    }
    void release_screen(const char* why) {
        free += built;
        built = 0;
        log.emplace_back(std::string("release: ") + why);
    }
    void allocate_slot(std::uint64_t bytes) {
        if (bytes > free) throw std::runtime_error("KV slot allocation failed");
        free -= bytes;
        log.emplace_back("slot");
    }
};

// CUDA screened-head variant: the KV slots the full-head budget admits are committed first, the plane after them.
void screen_after_slots() {
    const auto serve = [](FakeScreenDecoder& decoder, bool variant, std::uint32_t& lanes) {
        decoder.allocate_slot(100);  // the decoder holds slot 0 when it is constructed
        const auto memory = [&decoder] { return decoder.memory(); };
        return strix::inkling2::mini_commit_slots_then_screen(decoder, memory, 0, 0, 100, 5, variant, lanes,
            [&decoder](std::uint32_t admitted) {
                for (std::uint32_t slot = 1; slot < admitted; ++slot) decoder.allocate_slot(100);
                return admitted;
            });
    };
    // The reviewed case: the plane (30) and the first slot (100) each fit in 120 free bytes, both do not. Plane first
    // (the old order) fails the first slot; slots first starts the incumbent full-head server.
    FakeScreenDecoder old_order{120, 30};
    old_order.build_screen();
    bool failed = false;
    try {
        old_order.allocate_slot(100);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed, "the reviewed memory state does not fail the plane-first order");
    std::uint32_t lanes = 0;
    FakeScreenDecoder tight{120, 30};
    require(serve(tight, true, lanes) == std::pair{std::uint32_t{1}, std::string_view("plane-unavailable")} &&
                lanes == 1U && tight.built == 0U &&
                tight.log == std::vector<std::string>{"slot", "plane", "release: screened head variant: plane not built"},
            "plane fits, first slot does not: the incumbent server must still start on the full head");
    FakeScreenDecoder roomy{400, 10};  // 300 free after slot 0 -> 3 slots, then a 10-byte plane costs none of them
    require(serve(roomy, true, lanes) == std::pair{std::uint32_t{3}, std::string_view{}} && roomy.built == 10U &&
                roomy.log == std::vector<std::string>{"slot", "slot", "slot", "plane"},
            "a plane that costs no committed slot stays");
    FakeScreenDecoder headroom{400, 80};  // the plane would eat the runtime headroom of the committed slots
    require(serve(headroom, true, lanes) == std::pair{std::uint32_t{3}, std::string_view("slot-budget-or-headroom")} &&
                headroom.built == 0U && headroom.free == 100U,
            "a plane inside the runtime headroom is released");
    FakeScreenDecoder off{400, 10};  // variant not admitted: the incumbent, no plane work at all
    require(serve(off, false, lanes) == std::pair{std::uint32_t{3}, std::string_view{}} && off.built == 0U &&
                off.log == std::vector<std::string>{"slot", "slot", "slot"},
            "the incumbent full-head server built a plane");
    FakeScreenDecoder hip_pressure{400, 50, 50};  // the HIP plane is already resident before admission
    require(serve(hip_pressure, false, lanes) == std::pair{std::uint32_t{4}, std::string_view{}} && hip_pressure.built == 0U &&
                hip_pressure.log.at(1) == "release: the screened plane would cost a KV slot",
            "the incumbent HIP pressure path must release the plane before committing the extra slot");
    require(!strix::inkling2::mini_keep_screen(100, 100, 1000, 0, 0, 100, 5, 3), "no plane built is never kept");
}

void admission() {
    require(sv::fixed_kv_slots(99, 100, 3) == 0, "KV budget too small");
    require(sv::fixed_kv_slots(400, 100, 3) == 3, "row budget cap");
    require(sv::fixed_kv_slots(400, 100, 3, 1) == 1, "explicit slot cap");
    require(sv::resident_kv_slots(300, 1000, 0, 0, 100, 5) == 3, "physical free memory admission");
    require(sv::resident_kv_slots(300, 1000, 800, 0, 100, 5) == 1, "declared device memory admission");
    require(sv::resident_kv_slots(300, 1000, 699, 0, 100, 5) == 0, "resident allocations exceed declaration");
    require(sv::resident_kv_slots(300, 1000, 0, 99, 100, 5) == 0, "explicit budget includes initial slot");
    require(sv::resident_kv_slots(0, 100, 0, 0, 100, 5) == 1, "initial slot already resident");
    // The screened head plane stays only when it costs no KV slot (with it: 300 free -> 3 slots).
    {
        using strix::inkling2::mini_serve_slots;
        const auto costly = mini_serve_slots(300, 1000, 0, 0, 100, 5, 50);  // without it: 350 free -> 4 slots
        require(costly.lanes == 4U && !costly.keep_screen, "a plane that costs a slot is released");
        const auto cheap = mini_serve_slots(300, 1000, 0, 0, 100, 5, 10);  // without it: 310 free -> 3 slots
        require(cheap.lanes == 3U && cheap.keep_screen, "a plane that costs no slot stays");
        const auto none = mini_serve_slots(300, 1000, 0, 0, 100, 5, 0);
        require(none.lanes == 3U && !none.keep_screen, "no plane: the full head budget");
        const auto declared = mini_serve_slots(300, 1000, 800, 0, 100, 5, 100);  // declared cap: 1 -> 2 slots
        require(declared.lanes == 2U && !declared.keep_screen, "a declared device budget counts the plane too");
        const auto full = mini_serve_slots(1000, 1000, 0, 0, 100, 5, 50);
        require(full.lanes == sv::resident_kv_slots(1000, 1000, 0, 0, 100, 5), "free never exceeds total");
    }
    screen_after_slots();
    const auto slots = sv::fixed_kv_slots(250, 100, 3);
    require(slots == 2, "KV budget holds two lanes");
    auto device = std::make_unique<FixtureDecoder>();
    const auto* trace = device.get();
    FixtureCohort backend(std::move(device), 32U, 3U, slots);
    sv::GenerationQueue queue;
    sv::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    sv::CohortSession cohort(queue, backend, config);
    auto& owner = cohort.runtime();
    std::vector<std::shared_ptr<sv::GenerationTicket>> tickets;
    for (const auto id : {0U, 1U, 0U}) {
        sv::GenerationJob job;
        job.prompt = {1U, id};
        job.max_tokens = 3;
        tickets.push_back(queue.submit(std::move(job)));
    }
    require(owner.step(), "first cohort turn");
    require(queue.snapshot().running == slots && queue.snapshot().queued == 1, "excess request stays queued");
    while (queue.snapshot().completed != tickets.size()) require(owner.step(), "cohort turn");
    require(trace->peak == slots && trace->started == 3 && trace->reused == 1, "batching and slot reuse");
    for (std::size_t i = 0; i < tickets.size(); ++i) {
        bool done = false;
        const auto id = static_cast<std::uint32_t>((i + 1U) % 2U);
        require(tickets[i]->wait(std::chrono::milliseconds{0}, done) == std::vector{id, 1U - id, id} &&
            done && tickets[i]->error().empty(), "greedy ids differ from serial");
    }
    sv::GenerationJob job;
    job.prompt = {1U, 0U};
    job.max_tokens = 4;
    const auto cancelled = queue.submit(std::move(job));
    require(owner.step() && owner.step(), "decode before cancellation");
    cancelled->cancel();
    require(owner.step() && cancelled->stats().finish == "cancelled" && owner.live() == 0U,
            "cancelled stream did not release its slot");

    sv::CohortDecoder boundary(backend, config);
    boundary.submit({1U, {1U, 0U}, 30U, {}});
    boundary.submit({2U, {1U, 0U}, 31U, {}});
    boundary.run();
    const auto results = boundary.take_finished();
    require(results.size() == 2U && results[0].id == 2U && !results[0].error.empty() &&
            results[1].id == 1U && results[1].ids.size() == 30U, "cohort context boundary");
}

// A sampled request decides from the final row of its own lane and its absolute position, so a fixed seed
// replays the same ids alone or next to other lanes. Greedy lanes never read the scores.
void sampling() {
    sv::CohortConfig config;
    config.context_draft = false;
    config.max_draft = 0;
    const auto make = [](std::uint64_t id, std::uint32_t first, float temperature, std::uint64_t seed) {
        sv::CohortRequest request;
        request.id = id;
        request.prompt = {1U, first};
        request.max_tokens = 6U;
        request.sampling.temperature = temperature;
        request.sampling.seed = seed;
        return request;
    };
    const auto run = [&](std::vector<sv::CohortRequest> requests, unsigned& reads) {
        auto device = std::make_unique<FixtureDecoder>();
        const auto* trace = device.get();
        FixtureCohort backend(std::move(device), 32U, 3U, 3U);
        require(backend.supports_final_logits(), "Inkling cohort offers final logits");
        sv::CohortDecoder cohort(backend, config);
        for (auto& request : requests) cohort.submit(std::move(request));
        cohort.run();
        std::map<std::uint64_t, sv::CohortResult> out;
        for (auto& result : cohort.take_finished()) {
            require(result.error.empty(), "sampled request refused");
            out.emplace(result.id, std::move(result));
        }
        reads = trace->reads;
        return out;
    };
    unsigned greedy_reads = 0, alone_reads = 0, mixed_reads = 0, solo_greedy_reads = 0;
    const auto solo_greedy = run({make(1U, 0U, 0.0F, 0U)}, solo_greedy_reads);
    const auto greedy_pair = run({make(1U, 0U, 0.0F, 0U), make(2U, 1U, 0.0F, 0U)}, greedy_reads);
    const auto alone = run({make(3U, 0U, 1.0F, 11U)}, alone_reads);
    const auto mixed = run({make(1U, 0U, 0.0F, 0U), make(3U, 0U, 1.0F, 11U), make(4U, 1U, 1.0F, 12U)}, mixed_reads);
    require(solo_greedy_reads == 0U && greedy_reads == 0U, "greedy lanes read no final logits");
    require(alone_reads > 0U && mixed_reads > 0U, "sampled lanes read final logits");
    const std::vector<std::uint32_t> expected{1U, 0U, 1U, 0U, 1U, 0U};
    require(solo_greedy.at(1U).ids == expected && greedy_pair.at(1U).ids == expected && mixed.at(1U).ids == expected,
            "greedy ids changed beside sampled lanes");
    require(solo_greedy.at(1U).sampling == "greedy" && alone.at(3U).sampling != "greedy", "sampling receipt");
    require(alone.at(3U).ids.size() == 6U && alone.at(3U).ids == mixed.at(3U).ids,
            "same seed differs alone and beside other lanes");
}

struct Writer final : sv::HttpResponseWriter {
    using Headers = std::vector<std::pair<std::string, std::string>>;
    int status = 0;
    bool ended = false;
    std::string body;
    void send(int s, std::string_view, std::string_view data, const Headers&) override {
        status = s;
        body = data;
    }
    bool begin_stream(int s, std::string_view, const Headers&) override { status = s; return true; }
    bool write(std::string_view data) override { body += data; return true; }
    void end_stream() override { ended = true; }
    bool client_gone() const override { return false; }
};

constexpr std::string_view sample_request = R"({"messages":[{"role":"user","content":"a"}],"max_tokens":3,"temperature":0.8,"top_k":1,"seed":544,"return_token_ids":true})";

void require_sample_reply(int status, std::string_view body, bool sample_capable) {
    const auto reply = sv::parse_json(body);
    if (!sample_capable) {
        const auto* error = reply.find("error");
        const auto* param = error ? error->find("param") : nullptr;
        require(status == 400 && param && param->as_string() == "temperature",
                "backend without final logits accepted positive temperature");
        return;
    }
    const auto* receipt = reply.find("strix");
    require(status == 200 && receipt, "HTTP runtime refused supported sampling");
    const auto* mode = receipt->find("sampling");
    const auto* ids = receipt->find("token_ids");
    const auto* execution = receipt->find("sampling_receipt");
    const auto* seed = execution ? execution->find("seed") : nullptr;
    require(mode && mode->as_string() == "sampling-v1" && ids && ids->dump() == "[1,3,1]" &&
            seed && seed->as_int() == 544,
            "sampled HTTP reply did not use the fixture's final logits and specified seed");
}

// Run the production HTTP listener and owner on loopback. The device decoder
// and tokenizer are fixtures.
void socket_startup(const sv::ServiceFactory& factory, const char* backend, bool sample_capable) {
    sv::ServiceSelection selection;
    selection.model.name = "inkling2-mini";
    selection.arguments = {"--checkpoint", "MODEL.safetensors", "--backend", backend, "--profile", "fast"};
    factory.validate(selection);
    struct Listener final : sv::SessionService {
        explicit Listener(std::unique_ptr<sv::SessionService> delegate) : service(std::move(delegate)) {}
        sv::SessionRuntime& runtime() override { return service->runtime(); }
        void handle(const sv::HttpRequest& request, sv::HttpResponseWriter& response) override {
            service->handle(request, response);
        }
        void listening(const sv::HttpServerConfig& config, std::uint16_t port) override {
            service->listening(config, port);
            ready.set_value(port);
        }
        std::unique_ptr<sv::SessionService> service;
        std::promise<std::uint16_t> ready;
    };
    Listener live(factory.create(selection));
    sv::HttpServerConfig config;
    config.port = 0;
    live.service->configure_listener(config);
    auto ready = live.ready.get_future();
    std::exception_ptr failure;
    std::thread listener([&] {
        try { sv::serve_session(config, live, false); }
        catch (...) { failure = std::current_exception(); }
    });
    const auto stop = [&] {
        live.runtime().shutdown();
        listener.join();
    };
    try {
        require(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "Inkling listener readiness timed out");
        sv::HttpClientRequest request;
        request.port = ready.get();
        request.connect_timeout_ms = request.idle_timeout_ms = 2000;
        std::string body;
        for (const auto* target : {"/health", "/api/status"}) {
            request.target = target;
            const auto result = sv::http_fetch(request, body);
            if (!result.ok) throw std::runtime_error(request.target + ": " + result.error);
            require(result.head.status == 200, "Inkling listener did not return ready status");
        }
        const auto replica = sv::replica_state_from_status(body);
        require(replica && replica->healthy && replica->capacity.slots_total == 2, "ready replica advertised wrong capacity");
        request.method = "POST";
        request.target = "/v1/chat/completions";
        request.headers = {{"Content-Type", "application/json"}};
        request.body = sample_request;
        const auto sample = sv::http_fetch(request, body);
        if (!sample.ok) throw std::runtime_error(request.target + ": " + sample.error);
        require_sample_reply(sample.head.status, body, sample_capable);
    } catch (...) {
        stop();
        if (failure) std::rethrow_exception(failure);
        throw;
    }
    stop();
    if (failure) std::rethrow_exception(failure);

    Listener invalid(factory.create(selection));
    auto invalid_ready = invalid.ready.get_future();
    config.bind_address = "invalid-inkling-test-address";
    try {
        sv::serve_session(config, invalid, false);
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()).find("HTTP bind address") != std::string_view::npos &&
                invalid_ready.wait_for(std::chrono::seconds(0)) == std::future_status::timeout &&
                invalid.runtime().stopping(), "Inkling bind failure did not propagate or stop the owner");
        return;
    }
    throw std::runtime_error("invalid Inkling bind unexpectedly became ready");
}

using W4a4 = strix::moe_core::grouped_w4a4::Activations;

// The served W4A4 lever value (process environment, as strix-serve reads it); nullptr clears it.
void set_w4a4_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_MOE_W4A4", value ? value : "") == 0, "cannot set STRIX_INK_MOE_W4A4");
#else
    require((value ? setenv("STRIX_INK_MOE_W4A4", value, 1) : unsetenv("STRIX_INK_MOE_W4A4")) == 0,
            "cannot set STRIX_INK_MOE_W4A4");
#endif
}

void wiring(const char* root, bool loopback, const char* backend, bool qualify_q8, bool sample_capable = false,
            W4a4 w4a4 = W4a4::bf16, bool screen_without_plane = false) {
    sv::ServiceFactories factories;
    std::ostringstream startup;
    bool created = false;
    const auto create = [&](const mini::MiniServeOptions& options) {
        require(options.checkpoint == "MODEL.safetensors", "checkpoint did not reach native adapter");
        // The served prefill runs the operand class the profile resolved, never a class-less on/off.
        require(options.moe_w4a4 == w4a4, "W4A4 operand class did not reach native adapter");
        require(options.tier == strix::profile::Name::fast && options.weight_options.qualify_q8 == qualify_q8,
                "FAST request did not reach native adapter");
        const auto kind = std::string_view(backend) == "hip" ? strix::weight::Backend::hip : strix::weight::Backend::cuda;
        const strix::weight::WeightLoadScope scope{kind, {}, {}, {}, {}, {}};
        const auto weights = mini::resolve_mini_resident_weight_load(options.tier, options.weight_options, scope);
        // No switch: both backends serve the incumbent grouped tile kernel (packed serving is opt-in).
        require(!options.packed_decode, "served decode expert form differs from the serving default (grouped)");
        created = true;
        FakeScreenDecoder screen{250, 80};  // two committed slots leave 50 free, insufficient for the plane
        screen.allocate_slot(100);
        std::uint32_t lanes = 0;
        auto [cohort, decline] = mini::mini_commit_slots_then_screen(
            screen, [&] { return screen.memory(); }, 0, 0, 100, 3,
            options.screened_head, lanes, [&](std::uint32_t admitted) {
                for (std::uint32_t slot = 1; slot < admitted; ++slot) screen.allocate_slot(100);
                return std::make_unique<FixtureCohort>(std::make_unique<FixtureDecoder>(sample_capable),
                                                        32U, 3U, admitted);
            });
        mini::MiniServeIdentity identity(options, decline);
        identity.write_startup(startup);
        return mini::MiniServeCohort{std::move(cohort), weights, std::move(identity)};
    };
    mini::MiniServeBackends backends;
    if (std::string_view(backend) == "hip") backends.hip = create;
    else backends.cuda = create;
    mini::install_mini_service(factories, std::move(backends),
        [](const std::string&) {
            return sv::BpeTokenizer::from_parts({"a", "b", "c", "d", "e", "f", "g", "h"},
                                                 {1, 1, 1, 1, 1, 1, 1, 1}, {}, "o200k");
        },
        [](const sv::Json& messages, const sv::Json&, const sv::Json& effort) {
            require(effort.as_string() == "none", "wrong chat effort");
            return "a" + messages.as_array().back().find("content")->as_string();
        });
    const char* args[]{"strix-serve", "--model", "inkling2-mini", "--registry-root", root,
                       "--checkpoint", "MODEL.safetensors", "--backend", backend,
                       "--profile", "fast", "--speculation", "off"};
    std::vector<std::string> strings(std::begin(args), std::end(args));
    if (qualify_q8) strings.emplace_back("--qualify-q8");
    const auto run = [&](sv::ServeShell shell, std::ostream& out, std::ostream& error) {
        std::vector<char*> argv;
        for (auto& arg : strings) argv.push_back(arg.data());
        return sv::run(static_cast<int>(argv.size()), argv.data(), factories, out, error, shell);
    };
    const auto inspect = [&] {
        std::ostringstream out, error;
        strings.emplace_back("--family-inspect");
        const auto rc = run({}, out, error);
        strings.pop_back();
        require(rc == 0 && !created, "Inkling profile inspection should run without device work");
        const auto expected = qualify_q8 ? "weight_storage gguf.q8_0@1 weight_status unqualified" :
                                           "weight_storage plain.bf16@1 weight_status pending";
        require(out.str().find(expected) != std::string::npos, "inspection weight plan differs from CLI plan");
    };
    inspect();

    for (const auto* field : {"--dspark", "--speculation", "--no-weight-q8"}) {
        if (std::string_view(field) == "--no-weight-q8" && !qualify_q8) continue;
        std::ostringstream out, error;
        strings.emplace_back(field);
        if (std::string_view(field) == "--speculation") strings.emplace_back("auto");
        const auto expected = std::string_view(field) == "--no-weight-q8" ? "--qualify-q8" : field;
        require(run({}, out, error) == 2 && !created && error.str().find(expected) != std::string::npos,
                "unsupported or conflicting option did not fail before native load");
        if (std::string_view(field) == "--speculation") strings.pop_back();
        strings.pop_back();
    }
    {
        // The two decode-form switches exclude each other before any native load.
        std::ostringstream out, error;
        strings.emplace_back("--packed-decode");
        strings.emplace_back("--grouped-decode");
        require(run({}, out, error) == 2 && !created && error.str().find("mutually exclusive") != std::string::npos,
                "--packed-decode with --grouped-decode did not fail before native load");
        strings.resize(strings.size() - 2U);
    }
    if (std::string_view(backend) == "cuda") {
        // CUDA packed serving has no device comparison against its reference (the lone decode) yet: held at launch.
        std::ostringstream out, error;
        strings.emplace_back("--packed-decode");
        require(run({}, out, error) == 2 && !created &&
                    error.str().find("identity variant inkling2-mini-moe-packed@1 is not declared for backend cuda") !=
                        std::string::npos,
                "CUDA --packed-decode serving was not refused by the variant admission before native load");
        strings.pop_back();
    }
    {
        // Identity variants through the real server parser: selected by name, admitted by backend and tier through the
        // profile resolver, printed on the inspection line; every spelling of a request resolves to the same line.
        const auto inspect_with = [&](std::initializer_list<const char*> extra, std::string& text) {
            std::ostringstream out, error;
            for (const auto* item : extra) strings.emplace_back(item);
            strings.emplace_back("--family-inspect");
            const auto rc = run({}, out, error);
            strings.resize(strings.size() - extra.size() - 1U);
            text = rc == 0 ? out.str() : error.str();
            return rc;
        };
        const auto has = [](const std::string& text, const char* part) { return text.find(part) != std::string::npos; };
        const bool hip = std::string_view(backend) == "hip";
        std::string text, alias;
        require(inspect_with({}, text) == 0 && has(text, "moe_decode grouped moe_decode_reference served-grouped "
                                                         "identity incumbent"),
                "default serving identity is not the incumbent");
        const auto packed_rc = inspect_with({"--packed-decode"}, alias);
        require(hip ? packed_rc == 0 && has(alias, "moe_decode packed moe_decode_reference lone-decode identity "
                                                   "inkling2-mini-moe-packed@1")
                    : packed_rc == 2 && has(alias, "is not declared for backend cuda"),
                "packed variant admission");
        require(inspect_with({"--identity-variant", "inkling2-mini-moe-packed@1"}, text) == packed_rc && text == alias,
                "--identity-variant and the variant option resolve differently");
        const auto screened_rc = inspect_with({"--screened-head"}, text);
        require(hip ? screened_rc == 2 && has(text, "inkling2-mini-cuda-screened-head@1 is not declared for backend hip")
                    : screened_rc == 0 && has(text, "moe_decode grouped moe_decode_reference served-grouped identity "
                                                    "inkling2-mini-cuda-screened-head@1 "
                                                    "screened_head_admission pending-device"),
                "screened-head variant admission");
        require(inspect_with({"--identity-variant", "inkling2-mini-nope"}, text) == 2 &&
                    has(text, "unknown identity variant 'inkling2-mini-nope'"),
                "an undeclared variant was not refused");
        require(inspect_with({"--grouped-decode", "--identity-variant", "inkling2-mini-moe-packed"}, text) == 2 &&
                    has(text, "mutually exclusive"),
                "grouped with a packed variant request was not refused");
        // EXACT admits the distinct packed identity. --qualify-q8 requires FAST, so use the native-weight calls.
        if (!qualify_q8) {
            auto tier = std::find(strings.begin(), strings.end(), std::string("fast"));
            *tier = "exact";
            require(inspect_with({"--packed-decode"}, text) == packed_rc &&
                        (!hip || has(text, "profile exact speculation off moe_decode packed")),
                    "EXACT admission of the exact packed identity variant");
            *tier = "fast";
        }
    }
    sv::ServeShell shell;
    shell.serve = [qualify_q8, sample_capable, screen_without_plane](
        const sv::HttpServerConfig&, sv::HttpHandler handle, sv::SessionRuntime& owner) {
        std::exception_ptr failure;
        std::thread control([&] { try { owner.run(); } catch (...) { failure = std::current_exception(); } });
        try {
            sv::HttpRequest req;
            req.method = "GET";
            req.target = req.path = "/api/status";
            Writer status;
            handle(req, status);
            const auto replica = sv::replica_state_from_status(status.body);
            require(status.status == 200 && replica && replica->healthy && replica->capacity.slots_total == 2,
                    "Inkling replica capacity missing");
            req.target = req.path = "/v1/capabilities";
            Writer capabilities;
            handle(req, capabilities);
            const auto effective = sv::parse_json(capabilities.body);
            const auto* identity = effective.find("identity");
            require(identity && identity->as_string() == "identity incumbent",
                    "server without a screened plane advertised a non-incumbent identity");
            if (screen_without_plane) {
                const auto* decline = effective.find("screened_head_decline");
                require(decline && decline->as_string() == "plane-unavailable",
                        "screened request without a plane did not report its decline");
            } else {
                require(!effective.find("screened_head_decline"), "incumbent launch reported a screened-head decline");
            }
            const auto* profile = effective.find("profile");
            const auto* weights = effective.find("weight_load");
            require(capabilities.status == 200 && profile && profile->as_string() == "fast" && weights &&
                    weights->find("storage")->as_string() == (qualify_q8 ? "gguf.q8_0" : "plain.bf16") &&
                    weights->find("status")->as_string() == (qualify_q8 ? "unqualified" : "pending"),
                    "Inkling capability response does not match the loaded plan");
            const auto* sampling = effective.find("sampling");
            const auto* temperature = sampling ? sampling->find("positive_temperature") : nullptr;
            require(temperature && temperature->as_string() == (sample_capable ? "sample" : "reject"),
                    "Inkling capability response advertised the wrong sampling mode");
            req.method = "POST";
            req.target = req.path = "/v1/chat/completions";
            req.body = sample_request;
            Writer sample;
            handle(req, sample);
            require_sample_reply(sample.status, sample.body, sample_capable);
            req.body = R"({"messages":[{"role":"user","content":"a"}],"max_tokens":2,"dspark":"draft.safetensors"})";
            Writer draft;
            handle(req, draft);
            require(draft.status == 400 && draft.body.find("dspark") != std::string::npos,
                    "greedy-only cohort accepted a DSpark request");
            req.body = R"({"messages":[{"role":"user","content":"a"}],"max_tokens":3})";
            Writer completion;
            handle(req, completion);
            require(completion.status == 200 && completion.body.find("\"content\":\"bab\"") != std::string::npos,
                    "chat reply missing fixture ids");
            req.body = R"({"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true})";
            Writer streaming;
            handle(req, streaming);
            require(streaming.status == 200 && streaming.ended && streaming.body.find("data: [DONE]") != std::string::npos,
                    "shared SSE did not complete");
        } catch (...) {
            owner.shutdown();
            control.join();
            throw;
        }
        owner.shutdown();
        control.join();
        if (failure) std::rethrow_exception(failure);
    };
    if (screen_without_plane) strings.emplace_back("--screened-head");
    std::ostringstream out, error;
    const auto rc = run(shell, out, error);
    if (rc != 0) throw std::runtime_error(error.str());
    require(created, "selected backend was not constructed");
    require(startup.str() == (screen_without_plane ? "identity incumbent screened_head_decline plane-unavailable"
                                                    : "identity incumbent"),
            "startup did not report the effective head identity and decline");
    if (loopback && !screen_without_plane && !qualify_q8 && (std::string_view(backend) == "hip" || sample_capable)) {
        const auto* factory = factories.find("inkling2");
        require(factory != nullptr, "Inkling service factory was not registered");
        socket_startup(*factory, backend, sample_capable);
    }
}
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--in-process"),
            "expected repository root [--in-process]");
    const bool loopback = argc == 2;
    admission();
    sampling();
    wiring(argv[1], loopback, "cuda", false, true, W4a4::bf16, true);
    wiring(argv[1], loopback, "cuda", false, true);
    for (const auto* backend : {"hip", "cuda"}) {
        wiring(argv[1], loopback, backend, false);
        wiring(argv[1], loopback, backend, true);
    }
    // One decode expert form resolver for the CLI and strix-serve: the incumbents (CLI packed, serving grouped),
    // an explicit switch wins on either entry point.
    using mini::MiniDecodeEntry;
    require(mini::mini_packed_decode(MiniDecodeEntry::cli, std::nullopt) &&
            !mini::mini_packed_decode(MiniDecodeEntry::serve, std::nullopt) &&
            mini::mini_packed_decode(MiniDecodeEntry::serve, true) &&
            !mini::mini_packed_decode(MiniDecodeEntry::cli, false),
            "decode expert form resolution");
    // Each served form names its reference.
    require(std::string_view(mini::mini_decode_reference(true)) == "lone-decode" &&
            std::string_view(mini::mini_decode_reference(false)) == "served-grouped",
            "served decode form reference");
    const auto& packed = mini::kMiniVariants.front();
    require(packed.reference == "each lane's lone packed one-row decode" &&
            packed.changes.find("verify") == std::string_view::npos &&
            packed.source == "HIP BF16 full-head decode fixture: C17 and C34 scores and picks match lone packed decode bitwise",
            "packed variant evidence must be scoped to the HIP BF16 full-head decode fixture");
    // Every lever value reaches the served decoder as its own class (mse used to run nearest@1 there).
    for (const auto& [value, expected] : {std::pair{"1", W4a4::nvfp4}, std::pair{"mse", W4a4::nvfp4_mse},
                                          std::pair{"mse-gate-up", W4a4::nvfp4_mse_gate_up}}) {
        set_w4a4_env(value);
        wiring(argv[1], loopback, "cuda", false, false, expected);
    }
    set_w4a4_env(nullptr);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

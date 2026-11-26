#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "inkling2/inkling2_mini_serve.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "inkling2_mini_cli.hpp"
#include "strix/model_data/inkling2_mini.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/http_client.hpp"
#include "strix/serve/replica_router.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace sv = strix::serve;
namespace mini = strix::inkling2;

// Served decode form: HIP packs only with the admitted lever; CUDA runs the command line's packed form at every width.
static_assert(strix::inkling2::mini_served_packed_decode(true, true) &&
              !strix::inkling2::mini_served_packed_decode(true, false) &&
              strix::inkling2::mini_served_packed_decode(false, false) &&
              !strix::inkling2::mini_served_packed_decode(false, false, false, true) &&
              strix::inkling2::mini_served_packed_decode(false, false, true, false));

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

void admission() {
    require(!mini::mini_server_calibrated_row_allowed(true, false),
            "a held-back HTTP prompt anchor cannot enter the calibrated decode row");
    require(mini::mini_server_calibrated_row_allowed(true, true),
            "the calibrated row remains available when served prefill emits its first token");
    require(!mini::mini_server_calibrated_row_allowed(false, true), "unset calibration stays off on the served route");
    require(sv::fixed_kv_slots(99, 100, 3) == 0, "KV budget too small");
    require(sv::fixed_kv_slots(400, 100, 3) == 3, "row budget cap");
    require(sv::fixed_kv_slots(400, 100, 3, 1) == 1, "explicit slot cap");
    require(sv::resident_kv_slots(300, 1000, 0, 0, 100, 5) == 3, "physical free memory admission");
    require(sv::resident_kv_slots(300, 1000, 800, 0, 100, 5) == 1, "declared device memory admission");
    require(sv::resident_kv_slots(300, 1000, 699, 0, 100, 5) == 0, "resident allocations exceed declaration");
    require(sv::resident_kv_slots(300, 1000, 0, 99, 100, 5) == 0, "explicit budget includes initial slot");
    require(sv::resident_kv_slots(0, 100, 0, 0, 100, 5) == 1, "initial slot already resident");
    // The screened-head lever's plane (CUDA) never costs a KV slot: with it 300 free of 1000 -> 3 slots of 100 bytes.
    {
        struct PlaneDecoder {
            std::uint64_t plane = 0;
            int releases = 0;
            [[nodiscard]] std::uint64_t slot_bytes() const { return 100U; }
            [[nodiscard]] std::uint64_t screen_device_bytes() const { return plane; }
            void release_screen() { ++releases; plane = 0U; }
        };
        PlaneDecoder costly{50U};  // without it 350 free -> 4 slots
        require(mini::mini_serve_kv_slots(costly, 300, 1000, 0, 0, 5) == 4U && costly.releases == 1,
                "a screened plane that costs a KV slot is released before the slots are committed");
        PlaneDecoder cheap{10U};  // without it 310 free -> 3 slots
        require(mini::mini_serve_kv_slots(cheap, 300, 1000, 0, 0, 5) == 3U && cheap.releases == 0,
                "a screened plane that costs no KV slot stays");
        PlaneDecoder none{};  // lever off, or HIP's own binding: the incumbent budget, nothing released
        require(mini::mini_serve_kv_slots(none, 300, 1000, 0, 0, 5) == sv::resident_kv_slots(300, 1000, 0, 0, 100, 5) &&
                    none.releases == 0,
                "without the lever's plane the served slot budget is the incumbent one");
        PlaneDecoder declared{100U};  // declared device cap 800: 1 slot with the plane, 2 without
        require(mini::mini_serve_kv_slots(declared, 300, 1000, 800, 0, 5) == 2U && declared.releases == 1,
                "a declared device budget counts the plane too");
    }
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

// One lever switch of the served process environment, as strix-serve reads it; nullptr clears it.
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value ? value : "") == 0, "cannot set a lever switch");
#else
    require((value ? setenv(name, value, 1) : unsetenv(name)) == 0, "cannot set a lever switch");
#endif
}
void set_w4a4_env(const char* value) { set_env("STRIX_INK_MOE_W4A4", value); }

// The prepared-attention round fold switch of the served process (off unless 1 selects it, in any profile).
void set_round_fold_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_PREPARED_ROUND_FOLD", value ? value : "") == 0, "cannot set STRIX_INK_PREPARED_ROUND_FOLD");
#else
    require((value ? setenv("STRIX_INK_PREPARED_ROUND_FOLD", value, 1) : unsetenv("STRIX_INK_PREPARED_ROUND_FOLD")) == 0,
            "cannot set STRIX_INK_PREPARED_ROUND_FOLD");
#endif
}

// The served route lever value (process environment, as strix-serve reads it); nullptr clears it.
void set_cli_route_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_SERVED_CLI_ROUTE", value ? value : "") == 0, "cannot set STRIX_INK_SERVED_CLI_ROUTE");
#else
    require((value ? setenv("STRIX_INK_SERVED_CLI_ROUTE", value, 1) : unsetenv("STRIX_INK_SERVED_CLI_ROUTE")) == 0,
            "cannot set STRIX_INK_SERVED_CLI_ROUTE");
#endif
}

// The served down-image lever value (process environment, as strix-serve reads it); nullptr clears it.
void set_image_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_COMBINE_IMAGE", value ? value : "") == 0, "cannot set STRIX_INK_COMBINE_IMAGE");
#else
    require((value ? setenv("STRIX_INK_COMBINE_IMAGE", value, 1) : unsetenv("STRIX_INK_COMBINE_IMAGE")) == 0,
            "cannot set STRIX_INK_COMBINE_IMAGE");
#endif
}

// Shape of one served start: --prefill-rows / --context / --slots as given (nullopt: omitted).
struct ServedShape {
    std::optional<unsigned> prefill_rows;
    std::optional<unsigned> context;
    std::optional<unsigned> slots;
};

void set_expert_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_MOE_EXPERTS_NVFP4", value ? value : "") == 0, "cannot set STRIX_MOE_EXPERTS_NVFP4");
#else
    require((value ? setenv("STRIX_MOE_EXPERTS_NVFP4", value, 1) : unsetenv("STRIX_MOE_EXPERTS_NVFP4")) == 0,
            "cannot set STRIX_MOE_EXPERTS_NVFP4");
#endif
}

// Run the native CLI parser and the real service factory up to its decoder
// construction boundary. The factory needs neither a device nor a listener.
void profile_parity() {
    const sv::SerialGenerationFamily family{"mini", "Mini", "MODEL", 8U, 8U, 16U, 0U, nullptr, 8U,
        true, false, [](const std::string&) -> std::shared_ptr<const sv::BpeTokenizer> { return {}; },
        true, {}, true};
    const auto parse_cli = [&](const char* tier, std::string_view backend,
                               std::initializer_list<const char*> input = {"1", "--generate", "2"}) {
        std::vector<std::string> args{"mini", "MODEL.safetensors"};
        args.insert(args.end(), input.begin(), input.end());
        args.insert(args.end(), {"--profile", tier});
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return mini::cli::parse(static_cast<int>(argv.size()), argv.data(), family, backend);
    };
    const auto refuses = [](const auto& action, std::string_view diagnostic) {
        try { action(); }
        catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find(diagnostic) != std::string_view::npos;
        }
        return false;
    };
    constexpr std::string_view exact_error = "forbidden under --profile exact";
    constexpr std::string_view consumer_error = "requires native resident model execution";
    std::optional<mini::MiniServeOptions> loaded;
    const auto create = [&](const mini::MiniServeOptions& options) {
        loaded = options;
        return mini::MiniServeCohort{
            std::make_unique<FixtureCohort>(std::make_unique<FixtureDecoder>(), 32U, 3U, 2U), {}, {}, 0U, 0U};
    };
    sv::ServiceFactories factories;
    mini::install_mini_service(factories, {create, create},
        [](const std::string&) {
            return sv::BpeTokenizer::from_parts({"a"}, {1}, {}, "o200k");
        }, {});
    const auto* factory = factories.find("inkling2");
    require(factory != nullptr, "Inkling service factory was not registered");
    sv::ServiceSelection selection;
    selection.model.name = "inkling2-mini";
    selection.arguments = {"--checkpoint", "MODEL.safetensors", "--backend", "hip", "--profile", "fast"};
    set_image_env("1");
    for (const auto* backend : {"hip", "cuda"}) {
        selection.arguments[3] = backend;
        for (const auto* tier : {"fast", "turbo", "exact"}) {
            const bool exact = std::string_view(tier) == "exact";
            set_w4a4_env(exact || std::string_view(backend) == "hip" ? nullptr : "mse");
            selection.arguments.back() = tier;
            for (const auto* value : {static_cast<const char*>(nullptr), "0", "1"}) {
                set_expert_env(value);
                loaded.reset();
                if (exact && value && std::string_view(value) == "1") {
                    const bool cli_refused = refuses([&] { (void)parse_cli(tier, backend); }, exact_error);
                    const bool server_refused = refuses([&] { (void)factory->create(selection); }, exact_error);
                    require(cli_refused && server_refused && !loaded, "CLI and server must refuse narrow EXACT");
                    continue;
                }
                const auto cli = parse_cli(tier, backend);
                const auto service = factory->create(selection);
                const bool narrow = value && std::string_view(value) == "1";
                require(service && loaded && loaded->checkpoint == "MODEL.safetensors" &&
                        loaded->tier == cli.tier.profile &&
                        loaded->experts_nvfp4 == narrow && loaded->experts_nvfp4 == mini::mini_experts_nvfp4(cli.tier) &&
                        loaded->moe_w4a4 == cli.moe_w4a4 &&
                        loaded->combine_image == mini::mini_combine_image(cli.tier),
                        "CLI and server selected different decoder inputs");
            }
        }
    }
    set_expert_env(nullptr);
    set_w4a4_env(nullptr);
    const auto previous_arguments = selection.arguments;
    selection.arguments.back() = "exact";
    for (const auto* backend : {"hip", "cuda"}) {
        selection.arguments[3] = backend;
        for (const auto* head : {static_cast<const char*>(nullptr), "1"}) {
            set_env("STRIX_INK_SCREENED_HEAD", head);
            for (const auto& [rows, level] : {std::pair<const char*, unsigned>{nullptr, 0U},
                                             {"0", 0U}, {"1", 1U}, {"2", 2U}}) {
                set_env("STRIX_INK_VERIFY_ROWS", rows);
                loaded.reset();
                const auto cli = parse_cli("exact", backend);
                const auto service = factory->create(selection);
                require(service && loaded && loaded->screened_head == (head != nullptr) &&
                            loaded->screened_head == mini::mini_screened_head(cli.tier) &&
                            loaded->verify_rows == level && loaded->verify_rows == mini::mini_verify_rows(cli.tier),
                        "CLI and server must resolve both opt-in rows independently");
            }
        }
    }
    set_env("STRIX_INK_SCREENED_HEAD", nullptr);
    set_env("STRIX_INK_VERIFY_ROWS", nullptr);
    selection.arguments = previous_arguments;

    selection.arguments[3] = "hip";
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        selection.arguments.back() = tier;
        for (const auto* value : {static_cast<const char*>(nullptr), "0", "1"}) {
            set_env("STRIX_INK_LOWBIT_TURBO", value);
            loaded.reset();
            const auto cli = parse_cli(tier, "hip");
            const auto service = factory->create(selection);
            const bool expected = std::string_view(tier) == "turbo" && value && std::string_view(value) == "1";
            require(service && loaded && cli.weights.lowbit_turbo == expected &&
                        loaded->weight_options.lowbit_turbo == expected,
                    "CLI and server require the same explicit TURBO weight lever");
        }
    }
    set_env("STRIX_INK_LOWBIT_TURBO", nullptr);
    selection.arguments.back() = "fast";
    selection.arguments.emplace_back("--qualify-q8");
    std::ostringstream receipt;
    factory->inspect(selection, receipt);
    require(receipt.str().find("weight_storage gguf.q8_0@1 weight_status unqualified") != std::string::npos,
            "inspect keeps the incumbent device-independent Q8 qualification receipt");
    selection.arguments.pop_back();
    set_w4a4_env("mse");
    selection.arguments[3] = "hip";
    selection.arguments.back() = "fast";
    loaded.reset();
    require(refuses([&] { (void)parse_cli("fast", "hip"); }, "CUDA only") &&
            refuses([&] { (void)factory->create(selection); }, "CUDA only") && !loaded,
            "CLI and server must refuse the same HIP W4A4 class before loading");

    set_w4a4_env(nullptr);
    set_expert_env("1");
    require(refuses([&] { (void)parse_cli("fast", "cuda", {"--tokenize", "input.txt"}); }, consumer_error),
            "tokenize-only CLI must refuse the narrow image");
    require(refuses([&] {
        (void)parse_cli("exact", "cuda", {"--gate-manifest", "windows.tsv", "--gate-out", "out", "--gate-arm", "NULL",
                                "--gate-rows", "2", "--gate-mode", "decode", "--gate-order-null", "--no-step-graph"});
    }, exact_error), "the evaluation-order null must use the checkpoint image");
    const auto cuda_route = [](const mini::cli::Options& options) {
        return mini::cli::resolve_cuda_step_route(options, {}, strix::weight::kBf16WeightStorage, false);
    };
    require(refuses([&] { (void)cuda_route(parse_cli("fast", "cuda")); }, consumer_error),
            "CUDA step program must refuse an image it cannot load");
    const auto resident = parse_cli("fast", "cuda", {"1", "--generate", "2", "--no-step-graph"});
    require(!cuda_route(resident).program && mini::mini_experts_nvfp4(resident.tier),
            "CUDA resident replay must keep the explicit narrow image");
    std::cout << "inkling2-mini-profile-parity: ok\n";
}

// The native column the registry preset forwards for a served --profile (models/inkling2-mini.toml [presets.*]).
strix::profile::Name served_native_tier(std::string_view tier_name) {
    namespace presets = strix::model_data::inkling2_mini::presets;
    return strix::profile::parse(tier_name == "exact" ? presets::exact::levers
                                 : tier_name == "fast" ? presets::fast::levers
                                                       : presets::turbo::levers);
}

void wiring(const char* root, const char* backend, bool qualify_q8, bool sample_capable = false,
            W4a4 w4a4 = W4a4::bf16, std::string_view tier_name = "fast", bool image = false, bool round_fold = false,
            bool cli_route = false, ServedShape shape = {}, bool controller_one_plane = false,
            bool projected_bias = false, bool screened_head = false) {
    sv::ServiceFactories factories;
    bool created = false;
    // The fixture's projected-bias submissions: set only once the served requests have run, so a report taken before
    // the server stops reads 0.
    std::uint64_t served_dispatches = 0;
    const auto create = [&](const mini::MiniServeOptions& options) {
        require(options.checkpoint == "MODEL.safetensors", "checkpoint did not reach native adapter");
        // The served prefill runs the operand class the profile resolved, never a class-less on/off.
        require(options.moe_w4a4 == w4a4, "W4A4 operand class did not reach native adapter");
        require(options.prepared_round_fold == round_fold, "prepared round fold did not reach native adapter");
        // What the served backend then runs: the CUDA preparation has no fold, so a requested fold is declined there and the
        // startup line must say so; the qualified HIP target admits it. Same admission and same name as the CLI and the gate.
        const bool hip = std::string_view(backend) == "hip";
        const bool effective = mini::mini_prepared_round_fold_admitted(hip, "gfx1151");
        require(effective == hip, "the effective round fold follows the backend");
        // The startup field comes from the function the native cohort prints it with, given the backend's admission result.
        require(std::string_view(mini::mini_round_fold_startup_field(
                                         options, {false, options.prepared_round_fold && effective})) ==
                    (!round_fold ? "off" : hip ? "on" : "declined"),
                "the served round fold is reported by its effective state");
        // Injected admission results: a declined fold never reads as the requested bit.
        auto requested = options;
        requested.prepared_round_fold = true;
        require(std::string_view(mini::mini_round_fold_startup_field(requested, {false, false})) == "declined" &&
                    std::string_view(mini::mini_round_fold_startup_field(requested, {false, true})) == "on",
                "a requested fold the backend declines is reported declined");
        requested.prepared_round_fold = false;
        require(std::string_view(mini::mini_round_fold_startup_field(requested, {false, false})) == "off",
                "an unrequested fold is reported off");
        // The served prefill resolves the down-image lever like the CLI and the gate do: off unless selected explicitly.
        require(options.combine_image == image, "down-image selection did not reach native adapter");
        require(options.controller_one_plane == controller_one_plane, "controller plane value did not reach native adapter");
        // The served route lever (opt-in in every profile column): a HIP server runs the command line's route only when
        // STRIX_INK_SERVED_CLI_ROUTE=1 selects it; a CUDA server declines it and keeps the requested bit in its receipt.
        const bool hip_backend = std::string_view(backend) == "hip";
        const bool route = cli_route && hip_backend;
        require(options.served_route_requested == cli_route && options.served_cli_route == route,
                "served route lever: opt-in, HIP only");
        const auto field = mini::mini_served_route_startup_field(options);
        // Without a decoder the tail is the offered capacity; the startup line prints what the decoder admitted.
        require(!route || (field.find(" prefill_tail_offered " + std::to_string(options.prefill_tail) + ' ') != std::string::npos &&
                           mini::mini_served_route_startup_field(options, 0U).find(" prefill_tail 0 ") != std::string::npos),
                "served route receipt: offered tail without a decoder, the decoder's admitted tail at startup");
        require(route ? field.starts_with("on qualification unqualified moe_decode packed ")
                      : field == (cli_route ? "declined" : "off"),
                "served route receipt: off, declined or on with its qualification");
        const unsigned context = shape.context.value_or(8192U);
        require(options.context == context && options.slots == shape.slots.value_or(0U), "served shape options");
        if (!route) {
            // Without the lever the served shape is the held-back one, written out here from the rule the server had
            // before the lever existed: a 128-row chunk unless --prefill-rows, clamped to the context; lanes = --slots,
            // else the clamped 128 rows / --prefill-rows; no tail; held-back prompt token; grouped decode. The one change:
            // a chunk-invariant backend (CUDA) takes the command line's implicit chunk (8192 rows in every profile)
            // with the chunk ladder 8192 -> 2048 -> 128 (the largest that keeps the incumbent slots); HIP keeps 128
            // rows. The lane cap never follows the chunk.
            const unsigned incumbent = std::min(shape.prefill_rows.value_or(128U), context);
            const auto cli_rows = mini::mini_cli_prefill_rows(
                hip_backend ? mini::MiniPrefillBackend::hip : mini::MiniPrefillBackend::cuda, options.tier);
            const unsigned chunk = hip_backend ? incumbent : std::min(shape.prefill_rows.value_or(cli_rows), context);
            const unsigned lanes = shape.slots.value_or(0U) != 0U ? *shape.slots : incumbent;
            const unsigned fallback = hip_backend || shape.prefill_rows ? 0U
                : mini::mini_cli_prefill_fallback_rows(mini::MiniPrefillBackend::cuda, options.tier);
            require(options.prefill_rows == chunk && mini::mini_serve_lane_rows(options) == lanes &&
                        options.prefill_tail == 0U && options.lane_rows == incumbent &&
                        options.prefill_fallback_rows == fallback,
                    "lever off: held-back chunk (CUDA: the command line's), incumbent lane cap, no tail");
            // The load's candidates: an implicit CUDA chunk walks down to the incumbent 128 rows; one candidate else.
            const auto ladder = mini::mini_served_chunk_ladder(options.prefill_rows, options.prefill_fallback_rows);
            std::vector<std::uint32_t> expected{chunk};
            if (fallback != 0U && fallback < chunk) expected.push_back(fallback);
            if (fallback != 0U && 128U < expected.back()) expected.push_back(128U);
            require(std::ranges::equal(ladder.view(), expected), "lever off: served chunk candidates");
            // Beside a decoding lane the CUDA held-back route serves pieces of the incumbent 128 rows (bit-safe cuts of
            // the decoder's rule); HIP keeps its whole 128-row slices.
            const auto rule = mini::mini_prefill_piece_rule(hip_backend, false);
            const auto config = mini::mini_serve_scheduling(chunk, rule);
            require(config.prompt_slice == chunk && rule.cuttable() == !hip_backend &&
                        config.prompt_slice_decoding == (!hip_backend && chunk > 128U ? 128U : 0U) &&
                        config.max_draft == 0U && !config.context_draft,
                    "lever off: one admitted chunk alone, 128-row pieces beside a decoding lane (CUDA)");
            sv::validate_prompt_pieces(config, rule, chunk);
            // The CUDA device cohort's pieces (kMiniCudaServeDecodingRows) where the chunk is larger; never priced.
            const auto wide = mini::mini_serve_scheduling(chunk, rule, mini::kMiniCudaServeDecodingRows);
            require(wide.prompt_slice_decoding ==
                            (!hip_backend && chunk > mini::kMiniCudaServeDecodingRows ? mini::kMiniCudaServeDecodingRows
                                                                                       : 0U) &&
                        wide.prompt_gap_ms == 0.0 && config.prompt_gap_ms == 0.0,
                    "lever off: 1024-row CUDA pieces beside a decoding lane, no gap pricing");
            sv::validate_prompt_pieces(wide, rule, chunk);
        } else {
            // With the lever: the command line's chunk and tail rule, emits-first; lanes = --slots, else an explicit
            // (clamped) --prefill-rows, else the served lane cap.
            const auto cli = mini::mini_prefill_schedule_from_env(true, options.tier);
            const unsigned chunk = std::min(shape.prefill_rows.value_or(cli.chunk), context);
            const unsigned lanes = shape.slots.value_or(0U) != 0U ? *shape.slots
                                   : shape.prefill_rows                ? chunk
                                                                       : mini::kMiniServeLaneRows;
            require(options.prefill_rows == chunk && mini::mini_serve_lane_rows(options) == lanes &&
                        options.prefill_tail == mini::mini_prefill_tail_capacity(chunk, cli.tail_divisor, context) &&
                        mini::mini_prefill_piece_rule(true, true).cuttable(),
                    "lever on: the command line's schedule and the served lane cap");
        }
        // The served tier is the registry preset's native column for the requested --profile (models/inkling2-mini.toml:
        // TURBO keeps the FAST column until this model has a qualified TURBO set), resolved by the native command line's
        // resolver with the same lever classes.
        const auto native = served_native_tier(tier_name);
        mini::MiniWeightOptions cli_weights;
        cli_weights.tier.profile = native;
        cli_weights.qualify_q8 = qualify_q8;
        const auto cli_tier = mini::resolve_mini_profile(cli_weights, true);
        require(options.tier == native && options.tier == cli_tier.profile &&
                    options.moe_w4a4 == mini::mini_moe_w4a4_class(cli_tier) &&
                    options.combine_image == mini::mini_combine_image(cli_tier) &&
                    options.prepared_round_fold == mini::mini_prepared_round_fold(cli_tier) &&
                    options.weight_options.qualify_q8 == qualify_q8,
                "served profile differs from the native command line's resolution of the same --profile");
        // The projected-bias attention lever resolves in the one shared resolver the CLI also reads.
        require(options.attn_projected_bias == projected_bias, "projected-bias lever did not reach native adapter");
        // The screened-head lever resolves in the same shared resolver (CLI = server) on both backends.
        require(options.screened_head == screened_head, "screened-head lever did not reach native adapter");
        const auto kind = std::string_view(backend) == "hip" ? strix::weight::Backend::hip : strix::weight::Backend::cuda;
        const strix::weight::WeightLoadScope scope{kind, {}, {}, {}, {}, {}};
        const auto weights = mini::resolve_mini_resident_weight_load(options.tier, options.weight_options, scope,
                                                                       {options.context, options.slots});
        created = true;
        // As the native cohort: an observer of the decoder's projected-bias submissions only with the lever selected.
        std::function<std::uint64_t()> observed;
        if (options.attn_projected_bias) observed = [&served_dispatches] { return served_dispatches; };
        return mini::MiniServeCohort{
            std::make_unique<FixtureCohort>(std::make_unique<FixtureDecoder>(sample_capable), 32U, 3U, 2U), weights,
            std::move(observed), 0U, 0U};
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
    std::vector<std::string> strings{"strix-serve", "--model", "inkling2-mini", "--registry-root", root,
                                     "--checkpoint", "MODEL.safetensors", "--backend", backend,
                                     "--profile", std::string(tier_name), "--speculation", "off"};
    if (qualify_q8) strings.emplace_back("--qualify-q8");
    for (const auto& [flag, value] : {std::pair{"--prefill-rows", shape.prefill_rows}, std::pair{"--context", shape.context},
                                      std::pair{"--slots", shape.slots}}) {
        if (!value) continue;
        strings.emplace_back(flag);
        strings.emplace_back(std::to_string(*value));
    }
    if (controller_one_plane) strings.insert(strings.end(), {"--controller-planes", "1"});
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
        if (rc != 0) throw std::runtime_error(error.str());
        require(!created, "Inkling profile inspection should run without device work");
        const auto expected = qualify_q8 ? "weight_storage gguf.q8_0@1 weight_status unqualified" :
                                           "weight_storage plain.bf16@1 weight_status pending";
        // The weight plan line is the FAST plan; the other tiers run the same wiring with their own plan.
        if (tier_name != "fast") return;
        require(out.str().find(expected) != std::string::npos, "inspection weight plan differs from CLI plan");
        // The route line names the committed rows per regime (why the image stays BF16 on this backend).
        const bool hip = std::string_view(backend) == "hip";
        require(hip ? out.str().find(" weight_rows prefill=gate-only@gfx1151,decode=none") !=
                          std::string::npos
                    : out.str().find("weight_rows") == std::string::npos,
                "inspection does not name the committed per-regime rows");
        if (!hip || qualify_q8) return;
        // Even an explicit one-slot server can confirm the BF16 default without a device.
        std::ostringstream one, one_error;
        strings.insert(strings.end(), {"--slots", "1", "--family-inspect"});
        const auto one_rc = run({}, one, one_error);
        strings.resize(strings.size() - 3U);
        require(one_rc == 0 && !created && one.str().find(expected) != std::string::npos,
                "a one-slot server must report the same BF16 default as an automatic-slot server");
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
    // Decode form flags: a CUDA server takes the command line's (mini_served_packed_decode; naming both is refused before
    // native load); a HIP server refuses them, its form follows the served route lever's row.
    const bool hip_backend = std::string_view(backend) == "hip";
    for (const auto* flag : {"--packed-decode", "--grouped-decode"}) {
        std::ostringstream out, error;
        strings.insert(strings.end(), {flag, "--family-inspect"});
        const auto rc = run({}, out, error);
        strings.resize(strings.size() - 2U);
        if (hip_backend)
            require(rc == 2 && !created && error.str().find("takes no decode form flag") != std::string::npos,
                    "a decode form flag reached the HIP server outside the shared lever row");
        else require(rc == 0 && !created, "the CUDA server refused a decode form flag");
    }
    {
        std::ostringstream out, error;
        strings.insert(strings.end(), {"--packed-decode", "--grouped-decode"});
        const auto rc = run({}, out, error);
        strings.resize(strings.size() - 2U);
        require(rc == 2 && !created &&
                    error.str().find(hip_backend ? "takes no decode form flag" : "are exclusive") != std::string::npos,
                "both decode form flags did not fail before native load");
    }
    // The block-candidate switch through the production start-up parser (the tier under test; EXACT refuses sampled
    // verification with its own reason, as in mini_block_drafter_test): an explicit OFF starts,
    // every other value is refused before native load with the same reason as the CLI.
    for (const auto& [value, reason] : std::initializer_list<std::pair<const char*, const char*>>{
             {"off", ""},
             {"dflash2:8:greedy", "native forward and verify identity are not bound"},
             {"dflash2:8:sampled", tier_name == "exact" ? "EXACT permits only greedy-identical verification"
                                                        : "sampled rejection hook unqualified and unbound"},
             {"dflash2:7:greedy", "requires block 8"}}) {
        std::ostringstream out, error;
        strings.emplace_back("--block-drafter");
        strings.emplace_back(value);
        strings.emplace_back("--family-inspect");
        const auto rc = run({}, out, error);
        strings.resize(strings.size() - 3U);
        if (*reason == '\0') require(rc == 0 && !created, "explicit --block-drafter off did not start");
        else require(rc == 2 && !created && error.str().find(reason) != std::string::npos,
                     "--block-drafter value was not refused with its reason before native load");
    }
    // One strix-serve process serves one worker. An explicit replica placement resolves the model's two-device form, which only
    // strix-router executes: with real model arguments (a checkpoint) the shell refuses before the adapter validates or
    // creates anything. This is the documented standalone entry point (the `strix` CLI has no Inkling route).
    {
        std::ostringstream out, error;
        strings.emplace_back("--device-count");
        strings.emplace_back("2");
        strings.emplace_back("--placement");
        strings.emplace_back("replica");
        require(run({}, out, error) == 2 && !created && error.str().find("runs behind strix-router") != std::string::npos,
                "two-device Inkling launch with a checkpoint was not refused by strix-serve");
        strings.resize(strings.size() - 4U);
        // The same query inspected (no model arguments) reports the replica placement and its executor.
        std::vector<std::string> inspection{"strix-serve", "--model", "inkling2-mini", "--registry-root", root,
                                            "--device-count", "2", "--placement", "replica", "--print-effective-config"};
        std::vector<char*> argv;
        for (auto& arg : inspection) argv.push_back(arg.data());
        std::ostringstream inspected, inspect_error;
        require(sv::run(static_cast<int>(argv.size()), argv.data(), factories, inspected, inspect_error, {}) == 0 &&
                inspected.str().find("\"form\":\"replica\"") != std::string::npos &&
                inspected.str().find("\"executed_by\":\"router\"") != std::string::npos && !created,
                "two-device Inkling inspection did not report the replica placement");
    }
    sv::ServeShell shell;
    shell.serve = [qualify_q8, sample_capable, tier_name, cli_route, backend, &served_dispatches](
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
            const auto* profile = effective.find("profile");
            const auto* weights = effective.find("weight_load");
            require(capabilities.status == 200 && profile && profile->as_string() == mini::mini_profile_name(served_native_tier(tier_name)) && weights &&
                    (tier_name != "fast" ||  // the FAST plan; the other tiers carry their own weight plan
                     (weights->find("storage")->as_string() == (qualify_q8 ? "gguf.q8_0" : "plain.bf16") &&
                      weights->find("status")->as_string() == (qualify_q8 ? "unqualified" : "pending"))),
                    "Inkling capability response does not match the loaded plan");
            // The served route lever: absent without a request (the incumbent capabilities), else requested / admitted
            // (HIP only) / qualification of the admitted route.
            const auto* route = effective.find("served_route");
            const bool admitted = cli_route && std::string_view(backend) == "hip";
            require((route != nullptr) == cli_route &&
                        (!route || (route->find("requested")->as_bool() && route->find("admitted")->as_bool() == admitted &&
                                    route->find("qualification")->as_string() == (admitted ? "unqualified" : "n/a"))),
                    "Inkling capability response misreports the served route lever");
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
            served_dispatches = 5U;
        } catch (...) {
            owner.shutdown();
            control.join();
            throw;
        }
        owner.shutdown();
        control.join();
        if (failure) std::rethrow_exception(failure);
    };
    std::ostringstream out, error, report;
    auto* const console = std::cerr.rdbuf(report.rdbuf());
    int rc = 1;
    try {
        rc = run(shell, out, error);
    } catch (...) {
        std::cerr.rdbuf(console);
        throw;
    }
    std::cerr.rdbuf(console);
    if (rc != 0) throw std::runtime_error(error.str());
    require(created, "selected backend was not constructed");
    // With the lever selected the server reports the command line's projected-bias line once, when it stops (after the
    // served requests: 5 here, never the 0 of startup); without it, no such line.
    const auto line = mini::mini_projected_bias_dispatches_line(5U) + '\n';
    const auto text = report.str();
    const auto first = text.find("Inkling attn_projected_bias dispatches");
    require(projected_bias ? first != std::string::npos && text.compare(first, line.size(), line) == 0 &&
                                 text.find("Inkling attn_projected_bias dispatches", first + 1U) == std::string::npos
                           : first == std::string::npos,
            "the served projected-bias observation differs from the lever state");
    // The socket startup re-creates the service with its own arguments (default shape).
    if (!qualify_q8 && !shape.prefill_rows && !shape.context && !shape.slots && tier_name == "fast" &&
        (std::string_view(backend) == "hip" || sample_capable)) {
        const auto* factory = factories.find("inkling2");
        require(factory != nullptr, "Inkling service factory was not registered");
        socket_startup(*factory, backend, sample_capable);
    }
}

// A selected W4A4 class on the HIP backend (no W4A4 tile there) is refused before any device work: the server never
// starts a BF16 run whose startup line reports the class.
void hip_refuses(const char* root, W4a4 selected) {
    sv::ServiceFactories factories;
    bool created = false;
    mini::MiniServeBackends backends;
    backends.hip = [&](const mini::MiniServeOptions&) -> mini::MiniServeCohort {
        created = true;
        throw std::logic_error("HIP adapter constructed for a W4A4 class");
    };
    mini::install_mini_service(factories, std::move(backends),
        [](const std::string&) {
            return sv::BpeTokenizer::from_parts({"a", "b"}, {1, 1}, {}, "o200k");
        },
        [](const sv::Json&, const sv::Json&, const sv::Json&) { return std::string("a"); });
    const char* args[]{"strix-serve", "--model", "inkling2-mini", "--registry-root", root,
                       "--checkpoint", "MODEL.safetensors", "--backend", "hip",
                       "--profile", "fast", "--speculation", "off"};
    std::vector<std::string> strings(std::begin(args), std::end(args));
    std::vector<char*> argv;
    for (auto& arg : strings) argv.push_back(arg.data());
    std::ostringstream out, error;
    const auto rc = sv::run(static_cast<int>(argv.size()), argv.data(), factories, out, error, {});
    require(rc != 0 && !created, "HIP served a selected W4A4 class");
    require(error.str().find(strix::moe_core::grouped_w4a4::identity(selected)) != std::string::npos &&
                error.str().find("CUDA only") != std::string::npos,
            "HIP refusal does not name the W4A4 class");
}
} // namespace

int main(int argc, char** argv) try {
    if (argc == 3 && std::string_view(argv[2]) == "--profile-parity") {
        profile_parity();
        return 0;
    }
    if (argc == 3 && std::string_view(argv[2]) == "--controller-planes-only") {
        wiring(argv[1], "cuda", false);
        wiring(argv[1], "cuda", false, false, W4a4::bf16, "fast", false, false, false, {}, true);
        return 0;
    }
    require(argc == 2, "expected repository root");
    set_image_env(nullptr);
    set_round_fold_env(nullptr);
    set_cli_route_env(nullptr);
    admission();
    sampling();
    for (const auto* backend : {"hip", "cuda"}) {
        wiring(argv[1], backend, false);
        wiring(argv[1], backend, true);
        // The deployed TURBO preset with the served route switch unset: main's preset column, Q8 qualification included.
        wiring(argv[1], backend, true, false, W4a4::bf16, "turbo");
        // Every tier keeps the held-back served shape without the lever (CUDA: the command line's chunk, incumbent lane
        // cap), also for chunks above the context and explicit slots.
        for (const auto* tier : {"exact", "fast", "turbo"})
            for (const ServedShape shape : {ServedShape{}, ServedShape{{}, 15U, {}}, ServedShape{8192U, 15U, {}},
                                            ServedShape{{}, 15U, 9U}, ServedShape{32U, {}, {}}, ServedShape{32U, {}, 4U}})
                wiring(argv[1], backend, false, false, W4a4::bf16, tier, false, false, false, shape);
    }
    wiring(argv[1], "cuda", false, true);
    wiring(argv[1], "cuda", false, false, W4a4::bf16, "fast", false, false, false, {}, true);
    // Every lever value reaches the served decoder as its own class (mse used to run nearest@1 there).
    for (const auto& [value, expected] : {std::pair{"1", W4a4::nvfp4}, std::pair{"mse", W4a4::nvfp4_mse},
                                          std::pair{"mse-gate-up", W4a4::nvfp4_mse_gate_up},
                                          std::pair{"mse-r2-gate-up", W4a4::nvfp4_mse_r2_gate_up},
                                          std::pair{"mse-r2", W4a4::nvfp4_mse_r2},
                                          std::pair{"r2", W4a4::nvfp4_r2}}) {
        set_w4a4_env(value);
        wiring(argv[1], "cuda", false, false, expected);
        hip_refuses(argv[1], expected);
    }
    set_w4a4_env("0");
    wiring(argv[1], "hip", false);  // the off switch still serves HIP
    set_w4a4_env(nullptr);
    // Both selections are off by default (no profile selects them); an explicit 1 reaches the adapter, an explicit 0 keeps it off.
    set_round_fold_env(nullptr);
    set_image_env(nullptr);
    wiring(argv[1], "hip", false);
    set_round_fold_env("1");
    for (const auto* backend : {"hip", "cuda"}) wiring(argv[1], backend, false, false, W4a4::bf16, "fast", false, true);
    set_round_fold_env("0");
    wiring(argv[1], "hip", false);
    set_round_fold_env(nullptr);
    set_image_env("1");
    for (const auto* backend : {"hip", "cuda"})
        wiring(argv[1], backend, false, false, W4a4::bf16, "fast", true);
    set_image_env("0");
    wiring(argv[1], "hip", false);
    set_image_env(nullptr);
    // The served route lever: off by default in every tier (above); STRIX_INK_SERVED_CLI_ROUTE=1 selects the command
    // line's route on HIP (CUDA declines it and keeps the incumbent shape) in any tier; an explicit 0 keeps it off.
    set_cli_route_env("1");
    for (const auto* backend : {"hip", "cuda"})
        for (const auto* tier : {"exact", "fast", "turbo"})
            for (const ServedShape shape : {ServedShape{}, ServedShape{{}, 15U, {}}, ServedShape{8192U, 15U, {}},
                                            ServedShape{32U, {}, {}}, ServedShape{{}, {}, 9U}})
                wiring(argv[1], backend, false, false, W4a4::bf16, tier, false, false, true, shape);
    set_cli_route_env("0");
    wiring(argv[1], "hip", false);
    set_cli_route_env(nullptr);
    // The projected-bias lever per tier (opt-in): unset and 0 keep the image route, 1 selects the projected flash;
    // tests/inkling2/mini_gate_cli_test.cpp pins the same table through the CLI.
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, false}, {"0", false}, {"1", true}}) {
            set_env("STRIX_INK_ATTN_PROJECTED_BIAS", value);
            wiring(argv[1], "cuda", false, false, W4a4::bf16, tier, false, false, false, {}, false, expected);
        }
    }
    set_env("STRIX_INK_ATTN_PROJECTED_BIAS", nullptr);
    // The screened-head lever per backend and tier (opt-in): unset and 0 keep the full head, 1 selects the screened head;
    // tests/inkling2/mini_gate_cli_test.cpp pins the same table through the CLI.
    for (const auto* backend : {"cuda", "hip"}) {
        for (const auto* tier : {"exact", "fast", "turbo"}) {
            for (const auto& [value, expected] :
                 {std::pair<const char*, bool>{nullptr, false}, {"0", false}, {"1", true}}) {
                set_env("STRIX_INK_SCREENED_HEAD", value);
                wiring(argv[1], backend, false, false, W4a4::bf16, tier, false, false, false, {}, false, false,
                       expected);
            }
        }
    }
    set_env("STRIX_INK_SCREENED_HEAD", nullptr);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

#include "inkling2/inkling2_mini_resident_route.hpp"
#include "inkling2/inkling2_mini_serve.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "strix/serve/fixed_kv_cohort.hpp"
#include "strix/serve/http_client.hpp"
#include "strix/serve/replica_router.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <utility>

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

void admission() {
    require(sv::fixed_kv_slots(99, 100, 3) == 0, "KV budget too small");
    require(sv::fixed_kv_slots(400, 100, 3) == 3, "row budget cap");
    require(sv::fixed_kv_slots(400, 100, 3, 1) == 1, "explicit slot cap");
    require(sv::resident_kv_slots(300, 1000, 0, 0, 100, 5) == 3, "physical free memory admission");
    require(sv::resident_kv_slots(300, 1000, 800, 0, 100, 5) == 1, "declared device memory admission");
    require(sv::resident_kv_slots(300, 1000, 699, 0, 100, 5) == 0, "resident allocations exceed declaration");
    require(sv::resident_kv_slots(300, 1000, 0, 99, 100, 5) == 0, "explicit budget includes initial slot");
    require(sv::resident_kv_slots(0, 100, 0, 0, 100, 5) == 1, "initial slot already resident");
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

// The served down-image lever value (process environment, as strix-serve reads it); nullptr clears it.
void set_image_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_COMBINE_IMAGE", value ? value : "") == 0, "cannot set STRIX_INK_COMBINE_IMAGE");
#else
    require((value ? setenv("STRIX_INK_COMBINE_IMAGE", value, 1) : unsetenv("STRIX_INK_COMBINE_IMAGE")) == 0,
            "cannot set STRIX_INK_COMBINE_IMAGE");
#endif
}

void wiring(const char* root, const char* backend, bool qualify_q8, bool sample_capable = false,
            W4a4 w4a4 = W4a4::bf16, bool image = false, bool round_fold = false, bool projected_bias = true,
            const char* tier_name = "fast") {
    sv::ServiceFactories factories;
    bool created = false;
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
        require(std::string_view(mini::mini_round_fold_name(options.prepared_round_fold && effective,
                                                           options.prepared_round_fold)) ==
                    (!round_fold ? "off" : hip ? "on" : "declined"),
                "the served round fold is reported by its effective state");
        // The served prefill resolves the down-image lever like the CLI and the gate do: off unless selected explicitly.
        require(options.combine_image == image, "down-image selection did not reach native adapter");
        // The served Inkling resolution maps the turbo word to its fast column (the profile resolver's decision).
        const auto want = std::string_view(tier_name) == "exact" ? strix::profile::Name::standard
                                                                 : strix::profile::Name::fast;
        require(options.tier == want && options.weight_options.qualify_q8 == qualify_q8,
                "profile request did not reach native adapter");
        // The projected-bias attention lever resolves in the one shared resolver the CLI also reads.
        require(options.attn_projected_bias == projected_bias, "projected-bias lever did not reach native adapter");
        const auto kind = std::string_view(backend) == "hip" ? strix::weight::Backend::hip : strix::weight::Backend::cuda;
        const strix::weight::WeightLoadScope scope{kind, {}, {}, {}, {}, {}};
        const auto weights = mini::resolve_mini_resident_weight_load(options.tier, options.weight_options, scope);
        created = true;
        return mini::MiniServeCohort{
            std::make_unique<FixtureCohort>(std::make_unique<FixtureDecoder>(sample_capable), 32U, 3U, 2U), weights};
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
                       "--profile", tier_name, "--speculation", "off"};
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
        // The FAST weight plan is what this check pins; the exact and turbo columns resolve their own plan.
        if (std::string_view(tier_name) == "fast")
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
    sv::ServeShell shell;
    shell.serve = [qualify_q8, sample_capable, fast = std::string_view(tier_name) == "fast"](const sv::HttpServerConfig&, sv::HttpHandler handle, sv::SessionRuntime& owner) {
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
            // The FAST column's plan is pinned here; the exact and turbo columns resolve their own weight plan.
            require(capabilities.status == 200 && profile && weights &&
                    (!fast || (profile->as_string() == "fast" &&
                               weights->find("storage")->as_string() == (qualify_q8 ? "gguf.q8_0" : "plain.bf16") &&
                               weights->find("status")->as_string() == (qualify_q8 ? "unqualified" : "pending"))),
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
    std::ostringstream out, error;
    const auto rc = run(shell, out, error);
    if (rc != 0) throw std::runtime_error(error.str());
    require(created, "selected backend was not constructed");
    if (!qualify_q8 && (std::string_view(backend) == "hip" || sample_capable)) {
        const auto* factory = factories.find("inkling2");
        require(factory != nullptr, "Inkling service factory was not registered");
        socket_startup(*factory, backend, sample_capable);
    }
}
} // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "expected repository root");
    set_image_env(nullptr);
    set_round_fold_env(nullptr);
    admission();
    sampling();
    for (const auto* backend : {"hip", "cuda"}) {
        wiring(argv[1], backend, false);
        wiring(argv[1], backend, true);
    }
    wiring(argv[1], "cuda", false, true);
    // Every lever value reaches the served decoder as its own class (mse used to run nearest@1 there).
    for (const auto& [value, expected] : {std::pair{"1", W4a4::nvfp4}, std::pair{"mse", W4a4::nvfp4_mse},
                                          std::pair{"mse-gate-up", W4a4::nvfp4_mse_gate_up}}) {
        set_w4a4_env(value);
        wiring(argv[1], "cuda", false, false, expected);
    }
    set_w4a4_env(nullptr);
    // Both selections are off by default (no profile selects them); an explicit 1 reaches the adapter, an explicit 0 keeps it off.
    set_round_fold_env(nullptr);
    set_image_env(nullptr);
    wiring(argv[1], "hip", false);
    set_round_fold_env("1");
    for (const auto* backend : {"hip", "cuda"}) wiring(argv[1], backend, false, false, W4a4::bf16, false, true);
    set_round_fold_env("0");
    wiring(argv[1], "hip", false);
    set_round_fold_env(nullptr);
    set_image_env("1");
    for (const auto* backend : {"hip", "cuda"}) wiring(argv[1], backend, false, false, W4a4::bf16, true);
    set_image_env("0");
    wiring(argv[1], "hip", false);
    set_image_env(nullptr);
    // The projected-bias lever per tier: unset and 1 select the projected flash (the default), 0 keeps the image route; tests/inkling2/mini_gate_cli_test.cpp pins the same table through the CLI.
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, true}, {"0", false}, {"1", true}}) {
            set_env("STRIX_INK_ATTN_PROJECTED_BIAS", value);
            wiring(argv[1], "cuda", false, false, W4a4::bf16, false, false, expected, tier);
        }
    }
    set_env("STRIX_INK_ATTN_PROJECTED_BIAS", nullptr);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

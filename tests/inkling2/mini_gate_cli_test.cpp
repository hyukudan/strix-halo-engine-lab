#if defined(STRIX_INKLING2_APP_CLI_TEST)
#include "inkling2_mini_cli.hpp"
#endif
#include "inkling2/inkling2_mini_resident_route.hpp"
#include "strix/session/cli_output.hpp"
#include "strix/session/head_gate_cli.hpp"

#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

#if defined(STRIX_INKLING2_APP_CLI_TEST)
namespace cli = strix::inkling2::cli;
#endif
namespace session = strix::session;
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Check>
void expect_invalid(Check check) {
    try {
        check();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid_argument");
}

struct TempDir {
    fs::path path = fs::current_path() /
        ("strix-mini-gate-" + std::to_string(session::CliClock::now().time_since_epoch().count()));
    TempDir() { require(fs::create_directory(path), "could not create test directory"); }
    ~TempDir() {
        std::error_code error;
        for (const char* name : {"a.ids", "b.ids", "windows.tsv", "second-REF.f32"})
            fs::remove(path / name, error);
        fs::remove(path, error);
    }
};

struct Decoder {
    std::vector<unsigned> tokens;
    unsigned resets = 0;
    unsigned prefixes = 0;

    void reset_gate_session() {
        ++resets;
        tokens.clear();
    }

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) {
        require(logits.empty() && tokens.empty(), "teacher prefix unexpectedly requests logits or reused state");
        ++prefixes;
        tokens.assign(ids.begin(), ids.end());
        return 99U;
    }

    unsigned decode(unsigned token, std::span<float> logits) {
        tokens.push_back(token);
        require(logits.size() == 2U, "expected one vocabulary row");
        logits[0] = static_cast<float>(token);
        logits[1] = static_cast<float>(tokens.size());
        return 99U;
    }

    void prefill_gate_rows(std::span<const unsigned> ids, std::span<float> logits) {
        require(tokens.empty(), "stale state across windows");
        tokens.assign(ids.begin(), ids.end());
        std::ranges::fill(logits, 4.0F);
    }
};

#if defined(STRIX_INKLING2_APP_CLI_TEST)
cli::Options parse(const std::vector<std::string>& arguments, std::string_view backend = "cuda",
                   cli::StepProgramOptions* program = nullptr) {
    std::vector<std::string> storage{"mini", "model"};
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (auto& value : storage) argv.push_back(value.data());
    if (program) *program = cli::extract_step_program(argv);
    const strix::serve::SerialGenerationFamily family{"mini", "Mini", "MODEL",
        8U, 8U, 16U, 0U, nullptr, 8U, true, false,
        [](const std::string&) -> std::shared_ptr<const strix::serve::BpeTokenizer> { return {}; },
        strix::inkling2::kMiniPackedDecodeDefault, [](std::string_view) { return std::string{}; }, true};
    auto options = cli::parse(static_cast<int>(argv.size()), argv.data(), family, backend);
    // Only use fields that own their values after the temporary argument storage expires.
    return options;
}

void test_native_w4a4_admission() {
    namespace w4a4 = strix::moe_core::grouped_w4a4;
    using A = w4a4::Activations;
    const auto set_lever = [](const char* value) {
#if defined(_WIN32)
        require(_putenv_s("STRIX_INK_MOE_W4A4", value) == 0, "cannot set the W4A4 lever");
#else
        require(setenv("STRIX_INK_MOE_W4A4", value, 1) == 0, "cannot set the W4A4 lever");
#endif
    };
    const std::initializer_list<std::string> gate{"--gate-manifest", "missing.tsv", "--gate-out", "out", "--gate-arm",
        "FAST", "--gate-rows", "2", "--gate-mode", "prefill", "--profile", "fast", "--qualify-q8"};
    const std::initializer_list<std::string> generation{"1", "2", "--profile", "fast", "--qualify-q8"};
    for (const auto& [value, expected] : std::array{
            std::pair{"mse-r2-gate-up", A::nvfp4_mse_r2_gate_up}, std::pair{"mse-r2", A::nvfp4_mse_r2},
            std::pair{"r2", A::nvfp4_r2}}) {
        set_lever(value);
        for (const auto& args : {gate, generation}) {
            bool refused = false;
            try {
                (void)parse(args, "hip");
            } catch (const std::invalid_argument& error) {
                const std::string_view message = error.what();
                refused = message.find(w4a4::identity(expected)) != std::string_view::npos &&
                          message.find("CUDA only") != std::string_view::npos;
            }
            require(refused, "native CLI did not refuse the selected HIP W4A4 class before loading");
            require(parse(args, "cuda").moe_w4a4 == expected, "native CLI lost the admitted CUDA class");
        }
    }
    set_lever("0");
    require(parse(gate, "hip").moe_w4a4 == A::bf16, "HIP gate with W4A4 off must still parse");
    require(parse(generation, "hip").moe_w4a4 == A::bf16, "HIP generation with W4A4 off must still parse");
}

void test_cuda_route_options() {
    using Mode = strix::serve::CliSpeculation;
    constexpr auto bf16 = strix::weight::kBf16WeightStorage;
    constexpr auto q8 = strix::weight::kQ8WeightStorage;
    // The row default (auto) on CUDA: the verifying route (resident replay: drafter, arbiter, verify window program).
    {
        auto by_row = parse({"1", "--generate", "4"});
        cli::StepProgramOptions defaults;
        const auto route = cli::resolve_cuda_step_route(by_row, defaults, bf16, false);
        require(by_row.generation.speculation == Mode::automatic && !route.program &&
                    route.speculation == Mode::automatic && route.speculation_reason.empty() && route.plain_program,
                "the speculation row default on CUDA resolves the verifying route, plain rows on the step program");
        // A run that cannot draft under the default (token sampling; an embeddings prompt is known only at load) keeps
        // the verifying loader; the plain-row route then carries every decode row (DeviceCliDecoder::decode), so it
        // must stay bound for it.
        auto sampled = parse({"1", "--generate", "4", "--temperature", "0.7"});
        const auto sampled_route = cli::resolve_cuda_step_route(sampled, defaults, bf16, false);
        require(!sampled_route.program && sampled_route.plain_program,
                "a sampled run under the default binds the step program for its plain rows");
        require(!cli::resolve_cuda_step_route(by_row, defaults, q8, false).plain_program,
                "Q8 weights keep every row on the resident decoder (the step program needs BF16)");
    }
    auto options = parse({"1", "--generate", "4", "--speculation", "off"});
    cli::StepProgramOptions program;
    const auto resolve = [&](strix::weight::FormatId storage = strix::weight::kBf16WeightStorage,
                             bool resident_k1 = false) {
        return cli::resolve_cuda_step_route(options, program, storage, resident_k1);
    };
    require(resolve().program && resolve().speculation == Mode::off,
            "ordinary E1 CUDA decode did not resolve to the default program");
    options.generation.packed_decode = false;
    require(resolve().program, "grouped decode did not keep the E1 executor");
    program.requested = true;
    require(resolve().program && resolve().speculation_reason.empty(),
            "explicit grouped step program with speculation off");

    options.generation.speculation = Mode::automatic;
    const auto automatic = resolve();
    require(automatic.program && automatic.speculation == Mode::off && !automatic.speculation_reason.empty(),
            "automatic speculation without verifier must report its effective state");
    expect_invalid([&] { (void)resolve(q8); });
    require(resolve(bf16, true).program, "explicit native K1 preparation did not keep the BF16 program");
    program.requested = false;
    require(!resolve().program && resolve().speculation == Mode::automatic, "auto did not take the verifying route");
    require(!resolve(bf16, true).program && resolve(bf16, true).speculation == Mode::automatic,
            "auto with native K1 preparation did not keep the verifying route");
    expect_invalid([&] { (void)resolve(q8, true); });
    require(!resolve(q8).program && resolve(q8).speculation == Mode::automatic, "auto on Q8 left the resident replay");
    program.latency_forms_set = true;  // an explicit step-program option keeps the program (auto declined, reported)
    require(resolve().program && resolve().speculation == Mode::off && !resolve().speculation_reason.empty(),
            "an explicit step-program option did not keep the program under auto");
    program.latency_forms_set = false;

    options.step_graphs = false;
    require(!resolve(q8).program && resolve(q8).speculation == Mode::automatic,
            "--no-step-graph did not explicitly select resident Q8+auto");
    require(!resolve(bf16, true).program && resolve(bf16, true).speculation == Mode::automatic,
            "explicit replay did not preserve the BF16 resident K1 binding");
    expect_invalid([&] { (void)resolve(q8, true); });
    program.epilogue_fusion = false;
    expect_invalid([&] { (void)resolve(q8); });
    options.step_graphs.reset();
    require(resolve().program, "EPI toggle removed the E1 route");
    program.epilogue_fusion = true;

    options.step_graphs = true;
    require(!resolve(q8).program, "--step-graph did not explicitly select resident Q8");
    program.requested = true;
    expect_invalid([&] { (void)resolve(); });
    program.requested = false;
    options.step_graphs.reset();

    options.dspark = "sidecar.safetensors";
    require(!resolve().program && resolve().speculation == Mode::automatic,
            "auto with a drafter file did not take the verifying route");
    options.generation.speculation = Mode::off;
    expect_invalid([&] { (void)resolve(); });
    options.generation.speculation = Mode::automatic;
    options.step_graphs = false;
    require(!resolve().program, "explicit resident DSpark route was lost");
    options.step_graphs.reset();
    options.dspark = nullptr;
    for (auto mode : {Mode::lookup, Mode::verify}) {
        options.generation.speculation = mode;
        expect_invalid([&] { (void)resolve(); });
        options.step_graphs = false;
        require(resolve().speculation == mode, "resident forced drafting changed");
        options.step_graphs.reset();
    }
    options.generation.speculation = Mode::off;
    options.gate_order_null = true;
    expect_invalid([&] { (void)resolve(); });
    options.step_graphs = false;
    require(!resolve().program, "evaluation-order null must retain the explicitly selected resident route");
    // A null window that never reached the reversed order fails closed; an applied one is reported.
    bool unapplied_refused = false;
    try {
        (void)strix::inkling2::cli::gate_null_receipt("{\"version\":1}", 0U, "w0");
    } catch (const std::runtime_error&) {
        unapplied_refused = true;
    }
    require(unapplied_refused, "an unapplied evaluation-order null must not produce a receipt");
    require(strix::inkling2::cli::gate_null_receipt("{\"version\":1}", 48U, "w0") ==
                "{\"version\":1,\"null\":{\"order\":\"relative-bias-reverse-fma@1\",\"applications\":48}}",
            "evaluation-order null receipt changed");

    options.step_graphs = false;
    program.latency_forms_set = true;
    expect_invalid([&] { (void)resolve(); });
    program.latency_forms_set = false;
    options.step_graphs.reset();
    std::vector<std::string> argv_storage{"mini", "model", "--prompt", "--step-program", "--no-epilogue-fusion"};
    std::vector<char*> argv;
    for (auto& value : argv_storage) argv.push_back(value.data());
    const auto controls = cli::extract_step_program(argv);
    require(!controls.requested && !controls.epilogue_fusion && argv.size() == 4 &&
            std::string_view(argv.back()) == "--step-program", "EPI switch parsed a prompt payload as an option");
    require(controls.latency_forms == cli::kLatencyFormsMix && !controls.latency_forms_set,
            "mix latency forms are the default");
    const auto forms = [](std::vector<std::string> extra) {
        std::vector<std::string> storage{"mini", "model"};
        storage.insert(storage.end(), extra.begin(), extra.end());
        std::vector<char*> args;
        for (auto& value : storage) args.push_back(value.data());
        return cli::extract_step_program(args);
    };
    require(forms({"--no-latency-forms"}).latency_forms == 0U, "--no-latency-forms restores the incumbent leaves");
    require(forms({"--latency-forms"}).latency_forms == cli::kLatencyFormsAll, "--latency-forms binds every form");
    require(forms({"--latency-forms=mix"}).latency_forms == cli::kLatencyFormsMix, "--latency-forms=mix");
    require(forms({"--latency-forms=moe"}).latency_forms == cli::kLatencyFormsMoe, "--latency-forms=moe");
    require(forms({"--latency-forms=batch"}).latency_forms == cli::kLatencyFormsBatch, "--latency-forms=batch");
    require(forms({"--latency-forms=b3"}).latency_forms == cli::kLatencyFormsB3, "--latency-forms=b3");
    expect_invalid([&] { (void)forms({"--latency-forms", "--no-latency-forms"}); });
}

void test_strixpack_entry_points() {
    namespace mini = strix::inkling2;
    namespace weight = strix::weight;
    struct Entry {
        const char* name;
        const char* backend;
        bool step_program;
        bool explicit_program = false;
        bool gate = false;
        bool default_route = false;  // no route flag: the CLI's own default (auto: the verifying route)
    };
    constexpr Entry entries[]{
        {"HIP CLI", "hip", false},
        {"CUDA CLI", "cuda", false},
        {"automatic CUDA route (auto speculation: resident replay)", "cuda", false, false, false, true},
        {"explicit CUDA step program", "cuda", true, true},
        {"HIP gate", "hip", false, false, true},
        {"CUDA gate", "cuda", true, false, true},
    };
    unsigned failures = 0;
    for (const auto& entry : entries) {
        const bool hip = std::string_view(entry.backend) == "hip";
        const weight::WeightLoadScope scope{hip ? weight::Backend::hip : weight::Backend::cuda, {}, {}, {}, {}, {}};
        const weight::WeightLoadExtent extent{4U, 1U};
        unsigned loads = 0;
        const auto run = [&](const char* policy) {
            std::vector<std::string> args{"--profile", "fast"};
            if (entry.gate)
                args.insert(args.end(), {"--gate-manifest", "windows.tsv", "--gate-out", "out",
                                         "--gate-arm", "FAST", "--gate-rows", "2", "--gate-mode", "decode"});
            else
                args.insert(args.end(), {"1", "--generate", "3"});
            if (!hip && !entry.step_program && !entry.default_route) args.emplace_back("--no-step-graph");
            if (entry.explicit_program) args.emplace_back("--step-program");
            if (policy) args.insert(args.end(), {"--qualify-weights", policy});
            cli::StepProgramOptions program;
            auto options = parse(args, entry.backend, hip ? nullptr : &program);
            auto plan = mini::resolve_mini_resident_weight_load(options.tier.profile, options.weights, scope, extent);
            if (!hip) {
                options.gate_manifest = entry.gate ? "windows.tsv" : nullptr;
                const auto route = cli::resolve_cuda_step_route(options, program, plan.storage, false);
                require(route.program == entry.step_program, "SP test selected the wrong CUDA route");
                if (route.program && !entry.gate) {
                    ++loads;
                    return plan;
                }
            }
            std::ostringstream receipt;
            mini::load_mini_weight_pack(plan, options.tier.profile, options.weights, scope, extent, {}, nullptr, receipt,
                [&](const mini::MiniLowBitRows&) { ++loads; return 0; });
            return plan;
        };
        constexpr std::array formats{weight::kBf16WeightStorage, weight::kQ8WeightStorage};
        const auto incumbent = mini::resolve_mini_weight_load(strix::profile::Name::fast, scope, formats, true, false,
                                                               mini::kMiniWeightLoadDecisions, extent);
        const auto unset = run(nullptr);
        require(loads == 1U && weight::same_id(unset.policy, incumbent.policy) && unset.storage == incumbent.storage &&
                    unset.head_storage == incumbent.head_storage && unset.status == incumbent.status &&
                    unset.reason == incumbent.reason, "unset SP changed the incumbent FAST load");
        for (const char* policy : {"weights.attention_head.sp2@1", "weights.attention_head.sp3@1",
                                   "weights.attention_head.sp4@1"}) {
            loads = 0;
            bool refused = false;
            try {
                (void)run(policy);
            } catch (const std::invalid_argument& error) {
                refused = std::string_view(error.what()) == "refused: loader not present";
            }
            if (!refused || loads != 0U) {
                std::cerr << entry.name << ": " << policy << " reached routing without the SP refusal\n";
                ++failures;
            }
        }
    }
    require(failures == 0U, "explicit SP must refuse before the load callback");
}

void set_round_fold_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_INK_PREPARED_ROUND_FOLD", value ? value : "") == 0, "cannot set the round fold switch");
#else
    require((value ? setenv("STRIX_INK_PREPARED_ROUND_FOLD", value, 1) : unsetenv("STRIX_INK_PREPARED_ROUND_FOLD")) == 0,
            "cannot set the round fold switch");
#endif
}

// The projected-bias attention lever per tier through the CLI parse: the same table the server test pins
// (tests/inkling2/mini_serve_test.cpp): default on, so unset and 1 on, an explicit 0 off, anything else refused.
void test_projected_bias_lever() {
    const auto set = [](const char* value) {
#ifdef _WIN32
        require(_putenv_s("STRIX_INK_ATTN_PROJECTED_BIAS", value ? value : "") == 0, "cannot set the lever");
#else
        require((value ? setenv("STRIX_INK_ATTN_PROJECTED_BIAS", value, 1)
                       : unsetenv("STRIX_INK_ATTN_PROJECTED_BIAS")) == 0, "cannot set the lever");
#endif
    };
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, true}, {"0", false}, {"1", true}}) {
            set(value);
            require(strix::inkling2::mini_attn_projected_bias(parse({"1", "--profile", tier}).tier) == expected,
                    "CLI projected-bias lever differs from the server table");
        }
    }
    set("yes");
    expect_invalid([] { (void)parse({"1", "--profile", "fast"}); });
    set(nullptr);
}

// The screened-head lever per tier through the CLI parse: the same table the server test pins
// (tests/inkling2/mini_serve_test.cpp): default on, so unset and 1 the screened head, an explicit 0 the full head, anything else refused.
void test_screened_head_lever() {
    const auto set = [](const char* value) {
#ifdef _WIN32
        require(_putenv_s("STRIX_INK_SCREENED_HEAD", value ? value : "") == 0, "cannot set the lever");
#else
        require((value ? setenv("STRIX_INK_SCREENED_HEAD", value, 1) : unsetenv("STRIX_INK_SCREENED_HEAD")) == 0,
                "cannot set the lever");
#endif
    };
    for (const auto* tier : {"exact", "fast", "turbo"}) {
        for (const auto& [value, expected] : {std::pair<const char*, bool>{nullptr, true}, {"0", false}, {"1", true}}) {
            set(value);
            require(strix::inkling2::mini_screened_head(parse({"1", "--profile", tier}).tier) == expected,
                    "CLI screened-head lever differs from the server table");
        }
    }
    set("yes");
    expect_invalid([] { (void)parse({"1", "--profile", "fast"}); });
    set(nullptr);
}

void test_options() {
    using strix::profile::Name;
    require(parse({"1"}).tier.profile == Name::standard, "expected exact default");
    require(parse({"1", "--profile", "exact"}).tier.profile == Name::standard, "expected explicit exact");
    require(parse({"1", "--profile", "fast"}).tier.profile == Name::fast, "expected FAST");
    require(parse({"1", "--profile", "turbo"}).tier.profile == Name::turbo, "expected TURBO");
    // The prepared-attention round fold is off in every profile by default; STRIX_INK_PREPARED_ROUND_FOLD=1 selects it
    // through the same resolution the server reads (tests/inkling2/mini_serve_test.cpp).
    set_round_fold_env(nullptr);
    for (const auto* tier : {"exact", "fast", "turbo"})
        require(!strix::inkling2::mini_prepared_round_fold(parse({"1", "--profile", tier}).tier), "round fold off by default");
    set_round_fold_env("1");
    for (const auto* tier : {"exact", "fast", "turbo"})
        require(strix::inkling2::mini_prepared_round_fold(parse({"1", "--profile", tier}).tier), "explicit round fold");
    set_round_fold_env(nullptr);
    require(strix::inkling2::mini_controller_one_plane(parse({"1", "--profile", "fast", "--controller-planes", "1"}).tier),
            "CLI must resolve the explicit one-plane candidate");
    require(!strix::inkling2::mini_controller_one_plane(parse({"1", "--profile", "fast", "--controller-planes", "3"}).tier),
            "CLI must resolve the three-plane incumbent");
    expect_invalid([] { (void)parse({"1", "--profile", "exact", "--controller-planes", "1"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fast", "--controller-planes", "2"}); });
    require(parse({"1", "--profile", "fast", "--qualify-q8"}).weights.qualify_q8, "expected Q8 qualification arm");
    require(parse({"1", "--profile", "fast", "--no-weight-q8"}).weights.disable_q8, "expected Q8 off switch");
    expect_invalid([] { (void)parse({"1", "--qualify-q8"}); });
    require(parse({"1", "--profile", "turbo", "--qualify-q8"}).weights.qualify_q8, "TURBO nests the Q8 qualification arm");
    expect_invalid([] { (void)parse({"1", "--profile", "exact", "--qualify-q8"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fast", "--no-weight-q8", "--qualify-q8"}); });
    expect_invalid([] { (void)parse({"--tokenize", "input.txt", "--profile", "fast", "--qualify-q8"}); });
    require(parse({"1", "--profile", "turbo", "--qualify-weights", "weights.attention_head.q4_g64@1"}).weights.qualify ==
                &strix::weight::kAttentionHeadQ4Policy, "expected low-bit TURBO qualification arm");
    expect_invalid([] { (void)parse({"1", "--qualify-weights", "weights.attention_head.q4_g64@1"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fast", "--qualify-weights", "weights.unknown@1"}); });
    require(parse({"1"}).generation.packed_decode, "expected packed default");
    require(!parse({"1", "--grouped-decode"}).generation.packed_decode, "expected grouped override");
    expect_invalid([] { (void)parse({"1", "--packed-decode", "--grouped-decode"}); });
    // The command line's decode form for every tier and flag, against the shared resolver the server also calls
    // (the server side of the same table is tested in mini_serve_test).
    for (const char* tier : {"exact", "fast", "turbo"}) {
        const bool packed = strix::inkling2::mini_packed_decode(false, false);
        require(parse({"1", "--profile", tier}).generation.packed_decode == packed, "tier changed the default decode form");
        require(parse({"1", "--profile", tier, "--packed-decode"}).generation.packed_decode ==
                    strix::inkling2::mini_packed_decode(true, false), "--packed-decode not honoured");
        require(parse({"1", "--profile", tier, "--grouped-decode"}).generation.packed_decode ==
                    strix::inkling2::mini_packed_decode(false, true), "--grouped-decode not honoured");
        expect_invalid([&] { (void)parse({"1", "--profile", tier, "--grouped-decode", "--packed-decode"}); });
    }
    expect_invalid([] { (void)parse({"--messages", "chat.json", "--tokenize", "input.txt"}); });
    expect_invalid([] { (void)parse({"--tokenize", "input.txt", "--generate", "0"}); });
    expect_invalid([] { (void)parse({"--gate-rows", "64"}); });
    expect_invalid([] { (void)parse({"1", "--gate-order-null"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fake"}); });
    require(parse({"1", "--speculation", "auto", "--dspark", "mtp.safetensors"}).generation.speculation ==
                strix::serve::CliSpeculation::automatic,
            "expected DSpark in automatic speculation");
    expect_invalid([] { (void)parse({"1", "--speculation", "off", "--dspark", "mtp.safetensors"}); });
    // The speculation row (kMiniSpeculation): auto by default in every tier; an explicit mode is recorded and wins;
    // --dspark off switches the drafter off (no sidecar lookup), and a speculating run takes the verify rows body.
    require(parse({"1"}).generation.speculation == strix::serve::CliSpeculation::automatic &&
                !parse({"1"}).generation.speculation_requested,
            "speculation row default: auto");
    require(parse({"1", "--speculation", "off"}).generation.speculation == strix::serve::CliSpeculation::off &&
                parse({"1", "--speculation", "off"}).generation.speculation_requested,
            "an explicit --speculation off wins over the row");
    require(parse({"1", "--speculation", "auto"}).generation.speculation_requested, "explicit speculation recorded");
    {
        const auto off = parse({"1", "--speculation", "auto", "--dspark", "off"});
        require(off.dspark_off && off.dspark == nullptr &&
                    off.generation.speculation == strix::serve::CliSpeculation::automatic,
                "--dspark off: no drafter file, speculation unchanged");
    }
    require(strix::inkling2::mini_speculation_verify_rows(0U, true) == 1U &&
                strix::inkling2::mini_speculation_verify_rows(2U, true) == 2U &&
                strix::inkling2::mini_speculation_verify_rows(0U, false) == 0U,
            "a speculating run takes verify rows level >= 1");
    require(strix::inkling2::mini_served_draft_by_live() == std::vector<std::uint32_t>({7U, 0U}),
            "served drafts by live lanes: the model's data");
    expect_invalid([] {
        (void)parse({"1", "--speculation", "auto", "--dspark", "mtp.safetensors", "--temperature", "1"});
    });
    const auto gate = parse({"--gate-manifest", "windows.tsv", "--gate-out", "out", "--gate-arm", "REF",
                             "--gate-rows", "2", "--gate-mode", "decode"});
    require(gate.gate_mode == session::HeadGateMode::decode && gate.gate_rows == 2U, "gate options differ");
    require(parse({"--gate-manifest", "windows.tsv", "--gate-out", "out", "--gate-arm", "NULL",
                   "--gate-rows", "2", "--gate-mode", "decode", "--gate-order-null"}).gate_order_null,
            "expected evaluation-order null arm");
    std::ostringstream profile_line;
    cli::write_profile(gate.tier, profile_line, true);
    // The speculation row's builtin (auto) reports no lever; the two CUDA EXACT defaults are the only levers off their
    // builtin, named in table order.
    require(profile_line.str() == "profile exact levers inkling2-mini-attn-projected-bias@1=1 "
                                  "inkling2-mini-screened-head@1=1 reference BASE\n",
            "device profile receipt differs");
}

#endif

void test_windows(const fs::path& directory) {
    using session::HeadGateMode;
    const auto manifest = directory / "windows.tsv";
    std::ofstream(directory / "a.ids") << "1 2 3 4\n";
    std::ofstream(directory / "b.ids") << "5\n6 7\n";
    std::ofstream(manifest) << "# held-out\n\nfirst 4 a.ids\nsecond 3 b.ids\n";
    const auto read = [&](HeadGateMode mode, unsigned rows) {
        return session::read_head_gate_manifest(manifest, mode, rows, 16U, 8U);
    };
    const auto windows = read(HeadGateMode::decode, 2U);
    require(windows.size() == 2U && windows[0].name == "first" && windows[1].ids.back() == 7U,
            "parsed windows differ");
#if defined(STRIX_INKLING2_APP_CLI_TEST)
    const auto extent = cli::gate_weight_extent(windows);
    require(extent.context == 4U && extent.lanes == 1U,
            "the gate weight plan must use the longest full window and one decode lane");
#endif
    expect_invalid([&] { (void)read(HeadGateMode::decode, 3U); });
    require(read(HeadGateMode::prefill, 3U).size() == 2U, "prefill permits an all-scored prompt");
    std::ofstream(manifest) << "dup 4 a.ids\ndup 3 b.ids\n";
    expect_invalid([&] { (void)read(HeadGateMode::prefill, 2U); });
    std::ofstream(manifest) << "../bad 4 a.ids\n";
    expect_invalid([&] { (void)read(HeadGateMode::prefill, 2U); });
    std::ofstream(manifest) << "bad 3 a.ids\n";
    expect_invalid([&] { (void)read(HeadGateMode::prefill, 2U); });
    std::ofstream(manifest) << "bad 4 b.ids\n";
    expect_invalid([&] { (void)read(HeadGateMode::prefill, 2U); });
    std::ofstream(directory / "b.ids") << "5 6 8\n";
    std::ofstream(manifest) << "bad 3 b.ids\n";
    expect_invalid([&] { (void)read(HeadGateMode::prefill, 2U); });

    Decoder decoder;
    std::array<float, 4> logits{};
    const auto run = [&](std::size_t index, HeadGateMode mode) {
        session::run_head_gate_window(decoder, windows[index].ids, mode, 2U, 2U, logits);
    };
    run(0, HeadGateMode::decode);
    require(decoder.tokens == windows[0].ids && logits == std::array{3.0F, 3.0F, 4.0F, 4.0F},
            "decode must consume the teacher suffix and emit logits after each token");
    run(1, HeadGateMode::decode);
    require(decoder.resets == 2U && decoder.prefixes == 2U && decoder.tokens == windows[1].ids &&
            logits == std::array{6.0F, 2.0F, 7.0F, 3.0F}, "second decode window reused the first");
    const auto dump = directory / "second-REF.f32";
    session::write_logits(dump, logits);
    require(fs::file_size(dump) == 16U, "dump size must be 2 rows times 2 logits times 4 bytes");
    std::ifstream input(dump, std::ios::binary);
    std::array<unsigned char, 16> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(input.good(), "could not read the complete dump");
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const auto bits = std::bit_cast<std::uint32_t>(logits[index]);
        for (unsigned byte = 0; byte < 4U; ++byte)
            require(bytes[4U * index + byte] == static_cast<unsigned char>(bits >> (8U * byte)),
                    "dump is not row-major little-endian FP32");
    }
    run(0, HeadGateMode::prefill);
    require(decoder.resets == 3U && decoder.prefixes == 2U && decoder.tokens == windows[0].ids &&
            logits == std::array{4.0F, 4.0F, 4.0F, 4.0F}, "prefill must consume the full prompt");
}

}  // namespace

// Review r63 P3 / r75 P3: the command line writes the weight regime receipt once (one emitter, write_weight_load_plan),
// checked through the receipt writers the GPU command line itself calls (gate window receipt and run receipt).
void test_weight_receipt_once() {
    const auto count = [](const std::string& text, std::string_view key) {
        std::size_t n = 0;
        for (auto at = text.find(key); at != std::string::npos; at = text.find(key, at + key.size())) ++n;
        return n;
    };
    namespace weight = strix::weight;
    const auto& policy = weight::kAttentionQ4HeadQ6Policy;
    weight::WeightLoadPlan plan{};
    plan.policy = policy.id;
    plan.storage = policy.target;
    plan.head_storage = policy.head_target();
    plan.turbo_quality = "fixture quality";
    const auto check = [&](const std::string& text) {
        constexpr std::string_view plan_and_quality =
            "strix-weight-load/v1 weights.attention_q4_g64.head_q6_g64@1 strix.q4_g64@1 head strix.q6_g64@1\n"
            "strix-weight-load-quality/v1 weights.attention_q4_g64.head_q6_g64@1 fixture quality\n"
            "strix-weight-load-regimes/v1 ";
        require(count(text, "strix-weight-load/v1 ") == 1U, "one weight plan record");
        require(count(text, "strix-weight-load-quality/v1 ") == 1U, "one weight quality record");
        require(count(text, "strix-weight-load-regimes/v1 ") == 1U, "one weight regime record");
        require(text.find(plan_and_quality) != std::string::npos,
                "weight plan with head storage, then quality, then regimes");
    };
    std::ostringstream gate;
    cli::write_gate_receipt(gate, plan, "{\"version\":1}");
    check(gate.str());
    require(count(gate.str(), "strix-route-file/v1 ") == 1U, "gate receipt: one route record");
#if defined(STRIX_INKLING2_APP_CLI_TEST)
    for (const char* tier : {"exact", "fast", "turbo"}) {
        std::ostringstream run;
        cli::write_run_receipt(parse({"1", "--profile", tier}).tier, plan, run);
        check(run.str());
    }
#endif
}

int main() {
    try {
        const TempDir directory;
        test_weight_receipt_once();
#if defined(STRIX_INKLING2_APP_CLI_TEST)
        test_options();
        test_projected_bias_lever();
        test_screened_head_lever();
        test_native_w4a4_admission();
        test_cuda_route_options();
        test_strixpack_entry_points();
#endif
        test_windows(directory.path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

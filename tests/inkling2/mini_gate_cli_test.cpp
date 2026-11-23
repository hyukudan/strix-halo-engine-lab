#if defined(STRIX_INKLING2_APP_CLI_TEST)
#include "inkling2_mini_cli.hpp"
#endif
#include "strix/session/cli_output.hpp"
#include "strix/session/head_gate_cli.hpp"

#include <fstream>
#include <iostream>

#include <array>
#include <bit>
#include <cstdint>

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
cli::Options parse(std::initializer_list<std::string> arguments) {
    std::vector<std::string> storage{"mini", "model"};
    storage.insert(storage.end(), arguments);
    std::vector<char*> argv;
    for (auto& value : storage) argv.push_back(value.data());
    const strix::serve::SerialGenerationFamily family{"mini", "Mini", "MODEL",
        8U, 8U, 16U, 0U, nullptr, 8U, true, false,
        [](const std::string&) -> std::shared_ptr<const strix::serve::BpeTokenizer> { return {}; },
        true, [](std::string_view) { return std::string{}; }, true};
    auto options = cli::parse(static_cast<int>(argv.size()), argv.data(), family);
    // Only use fields that own their values after the temporary argument storage expires.
    return options;
}

// The block drafter a verifying route loads: mtp.safetensors beside the model only under --speculation auto,
// --dspark FILE wins, --dspark off loads none; --dspark with temperature only when the family verifies sampled rows.
void test_block_drafter(const fs::path& directory) {
    const auto checkpoint = (directory / "model.safetensors").string();
    const auto beside = directory / "mtp.safetensors";
    std::ofstream(beside).put('x');
    std::vector<std::string> storage;
    std::vector<char*> argv;
    const auto run = [&](std::initializer_list<std::string> arguments, bool sampled) {
        storage = {"mini", "model"};
        storage.insert(storage.end(), arguments);
        argv.clear();
        for (auto& value : storage) argv.push_back(value.data());
        strix::serve::SerialGenerationFamily family{"mini", "Mini", "MODEL",
            8U, 8U, 16U, 0U, nullptr, 8U, true, false,
            [](const std::string&) -> std::shared_ptr<const strix::serve::BpeTokenizer> { return {}; },
            true, [](std::string_view) { return std::string{}; }, true};
        family.sampled_speculation = sampled;
        return cli::parse(static_cast<int>(argv.size()), argv.data(), family);
    };
    auto options = run({"1", "--generate", "4", "--speculation", "auto"}, false);
    require(cli::resolve_block_drafter(options, checkpoint) == beside.string(), "auto loads the drafter beside");
    require(cli::block_drafter_source(options, beside.string()) == "beside-model", "beside-model source");
    options = run({"1", "--generate", "4"}, false);
    require(cli::resolve_block_drafter(options, checkpoint).empty(), "speculation off loads no drafter");
    options = run({"1", "--generate", "4", "--speculation", "auto", "--dspark", "off"}, false);
    require(options.dspark_off && !options.dspark, "--dspark off parses");
    require(cli::resolve_block_drafter(options, checkpoint).empty(), "--dspark off loads no drafter");
    require(cli::block_drafter_source(options, {}) == "off", "off source");
    options = run({"1", "--generate", "4", "--speculation", "auto", "--dspark", "other.safetensors"}, false);
    require(cli::resolve_block_drafter(options, checkpoint) == "other.safetensors", "explicit drafter wins");
    require(cli::block_drafter_source(options, "other.safetensors") == "explicit", "explicit source");
    expect_invalid([&] {
        (void)run({"1", "--generate", "4", "--speculation", "auto", "--dspark", "d", "--temperature", "0.7"}, false);
    });
    options = run({"1", "--generate", "4", "--speculation", "auto", "--dspark", "d", "--temperature", "0.7"}, true);
    require(options.dspark != nullptr, "sampled --dspark is accepted when the family verifies sampled rows");
    expect_invalid([&] { (void)run({"1", "--generate", "4", "--dspark", "d"}, true); });
    std::error_code error;
    fs::remove(beside, error);
    require(cli::resolve_block_drafter(run({"1", "--generate", "4", "--speculation", "auto"}, false), checkpoint)
                .empty(), "absent drafter file loads none");
}

void test_cuda_route_options() {
    using Mode = strix::serve::CliSpeculation;
    constexpr auto bf16 = strix::weight::kBf16WeightStorage;
    constexpr auto q8 = strix::weight::kQ8WeightStorage;
    auto options = parse({"1", "--generate", "4"});
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
    require(resolve().program, "auto changed the default executor");
    require(resolve(bf16, true).program && resolve(bf16, true).speculation == Mode::off,
            "default native K1 preparation changed program selection or verifier receipt");
    expect_invalid([&] { (void)resolve(q8, true); });
    expect_invalid([&] { (void)resolve(q8); });

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
    expect_invalid([&] { (void)resolve(); });
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
    expect_invalid([&] { (void)forms({"--latency-forms", "--no-latency-forms"}); });
}

void test_options() {
    using strix::profile::Name;
    require(parse({"1"}).tier.profile == Name::standard, "expected exact default");
    require(parse({"1", "--profile", "exact"}).tier.profile == Name::standard, "expected explicit exact");
    require(parse({"1", "--profile", "fast"}).tier.profile == Name::fast, "expected FAST");
    require(parse({"1", "--profile", "turbo"}).tier.profile == Name::turbo, "expected TURBO");
    require(parse({"1", "--profile", "fast", "--qualify-q8"}).weights.qualify_q8, "expected Q8 qualification arm");
    require(parse({"1", "--profile", "fast", "--no-weight-q8"}).weights.disable_q8, "expected Q8 off switch");
    expect_invalid([] { (void)parse({"1", "--qualify-q8"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "turbo", "--qualify-q8"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fast", "--no-weight-q8", "--qualify-q8"}); });
    expect_invalid([] { (void)parse({"--tokenize", "input.txt", "--profile", "fast", "--qualify-q8"}); });
    require(parse({"1"}).generation.packed_decode, "expected packed default");
    require(!parse({"1", "--grouped-decode"}).generation.packed_decode, "expected grouped override");
    expect_invalid([] { (void)parse({"1", "--packed-decode", "--grouped-decode"}); });
    expect_invalid([] { (void)parse({"--messages", "chat.json", "--tokenize", "input.txt"}); });
    expect_invalid([] { (void)parse({"--tokenize", "input.txt", "--generate", "0"}); });
    expect_invalid([] { (void)parse({"--gate-rows", "64"}); });
    expect_invalid([] { (void)parse({"1", "--gate-order-null"}); });
    expect_invalid([] { (void)parse({"1", "--profile", "fake"}); });
    require(parse({"1", "--speculation", "auto", "--dspark", "mtp.safetensors"}).generation.speculation ==
                strix::serve::CliSpeculation::automatic,
            "expected DSpark in automatic speculation");
    expect_invalid([] { (void)parse({"1", "--speculation", "off", "--dspark", "mtp.safetensors"}); });
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
    require(profile_line.str() == "profile exact levers none reference BASE\n", "device profile receipt differs");
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

int main() {
    try {
        const TempDir directory;
#if defined(STRIX_INKLING2_APP_CLI_TEST)
        test_options();
    test_cuda_route_options();
        test_block_drafter(directory.path);
#endif
        test_windows(directory.path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

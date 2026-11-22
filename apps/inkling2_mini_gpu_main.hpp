#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "inkling2/inkling2_mini_program_cuda.hpp"
#endif
#include "core/resident_row_plan.hpp"
#include "inkling2/inkling2_mini_resident_dspark.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/session/cli_output.hpp"
#include "strix/spec/engine_target.hpp"
#if defined(__HIPCC__)
#include "strix/device/hip_enumerator.hpp"
#include <hip/hip_version.h>
#else
#include "strix/device/cuda_enumerator.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

namespace strix::inkling2::cli {

// Prefill chunk rows. The logits are chunk-invariant (same bits for every chunk), so this is a schedule choice only.
// gfx1151 (prose prompts, one chunk size per run, 2048 -> 8192 rows): 4K 3.47 -> 3.13 s, 8K 7.27 -> 6.23 s,
// 16K 15.69 -> 14.06 s, 32K 35.6 -> 35.8 s. CUDA keeps 2048 rows until it is measured there.
#if defined(__HIPCC__)
inline constexpr unsigned kMiniDefaultPrefillChunk = 8192U;
#else
inline constexpr unsigned kMiniDefaultPrefillChunk = 2048U;
#endif

class DeviceCliDecoder final : public serve::SerialGenerationDecoder {
public:
    DeviceCliDecoder(const serve::SerialGenerationLoad& load, bool step_graphs, weight::FormatId storage,
                     const char* dspark)
        : decoder_(load.checkpoint, load.positions, load.prefill_chunk, load.packed_decode, step_graphs, storage,
                   load.verify_rows, dspark != nullptr) {
        if (dspark) drafter_.emplace(decoder_, load.positions, dspark);
    }

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
        return decoder_.prefill(ids, logits);
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return decoder_.decode(token, logits);
    }

    spec::VerifyTarget* verify_target() override { return &target_; }

    std::uint32_t max_verify_rows(serve::CliSpeculation) const override {
        return decoder_.max_verify_rows();
    }

    spec::ChainDrafter* chain_drafter(std::uint32_t anchor, std::uint32_t position) override {
        if (!drafter_) return nullptr;
        drafter_->start(anchor, position);
        return &*drafter_;
    }

private:
    MiniResidentDecoder decoder_;
    spec::EngineVerifyTarget<MiniResidentDecoder> target_{decoder_};
    std::optional<MiniResidentDSpark> drafter_;
};

#if defined(STRIX_INKLING2_STEP_PROGRAM)
// Leave option payloads in place for the shared serial and gate parsers.
inline bool extract_step_program(std::vector<char*>& args) {
    constexpr std::string_view value_options[]{
        "--generate", "--logits", "--temperature", "--top-k", "--seed", "--prompt", "--messages",
        "--tokenize", "--tokenizer", "--prefill-chunk", "--profile", "--gate-manifest", "--gate-out",
        "--gate-arm", "--gate-rows", "--gate-mode",
    };
    bool requested = false;
    auto output = args.begin() + 2;
    for (auto arg = output; arg != args.end(); ++arg) {
        const std::string_view value(*arg);
        if (value == "--step-program") {
            if (requested) throw std::invalid_argument("duplicate --step-program");
            requested = true;
            continue;
        }
        *output++ = *arg;
        if (std::ranges::find(value_options, value) != std::end(value_options) && arg + 1 != args.end())
            *output++ = *++arg;
    }
    args.erase(output, args.end());
    return requested;
}
#endif

inline void run_gate(const char* checkpoint, const Options& options, const weight::WeightLoadPlan& weights
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     , bool step_program
#endif
) {
    namespace fs = std::filesystem;
    const auto mode = *options.gate_mode;
    const auto windows = session::read_head_gate_manifest(options.gate_manifest, mode, options.gate_rows,
                                                          kMaxContext, kOutputVocabulary);
    const auto route_config = moe::route_tap_config_from_env();
    if (route_config.null_on)
        throw std::invalid_argument("use --gate-order-null for the Inkling evaluation-order null");
    constexpr std::string_view case_marker = "{case}";
    const bool per_window_routes = route_config.in_path.find(case_marker) != std::string::npos ||
                                   route_config.out_path.find(case_marker) != std::string::npos;
    if (per_window_routes &&
        ((!route_config.in_path.empty() && route_config.in_path.find(case_marker) == std::string::npos) ||
         (!route_config.out_path.empty() && route_config.out_path.find(case_marker) == std::string::npos)))
        throw std::invalid_argument("use {case} in every route path for per-window evidence");
    const auto expand = [&](std::string path, const std::string& name) {
        for (auto at = path.find(case_marker); at != std::string::npos; at = path.find(case_marker, at + name.size()))
            path.replace(at, case_marker.size(), name);
        return path;
    };

    const fs::path out(options.gate_out);
    fs::create_directories(out);
    fs::create_directories(out / "log");
    unsigned positions = 0;
    for (const auto& window : windows) positions = std::max(positions, static_cast<unsigned>(window.ids.size()));
    const auto prefix_positions = mode == session::HeadGateMode::decode ? positions - options.gate_rows : positions;
    const auto max_rows = std::min(options.generation.prefill_chunk, prefix_positions);
    const auto load_start = session::CliClock::now();
    MiniResidentDecoder decoder(checkpoint, positions, max_rows, options.generation.packed_decode,
                                 options.step_graphs.value_or(MiniResidentBackend::default_step_graphs), weights.storage);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    std::optional<MiniCudaProgramDecoder> program;
    if (step_program) program.emplace(decoder);
#endif
    session::write_load_timing(std::cout, load_start);
    std::vector<float> logits(std::size_t{options.gate_rows} * kOutputVocabulary);
    for (const auto& window : windows) {
        const auto dump = out / (window.name + "-" + options.gate_arm + ".f32");
        const auto ids_dump = out / (window.name + "-" + options.gate_arm + ".ids");
        const auto log = out / "log" / (window.name + "." + options.gate_arm + ".stderr");
        if (fs::exists(dump) || fs::exists(ids_dump) || fs::exists(log))
            throw std::runtime_error("gate output already exists for " + window.name);
        std::optional<moe::RouteTap> tap;
        if (per_window_routes) {
            auto config = route_config;
            config.in_path = expand(config.in_path, window.name);
            config.out_path = expand(config.out_path, window.name);
            if (!config.out_path.empty() && fs::exists(config.out_path))
                throw std::runtime_error("gate route output already exists: " + config.out_path);
            tap.emplace(std::move(config));
        }
        decoder.set_gate_order_null(options.gate_order_null);
        decoder.set_gate_route_tap(tap ? &*tap : nullptr);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        if (program)
            session::run_head_gate_window(*program, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        else
#endif
        session::run_head_gate_window(decoder, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        decoder.set_gate_route_tap(nullptr);
        if (!std::ranges::all_of(logits, [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("non-finite gate logits for " + window.name);
        session::write_logits(dump, logits);
        std::ofstream ids(ids_dump);
        for (unsigned row = 0; row < options.gate_rows; ++row) {
            const auto scores = std::span(logits).subspan(std::size_t{row} * kOutputVocabulary, kOutputVocabulary);
            ids << std::ranges::max_element(scores) - scores.begin() << '\n';
        }
        ids.close();
        if (!ids) throw std::runtime_error("could not write gate token IDs: " + ids_dump.string());

        auto routes = tap ? tap->receipt() : std::string{"{\"version\":1}"};
        if (options.gate_order_null) {
            routes.pop_back();
            routes += ",\"null\":{\"order\":\"relative-bias-reverse-fma@1\",\"applications\":" +
                      std::to_string(decoder.gate_null_applications()) + "}}";
        }
        std::ofstream receipt(log);
        cli::write_weight_load_plan(receipt, weights);
        receipt << "strix-route-file/v1 " << routes << '\n';
        receipt.close();
        if (!receipt) throw std::runtime_error("could not write gate route receipt: " + log.string());
    }
}

// The scope a load-time weight decision row is matched against: the current device's architecture, the toolkit
// this binary was built with, the resident regime, the resident Q8_0 consumers' implementation identity and the raw
// checkpoint SHA-256 (shared persistent stat-keyed cache: only the first FAST run of a checkpoint reads it whole).
// EXACT, or FAST with --no-weight-q8, never hashes: the scope stays empty and no row can match.
struct MiniWeightScope {
    std::string device_class, compiler_id;
    weight::WeightLoadScope scope{};
};

inline MiniWeightScope mini_weight_scope(const std::string& model, weight::Backend backend, bool needed) {
    MiniWeightScope out;
    out.scope.backend = backend;
    if (!needed) return out;
    int ordinal = 0;
#if defined(__HIPCC__)
    if (hipGetDevice(&ordinal) != hipSuccess) return out;
    const auto visible = device::HipEnumerator{}.enumerate();
    out.compiler_id = "hip-" + std::to_string(HIP_VERSION_MAJOR) + "." + std::to_string(HIP_VERSION_MINOR);
    const auto identity = weight::implementation_identity(core::kResidentQ8ArithmeticHip);
#else
    if (cudaGetDevice(&ordinal) != cudaSuccess) return out;
    const auto visible = device::CudaEnumerator{}.enumerate();
    out.compiler_id = "cuda-" + std::to_string(CUDART_VERSION / 1000) + "." + std::to_string(CUDART_VERSION % 1000 / 10);
    const auto identity = weight::implementation_identity(core::kResidentQ8ArithmeticCuda);
#endif
    for (const auto& device : visible)
        if (device.ordinal == ordinal) out.device_class = device.arch;
    out.scope.device_class = out.device_class;
    out.scope.compiler_id = out.compiler_id;
    out.scope.regime = kMiniResidentRegime;
    out.scope.build = identity;
    out.scope.checkpoint = weight::Digest256{sha256_file_content(model).bytes};
    return out;
}

}  // namespace strix::inkling2::cli

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace serve = strix::serve;
#if defined(__HIPCC__)
    constexpr auto backend = "HIP";
    constexpr auto executable = "strix-inkling2-mini-hip";
    constexpr auto weight_backend = strix::weight::Backend::hip;
    // Single-token decode runs the experts on the one-row packed GEMV (gfx1151, 128 tokens:
    // 19.7 tok/s against 16.0 tok/s on the grouped tile kernel); --grouped-decode keeps the tile kernel.
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
    constexpr auto weight_backend = strix::weight::Backend::cuda;
    // RTX PRO 6000, ink_cuda_opt p3 (128 tokens): with graphs, R1 ran at 58.1 vs 43.4 tok/s short and 45.2 vs 36.6 at 4K. Eager short was 44.3 vs 34.1.
#endif
    constexpr bool packed_decode_default = true;
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT | --messages FILE) [--tokenizer FILE]"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]"
                     " [--speculation off|auto|verify|lookup] [--dspark FILE]"
                     " [--step-graph | --no-step-graph"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " | --step-program"
#endif
                     "] [--temperature T] [--top-k K] [--seed N]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--gate-order-null] [--prefill-chunk N]"
                     " [--packed-decode | --grouped-decode]\n"
                     "       MODEL.safetensors --tokenize FILE [--tokenizer FILE]\n"
                     "       --profile exact|fast|turbo (default exact) [--qualify-q8 | --no-weight-q8]\n";
        return 2;
    }
    try {
        serve::SerialGenerationFamily family{
            executable, "Inkling 2 Mini", "MODEL.safetensors",
            kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
            {},
            strix::inkling2::cli::kMiniDefaultPrefillChunk, true, true,
            [](const std::string& path) {
                return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
            },
            packed_decode_default, serve::render_inkling2_chat_request, true};
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        std::vector<char*> arguments(argv, argv + argc);
        const bool step_program = cli::extract_step_program(arguments);
        auto options = cli::parse(static_cast<int>(arguments.size()), arguments.data(), family);
        if (step_program) {
            if (options.step_graphs)
                throw std::invalid_argument("--step-program is exclusive with --step-graph and --no-step-graph");
            if (!options.generation.packed_decode)
                throw std::invalid_argument("--step-program requires packed decode");
            if (options.gate_manifest ? options.gate_mode != strix::session::HeadGateMode::decode
                                      : options.generation.generate < 2U)
                throw std::invalid_argument("--step-program requires generation of at least two tokens or a decode gate");
            if (strix::moe::route_tap_config_from_env().active())
                throw std::invalid_argument("--step-program requires route tap and replay to be disabled");
            family.load = load_mini_cuda_program;
        }
#else
        auto options = cli::parse(argc, argv, family);
#endif
        constexpr std::array consumers{strix::weight::kBf16WeightStorage, strix::weight::kQ8WeightStorage};
        const bool compact_tier = options.tier.profile == strix::profile::Name::fast ||
                                  options.tier.profile == strix::profile::Name::turbo;
        const auto scope = cli::mini_weight_scope(argv[1], weight_backend, compact_tier && !options.disable_q8);
        const auto weights = resolve_mini_weight_load(options.tier.profile, scope.scope,
            consumers, !options.disable_q8, options.qualify_q8);
        if (options.qualify_q8 && weights.storage != strix::weight::kQ8WeightStorage)
            throw std::invalid_argument("native Q8_0 weights were not admitted");
        const auto step_graphs = options.step_graphs.value_or(MiniResidentBackend::default_step_graphs);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        if (step_program && options.dspark)
            throw std::invalid_argument("--step-program is exclusive with --dspark");
        if (!step_program)
#endif
        family.load = [step_graphs, storage = weights.storage, dspark = options.dspark](const serve::SerialGenerationLoad& load) {
            return std::make_unique<cli::DeviceCliDecoder>(load, step_graphs, storage, dspark);
        };
        const auto write_profile = [&] {
            cli::write_profile(options.tier, std::cerr, true);
            cli::write_weight_load_plan(std::cerr, weights);
            std::cerr << "strix-weight-load-status/v1 " << strix::profile::status_name(weights.status) << " ("
                      << weights.reason << ") device " << (scope.device_class.empty() ? "-" : scope.device_class)
                      << " compiler " << (scope.compiler_id.empty() ? "-" : scope.compiler_id) << " checkpoint ";
            if (scope.scope.checkpoint == strix::weight::Digest256{}) std::cerr << '-';
            for (std::size_t i = 0; i < 8 && !(scope.scope.checkpoint == strix::weight::Digest256{}); ++i)
                std::cerr << "0123456789abcdef"[scope.scope.checkpoint.bytes[i] >> 4U]
                          << "0123456789abcdef"[scope.scope.checkpoint.bytes[i] & 15U];
            std::cerr << '\n';
        };
        if (options.gate_manifest) {
            write_profile();
            cli::run_gate(argv[1], options, weights
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                          , step_program
#endif
            );
            return 0;
        }
        if (options.generation.tokenize_file) cli::write_profile(options.tier, std::cerr, true);
        serve::run_serial_generation(argv[1], std::move(options.generation), family, std::cout, write_profile);
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini " << backend << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}

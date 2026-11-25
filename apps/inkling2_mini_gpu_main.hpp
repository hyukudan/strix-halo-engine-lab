#pragma once

#include "inkling2_mini_cli.hpp"
#if defined(STRIX_INKLING2_MEDIA)
#include "inkling2_mini_media.hpp"
#endif

#include "inkling2/inkling2_mini_prefill_rows.hpp"
#include "inkling2/inkling2_mini_resident_decoder.hpp"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
#include "inkling2/inkling2_mini_program_cuda.hpp"
#endif
#include "inkling2/inkling2_mini_resident_dspark.hpp"
#include "inkling2/inkling2_mini_weight_device.hpp"
#include "strix/env_flags.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/session/cli_output.hpp"
#include "strix/session/embed_splice.hpp"
#include "strix/spec/engine_target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>

namespace strix::inkling2::cli {

// Prefill chunk rows and tail rule: the one schedule of the command line and the server (mini_prefill_schedule in
// src/inkling2/inkling2_mini_schedule.hpp, with the STRIX_INKLING_PREFILL_TAIL switch). Its chunk is the declared rows
// per backend and profile (inkling2::mini_cli_prefill_rows); the profile's own value is applied after parsing.
#if defined(__HIPCC__)
inline constexpr bool kMiniHipSchedule = true;
#else
inline constexpr bool kMiniHipSchedule = false;
#endif

class DeviceCliDecoder final : public serve::SerialGenerationDecoder {
public:
    DeviceCliDecoder(const serve::SerialGenerationLoad& load, bool step_graphs, weight::FormatId storage,
                     const char* dspark,
                     moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16,
                     bool combine_image = false, bool prepared_round_fold = false, bool narrow_experts = false,
                     bool controller_one_plane = false, bool attn_projected_bias = false)
        : decoder_(load.checkpoint, load.positions, load.prefill_chunk, load.packed_decode, step_graphs, storage,
                   load.verify_rows, dspark != nullptr, 1U, load.prefill_tail, narrow_experts, controller_one_plane),
          projected_bias_(attn_projected_bias) {
        const auto levers = configure_mini_decoder(decoder_, moe_w4a4, combine_image, prepared_round_fold, attn_projected_bias);
        std::cerr << "Inkling down_image " << mini_down_image_name(levers.down_image, combine_image) << '\n';
        std::cerr << "Inkling prepared_round_fold " << mini_round_fold_name(levers.prepared_round_fold, prepared_round_fold)
                  << '\n';
        if (dspark) drafter_.emplace(decoder_, load.positions, dspark);
    }

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
        const auto next = decoder_.prefill(ids, logits);
        // With the lever selected, the observed route: projected-bias appends submitted so far (0: every chunk declined
        // or was not admissible, so the image route ran).
        if (projected_bias_)
            std::cerr << "Inkling attn_projected_bias dispatches " << decoder_.projected_attention_dispatches() << '\n';
        return next;
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return decoder_.decode(token, logits);
    }
    unsigned prefill_embedded(std::span<const unsigned> ids, const std::shared_ptr<const session::EmbedTable>& table,
                              std::span<float> logits) override {
        if (!table) return decoder_.prefill(ids, logits);
        const auto run = session::resolve_embed_run(ids, table, {kHiddenWidth});
        return decoder_.prefill_spliced(run.tokens, run.splice, logits);
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
    bool projected_bias_ = false;
};

inline void run_gate(const char* checkpoint, const Options& options, std::span<const session::HeadGateWindow> windows,
                     unsigned positions, const weight::WeightLoadPlan& weights
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     , bool step_program, bool epilogue_fusion, unsigned latency_forms
#endif
) {
    namespace fs = std::filesystem;
    const auto mode = *options.gate_mode;
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
    const auto prefix_positions = mode == session::HeadGateMode::decode ? positions - options.gate_rows : positions;
    const auto max_rows = std::min(options.generation.prefill_chunk, prefix_positions);
    const auto load_start = session::CliClock::now();
    MiniResidentDecoder decoder(checkpoint, positions, max_rows, options.generation.packed_decode,
                                 options.step_graphs.value_or(MiniResidentBackend::default_step_graphs), weights.storage,
                                 0U, false, 1U, 0U, mini_experts_nvfp4(options.tier),
                                 mini_controller_one_plane(options.tier));
    const auto levers = configure_mini_decoder(decoder, options.tier);
    std::cerr << "Inkling down_image " << mini_down_image_name(levers.down_image, mini_combine_image(options.tier)) << '\n';
    std::cerr << "Inkling prepared_round_fold "
              << mini_round_fold_name(levers.prepared_round_fold, mini_prepared_round_fold(options.tier)) << '\n';
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    std::optional<MiniCudaProgramDecoder> program;
    if (step_program) program.emplace(decoder, epilogue_fusion, mini_latency_forms(latency_forms));
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
        const auto set_tap = [&](moe::RouteTap* route_tap) {
#if defined(STRIX_INKLING2_STEP_PROGRAM)
            if (program) {
                program->set_gate_route_tap(route_tap);
                return;
            }
#endif
            decoder.set_gate_route_tap(route_tap);
        };
        set_tap(tap ? &*tap : nullptr);
        // The null arm reverses the relative-bias order on the resident route (the CLI refuses the
        // step program with it); resetting per window makes the receipt count this window only.
        if (options.gate_order_null) decoder.set_gate_order_null(true);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        if (program)
            session::run_head_gate_window(*program, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        else
#endif
        session::run_head_gate_window(decoder, window.ids, mode, options.gate_rows, kOutputVocabulary, logits);
        set_tap(nullptr);
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
        if (options.gate_order_null)
            routes = cli::gate_null_receipt(std::move(routes), decoder.gate_null_applications(), window.name);
        std::ofstream receipt(log);
        cli::write_weight_load_plan(receipt, weights);
        receipt << "strix-route-file/v1 " << routes << '\n';
        receipt.close();
        if (!receipt) throw std::runtime_error("could not write gate route receipt: " + log.string());
    }
}

#if defined(STRIX_INKLING2_STEP_PROGRAM)
static_assert(kLatencyFormsMix == kMiniFormsMix && kLatencyFormsMoe == kMiniFormsMoe &&
              kLatencyFormsAll == kMiniFormsAll && kMiniDefaultForms == kMiniFormsMix);

using MiniCudaProgramLoad = std::unique_ptr<serve::SerialGenerationDecoder> (*)(const serve::SerialGenerationLoad&,
                                                                                moe_core::grouped_w4a4::Activations, bool, bool);

// The exact R1 latency forms are a template parameter of the loader; map the CLI mask to its instance.
inline MiniCudaProgramLoad select_mini_cuda_program(bool fuse, unsigned forms) {
    const auto pick = [forms](auto fused) -> MiniCudaProgramLoad {
        constexpr bool on = decltype(fused)::value;
        switch (forms) {
        case kMiniFormsMix: return load_mini_cuda_program<on, kMiniFormsMix>;
        case kMiniFormsMoe: return load_mini_cuda_program<on, kMiniFormsMoe>;
        case kMiniFormsAll: return load_mini_cuda_program<on, kMiniFormsAll>;
        default: return load_mini_cuda_program<on, 0U>;
        }
    };
    return fuse ? pick(std::true_type{}) : pick(std::false_type{});
}
#endif

}  // namespace strix::inkling2::cli

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace serve = strix::serve;
#if defined(__HIPCC__)
    constexpr auto backend = "HIP";
    constexpr auto executable = "strix-inkling2-mini-hip";
    // Single-token decode runs the experts on the one-row packed GEMV (gfx1151, 128 tokens:
    // 19.7 tok/s against 16.0 tok/s on the grouped tile kernel); --grouped-decode keeps the tile kernel.
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
    // RTX PRO 6000, ink_cuda_opt p3 (128 tokens): with graphs, R1 ran at 58.1 vs 43.4 tok/s short and 45.2 vs 36.6 at 4K. Eager short was 44.3 vs 34.1.
#endif
    // The one decode default of every entry point (strix-serve resolves the same flags to the same form).
    constexpr bool packed_decode_default = kMiniPackedDecodeDefault;
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT | --messages FILE) [--tokenizer FILE]"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]"
                     " [--speculation off|auto|verify|lookup] [--dspark FILE]"
                     " [--step-graph | --no-step-graph"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " | --step-program (default for decode)"
#endif
                     "]"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " [--no-epilogue-fusion] [--latency-forms[=mix|=moe] | --no-latency-forms]"
#endif
                     " [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--prefill-chunk N] [--packed-decode | --grouped-decode]"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " [--no-epilogue-fusion] [--latency-forms[=mix|=moe] | --no-latency-forms]"
#endif
                     "\n"
                     "       MODEL.safetensors --tokenize FILE [--tokenizer FILE]\n"
                     "       --profile exact|fast|turbo (default exact) [--qualify-q8 | --no-weight-q8]\n"
                     "       --controller-planes 3|1 (default 3; 1 is a CUDA FAST/TURBO candidate)\n";
        return 2;
    }
    try {
        serve::SerialGenerationFamily family{
            executable, "Inkling 2 Mini", "MODEL.safetensors",
            kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
            {},
            // The parse default; the profile's own value is applied after parsing (mini_cli_prefill_rows).
            mini_prefill_schedule(strix::inkling2::cli::kMiniHipSchedule, strix::profile::Name::standard).chunk, true, true,
            [](const std::string& path) {
                return serve::BpeTokenizer::load_hf_json(path, serve::inkling2_mini_tokenizer_identity);
            },
            packed_decode_default, serve::render_inkling2_chat_request, true};
        // The served schedule's resolver (STRIX_INKLING_PREFILL_TAIL=0 restores greedy chunks on both entry points).
        family.prefill_tail_divisor =
            mini_prefill_schedule_from_env(strix::inkling2::cli::kMiniHipSchedule, strix::profile::Name::standard).tail_divisor;
#if defined(STRIX_INKLING2_MEDIA)
        family.media = cli::open_mini_media;
        family.render_media_messages = serve::render_inkling2_chat_media_request;
#endif
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        std::vector<char*> arguments(argv, argv + argc);
        const auto program_options = cli::extract_step_program(arguments);
        auto options = cli::parse(static_cast<int>(arguments.size()), arguments.data(), family, kMiniBuildBackend);
#else
        auto options = cli::parse(argc, argv, family, kMiniBuildBackend);
#endif
        if (!options.generation.prefill_chunk_explicit) {  // the declared chunk of this backend and profile
            family.default_prefill_chunk = mini_prefill_schedule(cli::kMiniHipSchedule, options.tier.profile).chunk;
            options.generation.prefill_chunk = family.default_prefill_chunk;
        }
        // The serial CLI resolves its one lane at load using prompt + generated positions. A gate knows its
        // resident context from the manifest. CUDA step routing precedes serial loading and has no banded CUDA row.
        struct ResolvedWeights {
            MiniWeightScope scope;
            strix::weight::WeightLoadPlan plan;
        };
        const auto resolve_weights = [&options, model = std::string(argv[1])](strix::weight::WeightLoadExtent extent) {
            auto scope = mini_weight_scope(model, options.tier.profile, options.weights, extent);
            const auto plan = resolve_mini_resident_weight_load(options.tier.profile, options.weights, scope.view(), extent);
            return ResolvedWeights{std::move(scope), plan};
        };
        const auto gate_windows = options.gate_manifest
            ? strix::session::read_head_gate_manifest(options.gate_manifest, *options.gate_mode, options.gate_rows,
                                                      kMaxContext, kOutputVocabulary)
            : std::vector<strix::session::HeadGateWindow>{};
        const auto gate_extent = cli::gate_weight_extent(gate_windows);
        const auto early = resolve_weights(options.gate_manifest ? gate_extent : strix::weight::WeightLoadExtent{});
        std::optional<ResolvedWeights> loaded;
        const auto& weights = early.plan;
        const auto step_graphs = options.step_graphs.value_or(MiniResidentBackend::default_step_graphs);
        const auto moe_w4a4 = options.moe_w4a4;
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        const char* grouped_prepare = strix::envcfg::text("STRIX_INKLING_K1BIND");
        const bool resident_k1 = grouped_prepare && std::string_view(grouped_prepare) == "1";
        const auto route = cli::resolve_cuda_step_route(options, program_options, weights.storage, resident_k1);
        const bool step_program = route.program;
        options.generation.speculation = route.speculation;
        if (step_program)
            family.load = [program_load = cli::select_mini_cuda_program(program_options.epilogue_fusion,
                                                                        program_options.latency_forms),
                           moe_w4a4,
                           controller_one_plane = mini_controller_one_plane(options.tier),
                           projected_bias = mini_attn_projected_bias(options.tier)](
                              const serve::SerialGenerationLoad& load) {
                return program_load(load, moe_w4a4, controller_one_plane, projected_bias);
            };
        else
#endif
        family.load = [&loaded, &resolve_weights, step_graphs, dspark = options.dspark,
                       moe_w4a4,
                       combine_image = mini_combine_image(options.tier),
                       round_fold = mini_prepared_round_fold(options.tier),
                       narrow = mini_experts_nvfp4(options.tier),
                       controller_one_plane = mini_controller_one_plane(options.tier),
                       projected_bias = mini_attn_projected_bias(options.tier)](const serve::SerialGenerationLoad& load) {
            loaded = resolve_weights({load.positions, 1U});
            return std::make_unique<cli::DeviceCliDecoder>(load, step_graphs, loaded->plan.storage, dspark, moe_w4a4,
                                                           combine_image, round_fold, narrow, controller_one_plane,
                                                           projected_bias);
        };
        const auto write_profile = [&] {
            const auto& resolved = loaded ? *loaded : early;
            const auto& plan = resolved.plan;
            const auto& scope = resolved.scope;
            cli::write_profile(options.tier, std::cerr, true);
            cli::write_weight_load_plan(std::cerr, plan);
            std::cerr << strix::weight::weight_load_regime_receipt(plan) << '\n';
            // The admitted W4A4 class, the same identity the gate prints and the server reports as moe_prefill.
            if (mini_moe_w4a4(options.tier))
                std::cerr << "strix-moe-w4a4 class " << strix::moe_core::grouped_w4a4::identity(moe_w4a4) << '\n';
            std::cerr << "strix-prefill-route/v1 chunk " << options.generation.prefill_chunk << " tail_divisor "
                      << family.prefill_tail_divisor << " first_id prefill\n";
            std::cerr << "strix-decode-route/v1 moe_decode " << mini_decode_form_name(options.generation.packed_decode)
                      << '\n';
            std::cerr << mini_served_route_cli_line(options.tier) << '\n';
#if defined(STRIX_INKLING2_STEP_PROGRAM)
            if (!route.speculation_reason.empty())
                std::cerr << "strix-speculation-route/v1 requested auto effective off ("
                          << route.speculation_reason << ")\n";
#endif
            std::cerr << "strix-weight-load-status/v1 " << strix::profile::status_name(plan.status) << " ("
                      << plan.reason << ") device " << (scope.device_class.empty() ? "-" : scope.device_class)
                      << " compiler " << (scope.compiler_id.empty() ? "-" : scope.compiler_id) << " checkpoint ";
            if (scope.checkpoint == strix::weight::Digest256{}) std::cerr << '-';
            for (std::size_t i = 0; i < 8 && !(scope.checkpoint == strix::weight::Digest256{}); ++i)
                std::cerr << "0123456789abcdef"[scope.checkpoint.bytes[i] >> 4U]
                          << "0123456789abcdef"[scope.checkpoint.bytes[i] & 15U];
            std::cerr << '\n';
        };
        if (options.gate_manifest) {
            write_profile();
            cli::run_gate(argv[1], options, gate_windows, gate_extent.context, weights
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                          , step_program, program_options.epilogue_fusion, program_options.latency_forms
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

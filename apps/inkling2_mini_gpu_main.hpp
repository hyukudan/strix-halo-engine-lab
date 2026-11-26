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
#include "strix/spec/drafter_source.hpp"
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
    DeviceCliDecoder(const serve::SerialGenerationLoad& load, bool step_graphs, weight::RoleStorage storage,
                     const char* dspark,
                     moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16,
                     bool combine_image = false, bool prepared_round_fold = false, bool narrow_experts = false,
                     bool controller_one_plane = false, bool attn_projected_bias = false, bool screened_head = false,
                     unsigned verify_rows = 0U, MiniLowBitRows lowbit = {}, bool plain_program = false)
        : decoder_(load.checkpoint, load.positions, load.prefill_chunk, load.packed_decode, step_graphs, storage,
                   load.verify_rows, dspark != nullptr, 1U, load.prefill_tail, narrow_experts, controller_one_plane, false, lowbit),
          projected_bias_(attn_projected_bias) {
        const auto levers = configure_mini_decoder(decoder_, moe_w4a4, combine_image, prepared_round_fold,
                                                   attn_projected_bias, screened_head, verify_rows);
        require_mini_calibrated_cli_slot(lowbit, decoder_.slot_bytes());
        std::cerr << "Inkling down_image " << mini_down_image_name(levers.down_image, combine_image) << '\n';
        std::cerr << "Inkling prepared_round_fold " << mini_round_fold_name(levers.prepared_round_fold, prepared_round_fold)
                  << '\n';
        std::cerr << "Inkling verify_rows " << mini_verify_rows_name(levers.verify_rows, verify_rows) << '\n';
        if (dspark) drafter_.emplace(decoder_, load.positions, dspark);
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        // The speculator's one-row steps on the CUDA step program (the program route's forms and fusion) over this
        // decoder's own state: same KV / histories / DSpark taps, so drafting passes on the resident verify windows
        // continue from it. Only where both routes run the same arithmetic (no FAST weight-image levers); else resident.
        // Default on (q54: code 132.8 tok/s, prose 0.966 x the plain step program); off switch
        // STRIX_INKLING_PLAIN_ROW_ROUTE=resident keeps every row of the verifying route on the resident decoder.
        // A verifying-route load that cannot draft (token sampling, an embeddings prompt: the shared CLI loads it with
        // speculation off) runs every decode row here too (decode() below). For an embeddings prompt this is a path the
        // program route could not run (it had no embeddings prefill); no device receipt covers it yet.
        const char* row_route = strix::envcfg::text("STRIX_INKLING_PLAIN_ROW_ROUTE");
        if (row_route && std::string_view(row_route) == "resident") plain_program = false;
        if (plain_program && !combine_image && !prepared_round_fold && !narrow_experts) {
            plain_.emplace(decoder_, true, mini_latency_forms(kMiniDefaultForms));
            decoder_.set_plain_row_route([this](unsigned token, std::span<float> logits) {
                return plain_->decode(token, logits);
            });
        }
        std::cerr << "inkling2 plain-row route=" << (plain_ ? "program" : "resident") << '\n';
#else
        (void)plain_program;
#endif
#if !defined(__HIPCC__)
        // Verify window programs (one captured graph per window rows; device identity fixture q52r PASS; timing q62: code
        // 170.8 / 169.3 tok/s vs 132.0 / 132.8 eager, ids equal): default on for the profile the fixture covers (EXACT
        // levers: projected bias, screened or full head, BF16 attention and head storage); a FAST weight-image or expert
        // lever, or Q8_0 attention / head storage, declines with the reason.
        // Off switch STRIX_INKLING_WINDOW_PROGRAMS=0. With a drafter every width is captured here, at load, so no capture
        // wall enters a timed decode (prewarm_window_programs; the line counts them, and the end line counts any later).
        const char* window = strix::envcfg::text("STRIX_INKLING_WINDOW_PROGRAMS");
        const bool asked = !(window && std::string_view(window) == "0");
        const bool proven = !combine_image && !prepared_round_fold && !narrow_experts && !controller_one_plane &&
                            moe_w4a4 == moe_core::grouped_w4a4::Activations::bf16 && attn_projected_bias &&
                            storage.attention == weight::kBf16WeightStorage &&
                            storage.head == weight::kBf16WeightStorage;
        if (asked && proven) decoder_.set_window_programs(true);
        else if (asked) std::cerr << "inkling2 verify window programs declined (unproven for this profile: the identity "
                                     "fixture covers the EXACT levers only)\n";
        std::cerr << "inkling2 verify window programs " << (decoder_.window_programs() ? "on" : "off");
        if (dspark && decoder_.window_programs()) std::cerr << " prewarmed " << decoder_.prewarm_window_programs();
        std::cerr << '\n';
#endif
        // Calibration statistics (off unless STRIX_CALIB_MOMENTS=FILE): the prompt's attention / output-projection /
        // head input rows accumulate into strix/weight/calib_moments.hpp second moments, written after each prefill.
        if (const char* path = strix::envcfg::text("STRIX_CALIB_MOMENTS"); path && *path) {
            calibration_path_ = path;
            calibration_.emplace();
            decoder_.set_calibration_moments(&*calibration_);
        }
    }

    // Receipt of the verify window programs over the run: equal to the load line's prewarmed count when no width was
    // captured inside the decode.
    ~DeviceCliDecoder() override {
#if !defined(__HIPCC__)
        if (decoder_.window_programs())
            std::cerr << "inkling2 verify window programs captured " << decoder_.window_program_captures() << '\n';
#endif
    }
    DeviceCliDecoder(const DeviceCliDecoder&) = delete;
    DeviceCliDecoder& operator=(const DeviceCliDecoder&) = delete;

    unsigned prefill(std::span<const unsigned> ids, std::span<float> logits) override {
        // Provenance: the prompt is registered as a window before its rows are captured, every captured row counts
        // against it, and the window closes with the prefill (the packer refuses windows shared with held-out sets).
        if (calibration_) decoder_.set_calibration_window(calibration_->add_window(ids));
        const auto next = decoder_.prefill(ids, logits);
        if (calibration_) {
            decoder_.set_calibration_window(std::nullopt);
            calibration_->write(calibration_path_);
            std::cerr << weight::calib::kMomentFileTag << " rows " << ids.size() << " inputs " << calibration_->inputs().size()
                      << " file " << calibration_path_ << '\n';
        }
        // With the lever selected, the observed route: projected-bias appends submitted so far (0: every chunk declined
        // or was not admissible, so the image route ran).
        if (projected_bias_)
            std::cerr << mini_projected_bias_dispatches_line(decoder_.projected_attention_dispatches()) << '\n';
        return next;
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        // The plain-row route when bound (the step program over this decoder's state), else the resident step.
        return decoder_.decode_committed_row(token, logits);
    }
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    // Sampled rows on the plain-row route: the step program's full-head row sampled on the device (decode_sampled).
    [[nodiscard]] bool samples_on_device() const override { return plain_.has_value() && decoder_.plain_row_applies(); }
    std::uint32_t decode_sampled(std::uint32_t token, TokenSampler& sampler, std::uint64_t position) override {
        return plain_->decode_sampled(token, sampler, position);
    }
#endif
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
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    std::optional<MiniCudaProgramDecoder> plain_;  // after decoder_: destroyed first (it holds the decoder)
#endif
    std::optional<weight::calib::MomentSet> calibration_;
    std::string calibration_path_;
    bool projected_bias_ = false;
};

inline void run_gate(const char* checkpoint, const Options& options, std::span<const session::HeadGateWindow> windows,
                     weight::WeightLoadExtent extent, weight::WeightLoadPlan& weights,
                     const weight::WeightLoadScope& scope, std::string_view device, std::string_view product
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
    const auto positions = extent.context;
    const auto prefix_positions = mode == session::HeadGateMode::decode ? positions - options.gate_rows : positions;
    const auto max_rows = std::min(options.generation.prefill_chunk, prefix_positions);
    const auto load_start = session::CliClock::now();
    MiniDecoderLevers levers;
    auto resident = load_mini_weights(checkpoint, weights, options.tier.profile, options.weights, scope, extent,
                                     [&](MiniLowBitRows lowbit) {
        auto decoder = std::make_unique<MiniResidentDecoder>(checkpoint, positions, max_rows, options.generation.packed_decode,
            options.step_graphs.value_or(MiniResidentBackend::default_step_graphs), weights.roles(),
            0U, false, 1U, 0U, mini_experts_nvfp4(options.tier),
            mini_controller_one_plane(options.tier), false, lowbit);
        levers = configure_mini_decoder(*decoder, options.tier);
        require_mini_calibrated_cli_slot(lowbit, decoder->slot_bytes());
        return decoder;
    });
    auto& decoder = *resident;
    std::cerr << "Inkling down_image " << mini_down_image_name(levers.down_image, mini_combine_image(options.tier)) << '\n';
    std::cerr << "Inkling prepared_round_fold "
              << mini_round_fold_name(levers.prepared_round_fold, mini_prepared_round_fold(options.tier)) << '\n';
    std::cerr << "Inkling verify_rows " << mini_verify_rows_name(levers.verify_rows, mini_verify_rows(options.tier))
              << '\n';
#if defined(STRIX_INKLING2_STEP_PROGRAM)
    std::optional<MiniCudaProgramDecoder> program;
    if (step_program) program.emplace(decoder, epilogue_fusion, mini_latency_forms(latency_forms));
#endif
    session::write_load_timing(std::cout, load_start);
    std::vector<float> logits(std::size_t{options.gate_rows} * kOutputVocabulary);
    for (const auto& window : windows) {
        if (!kMiniProfile.ladder.empty()) {
            const auto prompt_tokens = mode == session::HeadGateMode::decode ? window.ids.size() - options.gate_rows : window.ids.size();
            decoder.set_moe_w4a4(mini_prompt_moe(options.weights, device, prompt_tokens, true, nullptr,
                                                 MiniPromptExecution{product, positions, static_cast<std::uint32_t>(max_rows)}));
        }
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
        cli::write_gate_receipt(receipt, weights, routes);
        receipt.close();
        if (!receipt) throw std::runtime_error("could not write gate route receipt: " + log.string());
    }
}

#if defined(STRIX_INKLING2_STEP_PROGRAM)
static_assert(kLatencyFormsMix == kMiniFormsMix && kLatencyFormsMoe == kMiniFormsMoe &&
              kLatencyFormsAll == kMiniFormsAll && kMiniDefaultForms == kMiniFormsMix &&
              kLatencyFormsBatch == kMiniFormsBatch && kLatencyFormsB3 == kMiniFormsB3);

using MiniCudaProgramLoad = std::unique_ptr<serve::SerialGenerationDecoder> (*)(const serve::SerialGenerationLoad&,
                                                                                moe_core::grouped_w4a4::Activations, bool, bool,
                                                                                bool);

// The exact R1 latency forms are a template parameter of the loader; map the CLI mask to its instance.
inline MiniCudaProgramLoad select_mini_cuda_program(bool fuse, unsigned forms) {
    const auto pick = [forms](auto fused) -> MiniCudaProgramLoad {
        constexpr bool on = decltype(fused)::value;
        switch (forms) {
        case kMiniFormsMix: return load_mini_cuda_program<on, kMiniFormsMix>;
        case kMiniFormsMoe: return load_mini_cuda_program<on, kMiniFormsMoe>;
        case kMiniFormsAll: return load_mini_cuda_program<on, kMiniFormsAll>;
        case kMiniFormsBatch: return load_mini_cuda_program<on, kMiniFormsBatch>;
        case kMiniFormsB3: return load_mini_cuda_program<on, kMiniFormsB3>;
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
                     " [--no-epilogue-fusion] [--latency-forms[=mix|moe|batch|b3] | --no-latency-forms]"
#endif
                     " [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n"
                     "       MODEL.safetensors --gate-manifest FILE --gate-out DIR --gate-arm NAME"
                     " --gate-rows N --gate-mode prefill|decode [--prefill-chunk N] [--packed-decode | --grouped-decode]"
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                     " [--no-epilogue-fusion] [--latency-forms[=mix|moe|batch|b3] | --no-latency-forms]"
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
        family.speculation_probe_loss = kMiniProbeLoss;  // models/inkling2-mini.toml [speculation.probe_loss]
        family.prefill_tail_divisor =
            mini_prefill_schedule_from_env(strix::inkling2::cli::kMiniHipSchedule, strix::profile::Name::standard).tail_divisor;
#if defined(STRIX_INKLING2_MEDIA)
        family.render_media_messages = serve::render_inkling2_chat_media_request;
#endif
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        std::vector<char*> arguments(argv, argv + argc);
        const auto program_options = cli::extract_step_program(arguments);
        auto options = cli::parse(static_cast<int>(arguments.size()), arguments.data(), family, kMiniBuildBackend);
#else
        auto options = cli::parse(argc, argv, family, kMiniBuildBackend);
#endif
#if defined(STRIX_INKLING2_MEDIA)
        family.media = cli::mini_media_opener(options.tier, std::cerr);  // image recipe: this run's resolution
#endif
        if (!options.generation.prefill_chunk_explicit) {  // the declared chunk of this backend and profile
            family.default_prefill_chunk = mini_prefill_schedule(cli::kMiniHipSchedule, options.tier.profile).chunk;
            options.generation.prefill_chunk = family.default_prefill_chunk;
        }
        // The implicit chunk's pressure fallback of this backend and profile (shared load; never an explicit chunk).
        family.prefill_chunk_fallback = mini_cli_prefill_fallback_rows(
            cli::kMiniHipSchedule ? MiniPrefillBackend::hip : MiniPrefillBackend::cuda, options.tier.profile);
        // The serial CLI resolves its one lane at load using prompt + generated positions. A gate knows its
        // resident context from the manifest. CUDA step routing precedes serial loading and has no banded CUDA row.
        struct ResolvedWeights {
            MiniWeightScope scope;
            strix::weight::WeightLoadPlan plan;
        };
        auto weight_options = options.weights;
        // The server's rule (mini_speculation_policy): an explicit speculation declines the low-bit row, the row default
        // yields to a requested low-bit row.
        // A decode-only weight arm (explicit qualification: one-row decode on its weights, verify windows on BF16) is a
        // low-bit target too: the server declines its windows (same_target), so the command line declines speculation.
        const bool decode_only_arm = weight_options.qualify != nullptr && weight_options.qualify_decode_only;
        const auto speculation_policy = mini_speculation_policy(
            options.generation.speculation != serve::CliSpeculation::off,
            options.generation.speculation_requested || options.dspark != nullptr,
            weight_options.lowbit_turbo || decode_only_arm);
        if (weight_options.lowbit_turbo && !speculation_policy.lowbit) {
            weight_options.lowbit_turbo = false;
            std::cerr << "strix-lowbit-pack/v2 declined (" << kMiniLowbitSpeculationDecline
                      << "); using incumbent weights\n";
        }
        if (speculation_policy.requested && (!speculation_policy.admitted || decode_only_arm)) {
            if (decode_only_arm && speculation_policy.admitted)
                std::cerr << "speculation declined (" << kMiniSpeculationLowbitDecline << ": decode-only weight arm)\n";
            options.generation.speculation = serve::CliSpeculation::off;
        }
        const auto resolve_weights = [&options, &weight_options, model = std::string(argv[1])](strix::weight::WeightLoadExtent extent) {
            auto scope = mini_weight_scope(model, options.tier.profile, weight_options, extent);
            const auto plan = resolve_mini_resident_weight_load(options.tier.profile, weight_options, scope.view(), extent);
            return ResolvedWeights{std::move(scope), plan};
        };
        const auto gate_windows = options.gate_manifest
            ? strix::session::read_head_gate_manifest(options.gate_manifest, *options.gate_mode, options.gate_rows,
                                                      kMaxContext, kOutputVocabulary)
            : std::vector<strix::session::HeadGateWindow>{};
        const auto gate_extent = cli::gate_weight_extent(gate_windows);
        auto early = resolve_weights(options.gate_manifest ? gate_extent : strix::weight::WeightLoadExtent{});
        std::optional<ResolvedWeights> loaded;
        auto& weights = early.plan;
        const auto step_graphs = options.step_graphs.value_or(MiniResidentBackend::default_step_graphs);
        bool plain_program = false;  // the verifying route's one-row steps on the CUDA step program
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        const char* grouped_prepare = strix::envcfg::text("STRIX_INKLING_K1BIND");
        const bool resident_k1 = grouped_prepare && std::string_view(grouped_prepare) == "1";
        const auto route = cli::resolve_cuda_step_route(options, program_options, weights.storage, resident_k1);
        const bool step_program = route.program;
        plain_program = route.plain_program;
        options.generation.speculation = route.speculation;
#endif
        // The resident route's drafter (strix/spec/drafter_source.hpp), resolved after the CUDA step route (which may
        // turn speculation off) and alive for the whole run: --dspark FILE|off, else the model's MTP block beside the
        // checkpoint when this run speculates (auto, greedy). A sidecar found beside the model is validated against the
        // MTP block schema before anything is reserved for it: a file the loader would refuse is declined with its
        // reason and the run keeps its context drafts (source absent / rejected = context drafts only); an explicit
        // --dspark file is never declined (its loader error stands).
        const auto drafter = [&] {
            auto found = strix::spec::resolve_drafter_file(
                {.file = options.dspark ? std::string_view(options.dspark) : std::string_view{},
                 .off = options.dspark_off,
                 .speculating = options.generation.speculation == serve::CliSpeculation::automatic &&
                                !options.gate_manifest && !options.generation.tokenize_file &&
                                !token_sampling_enabled(options.generation.sampling)},
                argv[1], strix::model_data::inkling2_mini::speculation::drafter);
            if (found.source == strix::spec::DrafterSource::beside_model)
                if (auto problem = strix::inkling2::MiniDSparkDeviceWeights::sidecar_problem(found.path); !problem.empty())
                    return strix::spec::decline_drafter(found, std::move(problem));
            return found;
        }();
        const bool speculating = options.generation.speculation != serve::CliSpeculation::off;
#if defined(STRIX_INKLING2_STEP_PROGRAM)
        if (step_program)
            family.load = [program_load = cli::select_mini_cuda_program(program_options.epilogue_fusion,
                                                                        program_options.latency_forms),
                           &options, device = early.scope.device_class, product = early.scope.cost_device,
                           controller_one_plane = mini_controller_one_plane(options.tier),
                           projected_bias = mini_attn_projected_bias(options.tier),
                           screened_head = mini_screened_head(options.tier)](const serve::SerialGenerationLoad& load) {
                return program_load(load, mini_prompt_moe(options.weights, device, load.prompt_tokens, false, &options.tier,
                                                MiniPromptExecution{product, load.positions, load.prefill_chunk}),
                                    controller_one_plane, projected_bias, screened_head);
            };
        else
#endif
        family.load = [&loaded, &resolve_weights, &options, &weight_options, step_graphs, &drafter, plain_program,
                       combine_image = mini_combine_image(options.tier),
                       round_fold = mini_prepared_round_fold(options.tier),
                       narrow = mini_experts_nvfp4(options.tier),
                       controller_one_plane = mini_controller_one_plane(options.tier),
                       projected_bias = mini_attn_projected_bias(options.tier),
                       screened_head = mini_screened_head(options.tier),
                       verify_rows = mini_speculation_verify_rows(mini_verify_rows(options.tier), speculating)](
                          const serve::SerialGenerationLoad& load) {
            // The drafter loads only when this request may draft (the shared CLI's resolved mode).
            const char* dspark = drafter.enabled() && load.speculation == serve::CliSpeculation::automatic
                                     ? drafter.path.c_str() : nullptr;
            const strix::weight::WeightLoadExtent extent{load.positions, 1U};
            if (const auto decline = mini_lowbit_row_decline(options.tier.profile, weight_options, kMiniWeightBackend,
                                                             extent);
                !decline.empty())
                std::cerr << "strix-lowbit-pack/v2 declined (" << decline << "); using incumbent weights\n";
            loaded = resolve_weights(extent);
            const auto moe_w4a4 = mini_prompt_moe(weight_options, loaded->scope.device_class,
                                                 load.prompt_tokens, false, &options.tier,
                                                 MiniPromptExecution{loaded->scope.cost_device, load.positions,
                                                                     load.prefill_chunk});
            return load_mini_weights(load.checkpoint, loaded->plan, options.tier.profile, weight_options, loaded->scope.view(),
                                     extent, [&](MiniLowBitRows lowbit) {
                return std::make_unique<cli::DeviceCliDecoder>(load, step_graphs, loaded->plan.roles(), dspark,
                    moe_w4a4, combine_image, round_fold, narrow, controller_one_plane, projected_bias, screened_head,
                    verify_rows, std::move(lowbit), plain_program);
            });
        };
        // The rows the decoder was loaded with once the shared load fell back (strix-prefill-chunk line): a second load
        // happens only then, and the prefill route receipt names the chunk that ran.
        unsigned loads = 0U;
        std::uint32_t fallback_chunk = 0U;
        family.load = [load = std::move(family.load), &loads, &fallback_chunk](const serve::SerialGenerationLoad& request) {
            if (loads++ != 0U) fallback_chunk = request.prefill_chunk;
            return load(request);
        };
        const auto write_profile = [&] {
            const auto& resolved = loaded ? *loaded : early;
            const auto& plan = resolved.plan;
            const auto& scope = resolved.scope;
            cli::write_run_receipt(options.tier, plan, std::cerr);
            // The admitted W4A4 class of the prompt that loaded (mini_prompt_moe), the same identity the gate prints
            // and the server reports as moe_prefill.
            if (mini_moe_w4a4(options.tier))
                std::cerr << "strix-moe-w4a4 class "
                          << strix::moe_core::grouped_w4a4::identity(mini_moe_w4a4_class(options.tier)) << '\n';
            std::cerr << "strix-prefill-route/v1 chunk " << (fallback_chunk ? fallback_chunk : options.generation.prefill_chunk)
                      << " tail_divisor " << family.prefill_tail_divisor << " first_id prefill\n";
            std::cerr << "strix-decode-route/v1 moe_decode " << mini_decode_form_name(options.generation.packed_decode)
                      << '\n';
            std::cerr << mini_served_route_cli_line(options.tier) << '\n';
            {
                auto shown = speculation_policy;  // the server prints the same line (mini_speculation_line)
                shown.admitted = speculating;     // after the CUDA step route, which may decline (its own line below)
                std::cerr << mini_speculation_line(shown, mini_speculation_verify_rows(mini_verify_rows(options.tier),
                                                                                         speculating))
                          << '\n';
            }
#if defined(STRIX_INKLING2_STEP_PROGRAM)
            if (!step_program)
#endif
            if (options.generation.speculation != serve::CliSpeculation::off)
                std::cerr << strix::spec::drafter_receipt("resident", drafter) << '\n';
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
            // The full scope a decision row of this load matches (digest literals for weight::digest_from_hex).
            if (!(scope.checkpoint == strix::weight::Digest256{}))
                std::cerr << strix::weight::weight_load_scope_receipt(scope.view()) << '\n';
        };
        if (options.gate_manifest) {
            cli::run_gate(argv[1], options, gate_windows, gate_extent, weights, early.scope.view(), early.scope.device_class,
                          early.scope.cost_device
#if defined(STRIX_INKLING2_STEP_PROGRAM)
                          , step_program, program_options.epilogue_fusion, program_options.latency_forms
#endif
            );
            write_profile();
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

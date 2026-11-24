// Keep the existing generation, tokenization and gate routes in the common
// app entry. The forced-row ticket is specific to the CUDA step program.
#define main inkling2_cuda_generation_main
#include "../../apps/inkling2_mini_gpu_main.hpp"
#undef main

#include "cuda/core/lookup_cuda.cuh"
#include "strix/serve/resident_plan.hpp"
#include "strix/spec/greedy_continuation.hpp"

#include <cstring>

namespace strix::inkling2::detail {
namespace {

template <class Args, auto Enqueue>
exec::KernelBinding bind_verify_leaf(const Args& args, const char* identity, exec::Grid grid) {
    exec::KernelBinding binding;
    binding.fn = +[](std::span<const std::byte> bytes, const exec::Grid&, void* stream) {
        Args unpacked{};
        if (bytes.size() != sizeof(unpacked)) throw std::invalid_argument("invalid Inkling verify binding bytes");
        std::memcpy(&unpacked, bytes.data(), sizeof(unpacked));
        Enqueue(unpacked, static_cast<cudaStream_t>(stream));
    };
    binding.identity = binding.exact_identity = identity;
    binding.grid = grid;
    const auto bytes = std::as_bytes(std::span(&args, 1U));
    binding.args.assign(bytes.begin(), bytes.end());
    binding.predicted.launches = 1U;
    return binding;
}

}  // namespace

exec::KernelBinding bind_mini_verify(const MiniCudaVerifyEmbedding& args) {
    constexpr auto enqueue = [](const MiniCudaVerifyEmbedding& a, cudaStream_t stream) {
        namespace lookup = cuda::core::lookup;
        const lookup::View view{{a.table, kVocabulary, kHiddenWidth}, lookup::Form::bf16_to_f32,
                                a.output, a.rows, 1U, nullptr, nullptr, false};
        mini_gpu::check(lookup::enqueue(view, lookup::StepTokens{a.step}, stream), "enqueue Inkling verify lookup");
    };
    return bind_verify_leaf<MiniCudaVerifyEmbedding, enqueue>(args, "cuda-inkling2-step-embedding-bf16-v1",
        {{args.rows, 1U, 1U}, {256U, 1U, 1U}, 0U});
}

}  // namespace strix::inkling2::detail

namespace strix::inkling2::cli {
namespace {

// Use the actual end of the resident plan to exercise the capped geometry
// without a model-maximum prompt. The repeated prefix supplies four drafts.
void compare_context_edge(MiniCudaProgramDecoder& program, unsigned positions) {
    constexpr unsigned budget = 6U;
    const auto first = positions - budget;
    const auto token_at = [](unsigned position) { return position % 7U == 0U ? 3U : 1U; };
    const auto anchor = token_at(first);
    std::vector<unsigned> prefix(first);
    for (unsigned position = 0; position < first; ++position) prefix[position] = token_at(position);
    const SerialTokenContinuationRequest request{anchor, first, kOutputVocabulary, positions, budget, kMiniEosToken};
    for (const auto& rule : {spec::RowRule{}, spec::RowRule({1.0F, 2U, 1.0F, 0.0F, 19U}, {})}) {
        program.reset_gate_session();
        program.prefill(prefix, {});
        std::vector<float> logits(kOutputVocabulary);
        const auto serial = execute_serial_token_continuation(request, [&](unsigned token, std::uint64_t position) {
            const auto pick = program.decode(token, logits);
            return rule.sampled() ? rule.row_token(logits, position + 1U) : pick;
        });
        program.reset_gate_session();
        program.prefill(prefix, {});
        spec::BlockConfig config;
        config.max_block = budget - 1U;
        config.fixed_costs = true;
        config.force_lookup = true;  // Exercise the edge independently of model acceptance of the fixture's drafts.
        spec::set_context_cost(config.arbiter, "off");
        spec::BlockDrafter drafter(config);
        for (unsigned row = 1U; row <= config.max_block; ++row) drafter.seed_verify_ms(row, 1.0);
        spec::EngineVerifyTarget target(program);
        const auto result = spec::execute_block_token_continuation(request, prefix, drafter, target, config.max_block,
                                                                   {}, rule);
        const auto& widths = drafter.stats().rows_passes;
        if (widths.size() <= 4U || widths[4U] == 0U ||
            result.generated_tokens != serial.generated_tokens || result.stop_reason != serial.stop_reason ||
            program.committed() != serial.final_position)
            throw std::runtime_error("Inkling context-edge verify differs from serial or did not draft a window");
    }
    std::cout << "context-edge greedy and sampled continuations match serial\n";
}

// Same-plan comparison: hash the committed K/V slots in physical order and the recurrent history.
std::uint64_t verify_state_digest(MiniResidentDecoder& decoder) {
    const auto view = decoder.slot_view(0U);
    const auto stream = decoder.step_program_backend().stream();
    auto digest = serve::resident::kFnv64Offset;
    std::vector<char> host;
    const auto append = [&](std::span<const std::byte> bytes) {
        host.resize(bytes.size());
        mini_gpu::copy(host.data(), bytes.data(), bytes.size(), mini_gpu::device_to_host, stream);
        mini_gpu::synchronize(stream);
        digest = serve::resident::fnv1a64({host.data(), host.size()}, digest);
    };
    for (const auto visit : MiniSessionStatePlan::visits()) {
        const auto& layout = view.plan.visit(visit);
        const auto elements = std::size_t{std::min(view.committed, layout.kv_rows)} * layout.kv.width;
        for (const auto plane : {view.kv.keys, view.kv.values})
            append(std::as_bytes(plane.subspan(layout.kv_offset, elements)));
    }
    append(std::as_bytes(view.history));
    return digest;
}

void run_verify(std::vector<char*>& args, unsigned rows, bool compare) {
    const serve::SerialGenerationFamily family{
        "strix-inkling2-mini-cuda", "Inkling 2 Mini", "MODEL.safetensors", kVocabulary, kOutputVocabulary,
        kMaxContext, kMiniEosToken, {}, kMiniDefaultPrefillChunk, true, true,
        nullptr, true};
    const auto options = parse(static_cast<int>(args.size()), args.data(), family);
    const auto& generation = options.generation;
    if (options.step_graphs || !generation.packed_decode || options.gate_manifest ||
        generation.generate_requested || generation.sampling_requested || options.tier.profile != profile::Name::standard)
        throw std::invalid_argument("--verify-rows requires exact packed --step-program without gate, graph or sampling options");
    if (moe::route_tap_config_from_env().active())
        throw std::invalid_argument("--verify-rows requires route tap and replay to be disabled");
    require_mini_verify_width(rows);
    if (!generation.logits || generation.prompt.size() <= rows || generation.prompt.size() > kMaxContext)
        throw std::invalid_argument("--verify-rows needs prefix IDs, R forced IDs and --logits FILE");

    const std::span<const unsigned> ids(generation.prompt);
    const auto prefix = ids.first(ids.size() - rows);
    const auto forced = ids.last(rows);
    const auto first = static_cast<unsigned>(prefix.size());
    (void)mini_verify_descriptor(rows, forced, first);
    if (!std::ranges::all_of(prefix, [](unsigned id) { return id < kVocabulary; }))
        throw std::invalid_argument("Inkling verify prefix token out of range");
    if (std::filesystem::exists(generation.logits)) throw std::invalid_argument("verify logits output already exists");

    write_profile(options.tier, std::cerr, true);
    const auto loaded = session::CliClock::now();
    // The window stages all R rows as one pending append from the prefix frontier: the session plan comes from the
    // serving rule (chunk >= R), not from the prefill chunk, which a short prefix clamps below R.
    const auto chunk = std::min(generation.prefill_chunk, first);
    const auto geometry = mini_step_session_geometry(static_cast<unsigned>(ids.size()), chunk, rows);
    MiniResidentDecoder resident(args[1], geometry.positions, chunk, true, false, weight::kBf16WeightStorage, 0U,
                                 false, 1U, 0U, geometry.append_rows);
    MiniCudaProgramDecoder program(resident);
    session::write_load_timing(std::cout, loaded);
    std::vector<float> logits(std::size_t{rows} * kOutputVocabulary);
    std::vector<unsigned> picks(rows);
    program.prefill(prefix, {});
    program.verify(forced, picks, logits);
    if (compare) {
        program.reset_gate_session();
        program.prefill(prefix, {});
        std::vector<float> reference(kOutputVocabulary);
        const auto matches = [&](unsigned row, unsigned pick) {
            const auto* actual = logits.data() + std::size_t{row} * kOutputVocabulary;
            return pick == picks[row] && std::memcmp(actual, reference.data(), reference.size() * sizeof(float)) == 0;
        };
        std::array<std::uint64_t, 2> serial_state{};
        for (unsigned row = 0; row < rows; ++row) {
            const auto pick = program.decode(forced[row], reference);
            if (rows >= 4U && row < serial_state.size()) serial_state[row] = verify_state_digest(resident);
            if (!matches(row, pick))
                throw std::runtime_error("Inkling verify differs from sequential R=1 at row " + std::to_string(row));
        }
        std::cout << "R=" << rows << " vs " << rows << "xR=1 row_logits_bitwise=" << rows << '/' << rows
                  << " ids=" << rows << '/' << rows << '\n';
        // With a prefix ending at the last local ring row, keep=1 ends at
        // the boundary and keep=2 publishes both the final row and row zero.
        if (rows >= 4U) {
            for (const unsigned keep : {1U, 2U}) {
                program.reset_gate_session();
                program.prefill(prefix, {});
                program.forward_rows(forced.data(), rows, MiniCudaProgramDecoder::LogitRows::all);
                program.commit(keep);
                if (program.committed() != first + keep || verify_state_digest(resident) != serial_state[keep - 1U])
                    throw std::runtime_error("Inkling verify state differs after keeping " + std::to_string(keep) +
                                             " rows");
                for (unsigned row = keep; row < rows; ++row) {
                    const auto pick = program.decode(forced[row], reference);
                    if (!matches(row, pick))
                        throw std::runtime_error("Inkling verify continuation differs after keeping " +
                                                 std::to_string(keep) + " rows at row " + std::to_string(row));
                }
            }
            std::cout << "partial-commit continuation row_logits_bitwise=yes ids=yes state_digest_equal=yes\n";
        }
        if (rows == kMiniVerifyMaxRows) compare_context_edge(program, geometry.positions);
    }
    session::write_logits(generation.logits, logits);
    for (unsigned row = 0; row < rows; ++row) {
        std::cout << "inkling2 verify row=" << row << " position=" << first + row << " forced=" << forced[row]
                  << " picked=" << picks[row] << " logits_offset=" << std::size_t{row} * kOutputVocabulary * sizeof(float)
                  << '\n';
    }
    std::cout << "inkling2 verify logits=" << generation.logits << " rows=" << rows << " columns=" << kOutputVocabulary << '\n';
}

}  // namespace
}  // namespace strix::inkling2::cli

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    if (argc < 3) {
        std::cerr << "       MODEL.safetensors PREFIX_IDS FORCED_IDS --step-program --verify-rows R --logits FILE"
                     " [--verify-compare]\n";
        return inkling2_cuda_generation_main(argc, argv);
    }
    try {
        std::vector<char*> args(argv, argv + argc);
        const auto verify = cli::extract_verify_options(args);
        if (!verify.rows && !verify.compare) return inkling2_cuda_generation_main(argc, argv);
        cli::run_verify(args, *verify.rows, verify.compare);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CUDA: " << error.what() << '\n';
        return 1;
    }
}

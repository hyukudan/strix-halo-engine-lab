// Keep the existing generation, tokenization and gate routes in the common
// app entry. The forced-row ticket is specific to the CUDA step program.
#define main inkling2_cuda_generation_main
#include "../../apps/inkling2_mini_gpu_main.hpp"
#undef main

#include "cuda/core/lookup_cuda.cuh"

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

struct VerifyOptions {
    std::optional<unsigned> rows;
    bool compare = false;
};

VerifyOptions extract_verify_options(std::vector<char*>& args) {
    constexpr std::string_view value_options[]{
        "--generate", "--logits", "--temperature", "--top-k", "--seed", "--prompt", "--messages",
        "--tokenize", "--tokenizer", "--prefill-chunk", "--profile", "--gate-manifest", "--gate-out",
        "--gate-arm", "--gate-rows", "--gate-mode",
    };
    VerifyOptions options;
    auto output = args.begin() + 2;
    for (auto arg = output; arg != args.end(); ++arg) {
        const std::string_view value(*arg);
        if (value == "--verify-rows") {
            if (options.rows || arg + 1 == args.end())
                throw std::invalid_argument("--verify-rows needs one width");
            options.rows = core::parse_u32(*++arg);
        } else if (value == "--verify-compare") {
            if (options.compare) throw std::invalid_argument("duplicate --verify-compare");
            options.compare = true;
        } else {
            *output++ = *arg;
            if (std::ranges::find(value_options, value) != std::end(value_options) && arg + 1 != args.end())
                *output++ = *++arg;
        }
    }
    args.erase(output, args.end());
    return options;
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
        for (unsigned row = 0; row < rows; ++row) {
            const auto pick = program.decode(forced[row], reference);
            const auto* actual = logits.data() + std::size_t{row} * kOutputVocabulary;
            if (pick != picks[row] || std::memcmp(actual, reference.data(), reference.size() * sizeof(float)) != 0)
                throw std::runtime_error("Inkling verify differs from sequential R=1 at row " + std::to_string(row));
        }
        std::cout << "R=" << rows << " vs " << rows << "xR=1 row_logits_bitwise=" << rows << '/' << rows
                  << " ids=" << rows << '/' << rows << '\n';
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
        if (!verify.rows || !cli::extract_step_program(args).requested)
            throw std::invalid_argument("verify requires --step-program and --verify-rows R");
        cli::run_verify(args, *verify.rows, verify.compare);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CUDA: " << error.what() << '\n';
        return 1;
    }
}

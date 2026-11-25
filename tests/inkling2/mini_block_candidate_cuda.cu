// Device fixture of the Inkling 2 Mini CUDA block candidate (no model files). An explicit greedy selection
// prepares three forced rows at positions - 4 (width four, the fourth row masked); append_tap binds the shared
// BF16 row-store leaf for tap 1 of a three-tap feature store into a CUDA program recipe, which is compiled and
// executed on a stream. The whole store is copied back and compared with a serial reference: the BF16 encode
// of the three valid hidden rows at their positions in tap 1's slice, every other cell the canary (the masked
// fourth row, taps 0 and 2, and every other position).
// Prints "inkling2-mini-block-candidate-cuda: PASS ..." and returns 0, or names the first mismatch and returns 1.
#include "exec/cuda_program_recipe.hpp"
#include "inkling2/inkling2_mini_block_candidate_cuda.hpp"
#include "strix/attn/cache_rows.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/spec/block_draft_policy.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mini = strix::inkling2;
namespace exec = strix::exec;
namespace pf = strix::profile;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <class T>
struct Device {
    explicit Device(std::size_t count) : size(count) { check(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc"); }
    ~Device() { (void)cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const T* host) { check(cudaMemcpy(data, host, size * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> read() const {
        std::vector<T> host(size);
        check(cudaMemcpy(host.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost), "read");
        return host;
    }
    T* data{};
    std::size_t size;
};

}  // namespace

int main() try {
    constexpr unsigned positions = 64U, first = positions - 4U, tap_count = 3U, tap = 1U;
    constexpr unsigned hidden = mini::kHiddenWidth, pitch = tap_count * hidden;
    constexpr std::uint16_t canary = 0xA5A5U;
    constexpr std::array<unsigned, 3> ids{11U, 17U, 23U};

    pf::Inputs options;
    options.profile = pf::Name::standard;
    options.env = [](const char*) -> const char* { return nullptr; };
    options.cli.emplace_back(strix::spec::kBlockDraftLeverName, "dflash2:8:greedy");
    const auto step = mini::MiniCudaBlockCandidateStep::prepare(pf::resolve(mini::kMiniProfile, options), ids, first,
                                                                positions);
    if (!step) throw std::runtime_error("explicit greedy selection did not prepare a candidate");
    const auto& block = step->descriptor();
    if (block.h.rows != 4U || block.h.valid_mask != 0b0111U)
        throw std::runtime_error("three forced rows need a width-four descriptor with the fourth row masked");

    // Every row, the masked fourth included, carries finite nonzero values with distinct BF16 encodings.
    std::vector<float> rows(std::size_t{block.h.rows} * hidden);
    for (std::size_t i = 0; i < rows.size(); ++i)
        rows[i] = (static_cast<float>(i % 997U) - 498.25F) * 0.0137F + static_cast<float>(i / hidden + 1U);
    rows[1] = 1.0F / 3.0F;  // a rounding tie candidate and specials ride along
    rows[2] = -0.0F;
    rows[3] = 65504.0F;

    Device<exec::StepBlock> device_step(1U);
    Device<float> device_hidden(rows.size());
    Device<std::uint16_t> features(std::size_t{positions} * pitch);
    device_step.upload(&block);
    device_hidden.upload(rows.data());
    const std::vector<std::uint16_t> filled(features.size, canary);
    features.upload(filled.data());

    exec::CudaProgramRecipe recipe;
    recipe.stage("fixture");
    step->append_tap(recipe, device_step.data, std::span<const float>(device_hidden.data, device_hidden.size),
                     std::span<std::uint16_t>(features.data, features.size), tap_count, tap);
    if (recipe.stage_label() != "fixture") throw std::runtime_error("append_tap did not restore the stage label");
    const auto compiled = recipe.compile({.phase = exec::Phase::verify, .rows = block.h.rows},
                                         exec::CudaReadback::resident);
    const auto& program = compiled.program;
    if (program.launches.size() != 1U) throw std::runtime_error("the tap must lower to one recipe leaf");

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    for (const auto& leaf : program.launches) {
        if (leaf.kernels != 1U || !leaf.fn) throw std::runtime_error("incorrect leaf launch receipt");
        leaf.fn(std::span(program.args).subspan(leaf.args_offset, leaf.args_bytes), leaf.grid, stream);
    }
    check(cudaGetLastError(), "launch");
    check(cudaStreamSynchronize(stream), "synchronize");
    check(cudaStreamDestroy(stream), "stream destroy");

    // Serial reference over the complete destination.
    std::vector<std::uint16_t> expected(filled);
    for (unsigned row = 0; row < ids.size(); ++row) {
        for (unsigned c = 0; c < hidden; ++c) {
            expected[std::size_t{first + row} * pitch + std::size_t{tap} * hidden + c] =
                strix::attn::cache_rows::encode(rows[std::size_t{row} * hidden + c]);
        }
    }
    const auto actual = features.read();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            const auto position = i / pitch, slice = (i % pitch) / hidden, element = i % hidden;
            std::fprintf(stderr, "inkling2-mini-block-candidate-cuda: FAIL cell %zu (position %zu, tap %zu, "
                         "element %zu) got %04x expected %04x\n", i, position, slice, element,
                         static_cast<unsigned>(actual[i]), static_cast<unsigned>(expected[i]));
            return 1;
        }
    }
    const std::size_t written = ids.size() * std::size_t{hidden};
    std::printf("inkling2-mini-block-candidate-cuda: PASS rows=3 width=4 tap=%u/%u positions=%u..%u of %u "
                "cells=%zu written=%zu canaries=%zu\n", tap, tap_count, first, first + 2U, positions, actual.size(),
                written, actual.size() - written);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "inkling2-mini-block-candidate-cuda: FAIL %s\n", error.what());
    return 1;
}

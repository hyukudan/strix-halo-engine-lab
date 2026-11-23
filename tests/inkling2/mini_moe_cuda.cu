#include "inkling2/inkling2_mini_moe_cuda.hpp"
#include "mini_moe_device_fixture.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

namespace {

struct Cuda {
    using MoeBackend = strix::inkling2::MiniCudaMoeBackend;
    using Stream = cudaStream_t;
    static constexpr auto enqueue_resident = strix::inkling2::enqueue_mini_cuda_resident_moe;

    static void require(cudaError_t status) {
        if (status == cudaSuccess) return;
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
    static void* allocate(std::size_t bytes) {
        void* memory = nullptr;
        require(cudaMallocManaged(&memory, bytes));
        return memory;
    }
    static Stream create_stream() {
        Stream stream{};
        require(cudaStreamCreate(&stream));
        return stream;
    }
    static void sync(Stream stream) { require(cudaStreamSynchronize(stream)); }
    static void destroy_stream(Stream stream) { require(cudaStreamDestroy(stream)); }
    static void release(void* memory) { require(cudaFree(memory)); }
};

template <class T>
auto allocate(std::size_t elements) {
    return std::unique_ptr<T, decltype(&Cuda::release)>(
        static_cast<T*>(Cuda::allocate(elements * sizeof(T))), Cuda::release);
}

bool check_router_bits(unsigned rows) {
    namespace mini = strix::inkling2;
    namespace cc = strix::cuda_core;
    constexpr unsigned width = mini::kHiddenWidth;
    constexpr unsigned experts = mini::kExperts;
    const auto input_elements = std::size_t{rows} * width;
    const auto output_elements = std::size_t{rows} * experts;
    auto input = allocate<float>(input_elements);
    auto expected = allocate<float>(output_elements);
    auto actual = allocate<float>(output_elements);
    auto weight = allocate<std::uint16_t>(std::size_t{width} * experts);
    for (std::size_t i = 0; i < input_elements; ++i)
        input.get()[i] = static_cast<float>(static_cast<int>((i * 31U + i / width) % 973U) - 500) / 997.0F;
    for (std::size_t i = 0; i < std::size_t{width} * experts; ++i)
        weight.get()[i] = static_cast<std::uint16_t>(0x3B00U + (i * 17U + i / width) % 1021U);

    const cc::DenseWeight router{weight.get(), cc::WeightFormat::bf16, experts, width};
    for (unsigned first = 0; first < rows;) {
        const unsigned count = std::min(cc::kMaxRows, rows - first);
        Cuda::require(cc::dense_rows(router, input.get() + std::size_t{first} * width,
                                     expected.get() + std::size_t{first} * experts, count, nullptr, nullptr,
                                     cc::DenseArith::fp32));
        first += count;
    }
    Cuda::require(mini::enqueue_mini_cuda_router_logits(router, rows, {input.get(), input_elements},
                                                        {actual.get(), output_elements}, nullptr));
    Cuda::sync(nullptr);
    if (std::memcmp(expected.get(), actual.get(), output_elements * sizeof(float)) == 0) return true;
    std::cerr << "CUDA Mini FP32 router projection bits differ at " << rows << " rows\n";
    return false;
}

namespace mma = strix::cuda::core::mma::microscaled;
namespace geometry = strix::cuda::core::native_mma;
namespace fixture = strix::tests::mini_moe;
namespace packed = strix::weight::microscaled;
namespace view = strix::moe_core::grouped_microscaled;

template<packed::MatrixFormat Format>
bool tile_forms(Cuda::Stream stream) {
    using Operand = packed::PackedOperand<Format>;
    constexpr unsigned width = fixture::kHiddenWidth;
    constexpr unsigned max_rows = 200U;
    constexpr unsigned max_tiles = (max_rows + mma::kRowsPerTile - 1U) / mma::kRowsPerTile;
    constexpr std::size_t weights = std::size_t{width} * width;
    constexpr std::size_t values = std::size_t{max_rows + 1U} * width;
    constexpr float sentinel = -1.0e30F;
    fixture::Managed<Cuda, std::uint8_t> codes(weights / (Operand::kNvfp4 ? 2U : 1U));
    fixture::Managed<Cuda, std::uint8_t> scales(weights / Operand::kBlockColumns);
    fixture::Managed<Cuda, std::uint16_t> input(values);
    fixture::Managed<Cuda, float> small(values), wide(values), shift_small(values), shift_wide(values);
    fixture::Managed<Cuda, float> tall(values), shift_tall(values);
    fixture::Managed<Cuda, view::TileRows> small_tiles(max_tiles), wide_tiles(max_tiles);
    fixture::Managed<Cuda, view::TileRows> shift_small_tiles(max_tiles), shift_wide_tiles(max_tiles);
    fixture::Managed<Cuda, view::TileRows> tall_tiles(max_tiles), shift_tall_tiles(max_tiles);
    std::mt19937 random(0x26390U);
    for (auto& code : codes.span()) {
        const auto bits = static_cast<std::uint8_t>(random() & 0xffU);
        code = !Operand::kNvfp4 && (bits & 0x7fU) == 0x7fU ? static_cast<std::uint8_t>(bits - 1U) : bits;
    }
    for (auto& scale : scales.span())
        scale = static_cast<std::uint8_t>(Operand::kNvfp4 ? 0x30U + random() % 16U : 124U + random() % 8U);
    for (auto& value : input.span()) {
        const float sample = static_cast<float>(static_cast<int>(random() % 2049U) - 1024) / 1024.0F;
        value = strix::weight::scalar_float::bf16_encode(sample);
    }
    const packed::PackedMatrix matrix{codes.data, scales.data, width, width, 0.625F,
                                      packed::RowOrder::interleaved_halves};
    for (const unsigned rows : {1U, 16U, 17U, 64U, max_rows}) {
        std::ranges::fill(small.span(), sentinel);
        std::ranges::fill(wide.span(), sentinel);
        std::ranges::fill(shift_small.span(), sentinel);
        std::ranges::fill(shift_wide.span(), sentinel);
        std::ranges::fill(tall.span(), sentinel);
        std::ranges::fill(shift_tall.span(), sentinel);
        unsigned tiles = 0U;
        for (unsigned first = 0; first < rows; first += mma::kRowsPerTile) {
            const unsigned live = std::min(rows - first, mma::kRowsPerTile);
            const auto base = std::size_t{first} * width;
            small_tiles.data[tiles] = {matrix, input.data + base, small.data + base, width, width, live};
            wide_tiles.data[tiles] = {matrix, input.data + base, wide.data + base, width, width, live};
            shift_small_tiles.data[tiles] = {matrix, input.data + base, shift_small.data + base, width, width, live};
            shift_wide_tiles.data[tiles] = {matrix, input.data + base, shift_wide.data + base, width, width, live};
            tall_tiles.data[tiles] = {matrix, input.data + base, tall.data + base, width, width, live};
            shift_tall_tiles.data[tiles] = {matrix, input.data + base, shift_tall.data + base, width, width, live};
            ++tiles;
        }
        const unsigned bound = std::min(rows, mma::kRowsPerTile);
        Cuda::require(mma::detail::launch_form<Format, geometry::kDecodeMBlocks, mma::kDecodeOutputs>(
            {small_tiles.data, tiles, width, width, nullptr, nullptr, bound}, stream));
        Cuda::require(mma::detail::launch_form<Format, geometry::kTileMBlocks, mma::kTileOutputs>(
            {wide_tiles.data, tiles, width, width, nullptr, nullptr, bound}, stream));
        // Exponent-shift weight decode (view::Decode::shift): bitwise the reference decode in both forms.
        view::Args shifted_small{shift_small_tiles.data, tiles, width, width, nullptr, nullptr, bound};
        view::Args shifted_wide{shift_wide_tiles.data, tiles, width, width, nullptr, nullptr, bound};
        shifted_small.decode = shifted_wide.decode = view::Decode::shift;
        Cuda::require(
            mma::detail::launch_form<Format, geometry::kDecodeMBlocks, mma::kDecodeOutputs>(shifted_small, stream));
        Cuda::require(
            mma::detail::launch_form<Format, geometry::kTileMBlocks, mma::kTileOutputs>(shifted_wide, stream));
        // Tall slab schedule (view::Schedule::tall): bitwise the staged forms, both decodes.
        view::Args tall_args{tall_tiles.data, tiles, width, width, nullptr, nullptr, bound};
        view::Args shifted_tall{shift_tall_tiles.data, tiles, width, width, nullptr, nullptr, bound};
        tall_args.schedule = shifted_tall.schedule = view::Schedule::tall;
        shifted_tall.decode = view::Decode::shift;
        Cuda::require(mma::launch<Format>(tall_args, stream));
        Cuda::require(mma::launch<Format>(shifted_tall, stream));
        Cuda::sync(stream);
        const auto live = std::size_t{rows} * width;
        if (std::memcmp(small.data, wide.data, live * sizeof(float)) != 0 ||
            std::memcmp(small.data, shift_small.data, values * sizeof(float)) != 0 ||
            std::memcmp(small.data, shift_wide.data, values * sizeof(float)) != 0 ||
            std::memcmp(small.data, tall.data, values * sizeof(float)) != 0 ||
            std::memcmp(small.data, shift_tall.data, values * sizeof(float)) != 0 ||
            !std::ranges::all_of(wide.span().first(live), [](float v) { return std::isfinite(v) && v != sentinel; }) ||
            !std::ranges::all_of(wide.span().subspan(live), [](float v) { return v == sentinel; }) ||
            !std::ranges::all_of(small.span().subspan(live), [](float v) { return v == sentinel; })) {
            std::cerr << "microscaled tile forms bitwise: FAIL rows=" << rows
                      << (Operand::kNvfp4 ? " nvfp4\n" : " mxfp8\n");
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!check_router_bits(33U) || !check_router_bits(2048U)) return 1;
    if (const int status = fixture::verify<Cuda>(); status != 0) return status;
    const auto stream = Cuda::create_stream();
    const bool passed = tile_forms<packed::MatrixFormat::nvfp4>(stream) && tile_forms<packed::MatrixFormat::mxfp8>(stream);
    Cuda::destroy_stream(stream);
    if (!passed) return 1;
    std::cout << "microscaled tile forms bitwise: PASS rows=1,16,17,64,200 nvfp4+mxfp8 (reference and shift decode; staged and tall)\n";
    return 0;
}

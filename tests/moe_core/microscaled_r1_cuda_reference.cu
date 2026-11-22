#include "microscaled_r1_cuda_fixture.cuh"

// Compile the original caller, arithmetic and epilogues in a distinct namespace
// so the fixture's candidate binding cannot affect this translation unit.
#define microscaled_r1 microscaled_r1_reference
#include "moe_core/microscaled_r1.hpp"
#undef microscaled_r1

namespace strix::tests::microscaled_r1 {

template <packed::MatrixFormat Format, unsigned TopK>
cudaError_t launch_reference(const RoutedCase& route, std::uint16_t* activation, float* output, cudaStream_t stream) {
    namespace r1 = moe_core::microscaled_r1_reference;
    return r1::launch<Format, TopK>(route.bind<r1::Args>(activation, output), stream);
}

template cudaError_t launch_reference<packed::MatrixFormat::nvfp4, 8U>(
    const RoutedCase&, std::uint16_t*, float*, cudaStream_t);
template cudaError_t launch_reference<packed::MatrixFormat::mxfp8, 8U>(
    const RoutedCase&, std::uint16_t*, float*, cudaStream_t);
template cudaError_t launch_reference<packed::MatrixFormat::nvfp4, 9U>(
    const RoutedCase&, std::uint16_t*, float*, cudaStream_t);
template cudaError_t launch_reference<packed::MatrixFormat::mxfp8, 9U>(
    const RoutedCase&, std::uint16_t*, float*, cudaStream_t);

} // namespace strix::tests::microscaled_r1

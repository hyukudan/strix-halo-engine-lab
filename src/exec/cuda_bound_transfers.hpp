#pragma once

#include "executor_cuda.hpp"

#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix::exec {
namespace cuda_detail {

// Use this for a transfer recorded on the same stream immediately after the
// preceding launch. X2c enqueues the native operation from its POD payload.
template <class Args>
void append_serial_transfer(Program& program, CudaProgramBinding& binding, CudaNodeKind kind,
                            std::string_view identity, const Args& args) {
    const auto index = program.launches.size();
    if (index != binding.nodes.size() || index >= no_launch)
        throw std::invalid_argument("incomplete CUDA program binding");

    const auto old_size = program.args.size();
    constexpr auto alignment = alignof(Args);
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
    if (old_size > limit - (alignment - 1U) - sizeof(Args))
        throw std::length_error("CUDA transfer argument buffer exceeds the program range");
    const auto offset = (old_size + alignment - 1U) & ~(alignment - 1U);

    BoundLaunch launch;
    launch.entry = identity;
    launch.identity = identity;
    launch.args_offset = static_cast<std::uint32_t>(offset);
    launch.args_bytes = sizeof(Args);
    launch.rows = program.key.rows;
    CudaNodeBinding node{kind, {}};
    if (index != 0) {
        launch.after = static_cast<std::uint32_t>(index - 1U);
        node.after.push_back({index - 1U, CudaEdgeKind::completion});
    }

    program.args.resize(offset);
    const auto bytes = std::as_bytes(std::span(&args, 1));
    try {
        program.args.insert(program.args.end(), bytes.begin(), bytes.end());
        program.launches.push_back(std::move(launch));
        binding.nodes.push_back(std::move(node));
    } catch (...) {
        program.args.resize(old_size);
        if (program.launches.size() != index) program.launches.pop_back();
        throw;
    }
}

}  // namespace cuda_detail

inline void append_cuda_memset(Program& program, CudaProgramBinding& binding, CudaMemsetArgs args) {
    if (!args.destination || args.bytes == 0) throw std::invalid_argument("empty CUDA memset binding");
    cuda_detail::append_serial_transfer(program, binding, CudaNodeKind::memset, "cuda-native-memset-v1", args);
}

inline void append_cuda_copy(Program& program, CudaProgramBinding& binding, CudaNodeKind kind,
                             CudaCopyArgs args) {
    if ((kind != CudaNodeKind::copy_d2d && kind != CudaNodeKind::copy_d2h_pinned) ||
        !args.destination || !args.source || args.bytes == 0)
        throw std::invalid_argument("invalid CUDA copy binding");
    const auto identity = kind == CudaNodeKind::copy_d2d ? "cuda-native-copy-d2d-v1" : "cuda-native-copy-d2h-v1";
    cuda_detail::append_serial_transfer(program, binding, kind, identity, args);
}

}  // namespace strix::exec

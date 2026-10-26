// CRC-32C for the MiniMax prefill exchange module (pf_exchange.hpp): the vector form (ISA-L folding, AVX-512 +
// VPCLMULQDQ) when the CPU has it, else the communicator's (SSE4.2 / table). Same values either way.
#include "minimax/pfx/pf_crc.hpp"
#include "strix/dist/communicator.hpp"

#include "dist_crc32c_vector.hpp"

#include <span>

namespace strix::minimax::pfx {

std::uint32_t crc32c(const std::byte* data, std::size_t bytes) noexcept {
    const std::span<const std::byte> s(data, bytes);
#if defined(__x86_64__)
    static const bool vector = dist::detail::crc32c_vector_available();
    if (vector && bytes >= 1024U) return dist::detail::crc32c_vector(s);
#endif
    return dist::crc32c(s);
}

}  // namespace strix::minimax::pfx

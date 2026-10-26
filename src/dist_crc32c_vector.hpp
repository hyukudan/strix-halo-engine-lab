// SPDX-License-Identifier: BSD-3-Clause
// CRC32C folding sequence/constants adapted from Intel ISA-L:
// crc32_iscsi_by16_10.asm (Git blob 6ba8ee3ee6d142e818dfde75b051ed7f01e275a1),
// crc_const.asm (Git blob 417ff3e8ba7b4cf2db989a576643bd18fca63e67).
// Original assembly authors: Erdinc Ozturk, Vinodh Gopal, James Guilford.
// https://github.com/intel/isa-l ; distribution notice: LICENSES/ISA-L-CRC32C.txt.
/*
Copyright(c) 2011-2020 Intel Corporation All rights reserved.
Copyright(c) 2026 Intel Corporation All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#if defined(__x86_64__)
#include <immintrin.h>

namespace strix::dist::detail {
inline bool crc32c_vector_available() noexcept {
    static const bool supported = __builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("pclmul") &&
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("vpclmulqdq");
    return supported;
}
#define STRIX_CRC_VECTOR_TARGET __attribute__((target("sse4.2,pclmul,avx512f,avx512dq,avx512bw,avx512vl,vpclmulqdq")))
STRIX_CRC_VECTOR_TARGET inline __m512i crc_fold512(__m512i a, __m512i k, __m512i b) {
    const auto x = _mm512_clmulepi64_epi128(a, k, 0x01);
    const auto y = _mm512_clmulepi64_epi128(a, k, 0x10);
    return _mm512_ternarylogic_epi64(x, y, b, 0x96);
}
STRIX_CRC_VECTOR_TARGET inline __m128i crc_fold128(__m128i a, __m128i k, __m128i b) {
    return _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128(a, k, 0x01), _mm_clmulepi64_si128(a, k, 0x10)), b);
}
// The entry is target-isolated: callers must check crc32c_vector_available().
// Unaligned loads stay within the span, including the 256-byte boundary/tail.
STRIX_CRC_VECTOR_TARGET inline std::uint32_t crc32c_vector(std::span<const std::byte> bytes) noexcept {
    const auto* p = bytes.data();
    auto n = bytes.size();
    if (n < 256U) {
        std::uint64_t crc = 0xffffffffU;
        while (n >= 8U) {
            std::uint64_t word; std::memcpy(&word, p, 8U);
            crc = _mm_crc32_u64(crc, word); p += 8U; n -= 8U;
        }
        while (n) { crc = _mm_crc32_u8(static_cast<std::uint32_t>(crc), static_cast<unsigned char>(*p++)); --n; }
        return static_cast<std::uint32_t>(crc) ^ 0xffffffffU;
    }
    auto a = _mm512_loadu_si512(p), b = _mm512_loadu_si512(p + 64U);
    auto c = _mm512_loadu_si512(p + 128U), d = _mm512_loadu_si512(p + 192U);
    a = _mm512_xor_si512(a, _mm512_zextsi128_si512(_mm_cvtsi32_si128(-1)));
    const auto k = _mm512_broadcast_i32x4(_mm_set_epi64x(0xdcb17aa4LL, 0xb9e02b86LL));
    p += 256U; n -= 256U;
    while (n >= 256U) {
        a = crc_fold512(a, k, _mm512_loadu_si512(p));
        b = crc_fold512(b, k, _mm512_loadu_si512(p + 64U));
        c = crc_fold512(c, k, _mm512_loadu_si512(p + 128U));
        d = crc_fold512(d, k, _mm512_loadu_si512(p + 192U));
        p += 256U; n -= 256U;
    }
    alignas(64) __m128i lanes[16];
    _mm512_store_si512(lanes, a); _mm512_store_si512(lanes + 4, b);
    _mm512_store_si512(lanes + 8, c); _mm512_store_si512(lanes + 12, d);
    const auto k1 = _mm_set_epi64x(0xec1068c50LL, 0x493c7d27LL);
    auto x = lanes[0];
    for (unsigned i = 1; i < 16U; ++i) x = crc_fold128(x, k1, lanes[i]);
    while (n >= 16U) {
        x = crc_fold128(x, k1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        p += 16U; n -= 16U;
    }
    std::uint64_t crc = _mm_crc32_u64(0, static_cast<std::uint64_t>(_mm_cvtsi128_si64(x)));
    crc = _mm_crc32_u64(crc, static_cast<std::uint64_t>(_mm_extract_epi64(x, 1)));
    while (n) { crc = _mm_crc32_u8(static_cast<std::uint32_t>(crc), static_cast<unsigned char>(*p++)); --n; }
    return static_cast<std::uint32_t>(crc) ^ 0xffffffffU;
}
#undef STRIX_CRC_VECTOR_TARGET
} // namespace strix::dist::detail
#endif

#pragma once

namespace strix::attn {

// Arithmetic order of the host attention reference. sequential: the order of the scalar reference (ascending serial
// chains). declared: the order a device kernel of the declared-arithmetic convention computes (lane-tree dots, per-wave
// online softmax with a wave merge, lane-strided halving norm sums, declared exp; attn_core/declared_order_reference.hpp),
// so that the host reference and the device agree bit for bit.
enum class ReferenceOrder : unsigned char { sequential, declared };

}  // namespace strix::attn

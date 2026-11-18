#pragma once
// Physical LDS index of the GDN recurrence reduction buffer: 128 key lanes x 32 value columns of FP32 (one block of
// the HIP recurrence kernels, src/hip_gdn_launcher.hip). Only the physical address of an element depends on the form;
// the logical (lane, column) element, the reduction pairs (lane, lane + stride), the expressions and the barriers do
// not, so every form gives the same bits.
//   lane_major      lane * 32 + column: the 32 lanes of a wave touch one bank group for every column (32-way conflict
//                   on 32 LDS banks; measured 15.8 ms per 512-row Qwen dense window, LDS-conflict bound).
//   lane_major_pad  lane * 33 + column (padded; 33 * 128 floats).
//   column_major    column * 128 + lane: consecutive lanes on consecutive banks (conflict-free b32).
// Home: shared (backend core, HIP GDN recurrence); host-compilable so the index invariants are tested on the host.
#include <cstddef>
#include <cstdint>

namespace strix::gdn_recurrent_lds {
inline constexpr std::uint32_t kLanes = 128;
inline constexpr std::uint32_t kColumns = 32;
enum class Form : std::uint8_t { lane_major = 0, lane_major_pad = 1, column_major = 2 };

// Elements the buffer must hold for a form.
constexpr std::size_t extent(Form form) {
    return form == Form::lane_major_pad ? static_cast<std::size_t>(kLanes) * (kColumns + 1U)
                                        : static_cast<std::size_t>(kLanes) * kColumns;
}
constexpr std::size_t index(Form form, std::uint32_t lane, std::uint32_t column) {
    return form == Form::column_major
        ? static_cast<std::size_t>(column) * kLanes + lane
        : static_cast<std::size_t>(lane) * (form == Form::lane_major_pad ? kColumns + 1U : kColumns) + column;
}
}  // namespace strix::gdn_recurrent_lds

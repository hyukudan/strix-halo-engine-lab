#pragma once
// Producer-side publication of an exchanged partial (pair decode, any engine).
//
// The kernel that writes the last bytes of a rank's partial can publish the message itself: it writes the partial
// straight into the channel's next TX slot and, when its last workgroup has stored, raises the TX flag. The exchange
// then starts at the end of the producing kernel instead of after one more launch (a copy + publish kernel), which
// on a latency-bound decode exchange is the launch gap plus the copy kernel per exchange. Bits are unchanged: the
// same values are stored, only their address differs.
//
// `ProducerPublish` has the layout of the channel's producer target (flag, counter, sequence); the host side fills it
// from the channel and hands it to the producer through the engine's call context.
#include <cstdint>


namespace strix::pair {

struct ProducerPublish {
    std::uint64_t* flag = nullptr;  // device pointer of the TX flag; nullptr = the producer does not publish
    unsigned* counter = nullptr;    // device scratch, zero between messages (reset by the last workgroup)
    std::uint64_t sequence = 0;     // value to publish
    [[nodiscard]] constexpr bool on() const { return flag != nullptr; }
};

}  // namespace strix::pair

#pragma once
#include <cstddef>
#include <exception>

namespace strix::core {
// Ordered independent payload blocks. A failure releases every remaining gate
// (after clearing unread bytes) before propagating; consumers must never hang
// behind an I/O exception. Readiness is published only after the complete read.
template<class Read, class Publish, class Clear>
void read_ahead_blocks(std::size_t blocks, Read read, Publish publish, Clear clear) {
    std::size_t block = 0;
    try {
        for (; block < blocks; ++block) {
            read(block);
            publish(block);
        }
    } catch (...) {
        for (; block < blocks; ++block) {
            clear(block);
            publish(block);
        }
        throw;
    }
}
} // namespace strix::core

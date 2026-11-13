#pragma once
#include <cstdint>
struct StepTokenDigest {
    std::uint64_t hash = 0xcbf29ce484222325ULL, tokens = 0;
    template <typename Tokens>
    void add(const Tokens& step) {
        for (const std::uint32_t id : step) {
            for (unsigned b = 0; b < 32U; b += 8U) hash = (hash ^ ((id >> b) & 0xFFU)) * 0x100000001b3ULL;
            ++tokens;
        }
    }
};

#pragma once
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x))                                                                                                      \
            throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__));        \
    } while (false)
template <class F> void refuses(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    CHECK(rejected);
}
inline void bytes(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float))) {
            std::cerr << "first byte mismatch at " << i << ": " << std::hexfloat << a[i] << " vs " << b[i] << '\n';
            throw std::runtime_error("g4-v1 byte mismatch");
        }
}

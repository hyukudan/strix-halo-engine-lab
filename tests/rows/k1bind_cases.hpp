#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace k1bind_test {

struct Projection {
    unsigned out;
    bool round_bf16 = false;
};

struct Group {
    unsigned input, width;
    std::span<const Projection> projections;
};

struct Case {
    const char* name;
    std::span<const Group> groups;
    std::span<const unsigned> tile_ends;
    std::size_t incumbent_launches, planned_launches;
};

inline constexpr std::array q{Projection{3072}, Projection{384}, Projection{24}, Projection{3072}};
inline constexpr std::array kv{Projection{1024}, Projection{1024}};
inline constexpr std::array k_or_v{Projection{1024}};
inline constexpr std::array with_taps{Projection{1024}, Projection{32}};
inline constexpr std::array o{Projection{2304}};
inline constexpr std::array down{Projection{64, true}};
inline constexpr std::array gates{Projection{2304}, Projection{2304}};

inline constexpr std::array local{Group{0, 2304, q}, Group{0, 2304, kv}};
inline constexpr std::array local_separate{Group{0, 2304, q}, Group{1, 2304, k_or_v}, Group{2, 2304, k_or_v}};
inline constexpr std::array full{Group{0, 2304, q}, Group{1, 2304, with_taps}, Group{2, 2304, with_taps}};
inline constexpr std::array attention_output{Group{0, 3072, o}};
inline constexpr std::array controller_down{Group{0, 4608, down}};
inline constexpr std::array controller_gates{Group{0, 64, gates}};

inline constexpr std::array local_ends{192U, 216U, 408U, 472U, 536U};
inline constexpr std::array full_ends{192U, 216U, 408U, 472U, 474U, 538U, 540U};
inline constexpr std::array output_ends{144U};
inline constexpr std::array down_ends{4U};
inline constexpr std::array gates_ends{144U, 288U};

inline constexpr std::array cases{
    Case{"local", local, local_ends, 3, 2},
    Case{"local-separate", local_separate, local_ends, 4, 2},
    Case{"full", full, full_ends, 4, 2},
    Case{"attention-output", attention_output, output_ends, 1, 1},
    Case{"controller-down", controller_down, down_ends, 1, 1},
    Case{"controller-gates", controller_gates, gates_ends, 1, 1},
};

}  // namespace k1bind_test

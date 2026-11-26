#pragma once

// Host-only dry binding compares field values. A packet remains in its original ABI layout,
// with the padding between fields zeroed. Nested aggregate arity is deduced at compile time,
// so a new member is visited automatically and an unsupported layout fails the build.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace strix::exec::test {

namespace packet_detail {
struct Placeholder {
    template <class T> operator T() const;  // Used only in an unevaluated aggregate initializer.
};

template <class T, std::size_t... I>
consteval bool constructible(std::index_sequence<I...>) {
    return requires { T{{(static_cast<void>(I), Placeholder{})}...}; };
}

template <class T, std::size_t N = 0>
consteval std::size_t members() {
    if constexpr (constructible<T>(std::make_index_sequence<N + 1>{})) return members<T, N + 1>();
    else return N;
}

// C++20 structured bindings enumerate direct members. Braced aggregate initialization above
// counts a native array as one member. The recursive write visits that array element by element.
template <class T, class Visit>
void fields(const T& packet, Visit visit) {
    constexpr auto count = members<T>();
#define STRIX_CAPTURE_FIELDS(n, ...)                 \
    if constexpr (count == n) {                      \
        const auto& [__VA_ARGS__] = packet;           \
        visit(__VA_ARGS__);                          \
    }
    STRIX_CAPTURE_FIELDS(1, a)
    else STRIX_CAPTURE_FIELDS(2, a, b)
    else STRIX_CAPTURE_FIELDS(3, a, b, c)
    else STRIX_CAPTURE_FIELDS(4, a, b, c, d)
    else STRIX_CAPTURE_FIELDS(5, a, b, c, d, e)
    else STRIX_CAPTURE_FIELDS(6, a, b, c, d, e, f)
    else STRIX_CAPTURE_FIELDS(7, a, b, c, d, e, f, g)
    else STRIX_CAPTURE_FIELDS(8, a, b, c, d, e, f, g, h)
    else STRIX_CAPTURE_FIELDS(9, a, b, c, d, e, f, g, h, i)
    else STRIX_CAPTURE_FIELDS(10, a, b, c, d, e, f, g, h, i, j)
    else STRIX_CAPTURE_FIELDS(11, a, b, c, d, e, f, g, h, i, j, k)
    else STRIX_CAPTURE_FIELDS(12, a, b, c, d, e, f, g, h, i, j, k, l)
    else STRIX_CAPTURE_FIELDS(13, a, b, c, d, e, f, g, h, i, j, k, l, m)
    else STRIX_CAPTURE_FIELDS(14, a, b, c, d, e, f, g, h, i, j, k, l, m, n)
    else STRIX_CAPTURE_FIELDS(15, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)
    else STRIX_CAPTURE_FIELDS(16, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
    else STRIX_CAPTURE_FIELDS(17, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
    else STRIX_CAPTURE_FIELDS(18, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r)
    else STRIX_CAPTURE_FIELDS(19, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)
    else STRIX_CAPTURE_FIELDS(20, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t)
    else STRIX_CAPTURE_FIELDS(21, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u)
    else STRIX_CAPTURE_FIELDS(22, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v)
    else STRIX_CAPTURE_FIELDS(23, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w)
    else STRIX_CAPTURE_FIELDS(24, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x)
#undef STRIX_CAPTURE_FIELDS
    else static_assert(count > 0 && count <= 24, "unsupported graph packet member count");
}

template <class T> struct IsArray : std::false_type {};
template <class T, std::size_t N> struct IsArray<std::array<T, N>> : std::true_type {};

template <class T>
void write(const T& value, std::byte* output) {
    if constexpr (std::is_scalar_v<T> || std::has_unique_object_representations_v<T>) {
        std::memcpy(output, std::addressof(value), sizeof(T));
    } else {
        const auto member = [&](const auto& field) {
            const auto offset = reinterpret_cast<std::uintptr_t>(std::addressof(field)) -
                                reinterpret_cast<std::uintptr_t>(std::addressof(value));
            write(field, output + offset);
        };
        if constexpr (std::is_array_v<T> || IsArray<T>::value) {
            for (const auto& entry : value) member(entry);
        } else {
            static_assert(std::is_aggregate_v<T>, "graph packet needs a declared field walker");
            fields(value, [&](const auto&... field) { (member(field), ...); });
        }
    }
}
}  // namespace packet_detail

template <class Packet>
void canonicalize_packet(std::span<std::byte> bytes) {
    static_assert(std::is_trivially_copyable_v<Packet>);
    if (bytes.size() != sizeof(Packet)) throw std::logic_error("incorrect physical capture packet type");
    Packet value{};
    std::memcpy(std::addressof(value), bytes.data(), sizeof(value));
    std::ranges::fill(bytes, std::byte{});
    packet_detail::write(value, bytes.data());
}

}  // namespace strix::exec::test

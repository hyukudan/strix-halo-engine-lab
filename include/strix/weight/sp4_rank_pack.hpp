#pragma once
// A rank's share of a pinned Strix pack SP4 image (host only, no device types): the backend-neutral part of every
// decode arm that reads tensor-parallel slices of transcoded or calibrated SP4 rows beside an incumbent image.
//  - Cut: which part of a whole pack matrix one rank reads (all of it, an output-row range or an input-column range);
//  - materialize: the rank's bytes of a cut (a view for whole / output cuts, a byte-exact gather for input cuts);
//  - pack_facts + decide: the one decline rule (nothing requested, an engine prerequisite, then the pack checks in
//    order, first failing one wins); a pack is never trusted by name: an empty pin declines.
// The engine supplies the data: the tensor names and the cut of each (its split rule) and the name of its pin switch.
#include "strix/weight/sp4_slice.hpp"
#include "strix/weight/types.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight::sp4_rank {

// A whole matrix is `rows` outputs x `columns` inputs. by_input: columns [first, first + count), else rows.
struct Cut {
    std::uint64_t rows = 0, columns = 0;
    bool by_input = false;
    std::uint64_t first = 0, count = 0;
    [[nodiscard]] std::uint64_t slice_rows() const noexcept { return by_input ? rows : count; }
    [[nodiscard]] std::uint64_t slice_columns() const noexcept { return by_input ? count : columns; }
    [[nodiscard]] std::uint64_t slice_bytes() const noexcept { return slice_rows() * sp4_slice::row_bytes(slice_columns()); }
};
[[nodiscard]] inline Cut whole(std::uint64_t rows, std::uint64_t columns) noexcept { return {rows, columns, false, 0, rows}; }
// Equal shares of `world` ranks; throws std::invalid_argument when the dimension does not split evenly.
[[nodiscard]] inline Cut output_share(std::uint64_t rows, std::uint64_t columns, unsigned rank, unsigned world) {
    if (rank >= world || (rows % world) != 0U) throw std::invalid_argument("sp4 rank cut: outputs do not split over the ranks");
    return {rows, columns, false, rank * (rows / world), rows / world};
}
[[nodiscard]] inline Cut input_share(std::uint64_t rows, std::uint64_t columns, unsigned rank, unsigned world) {
    if (rank >= world || (columns % world) != 0U || (columns / world) % sp4_slice::kBlock != 0U)
        throw std::invalid_argument("sp4 rank cut: inputs do not split over the ranks in whole superblocks");
    return {rows, columns, true, rank * (columns / world), columns / world};
}

// The rank's bytes of `cut` from the whole matrix `bytes`. `scratch` keeps a gathered input slice alive.
[[nodiscard]] inline std::span<const std::uint8_t> materialize(const Cut& cut, std::span<const std::uint8_t> bytes,
                                                               std::vector<std::uint8_t>& scratch) {
    if (cut.by_input) {
        scratch = sp4_slice::input_slice(bytes, cut.rows, cut.columns, cut.first, cut.count);
        return scratch;
    }
    return sp4_slice::output_slice(bytes, cut.rows, cut.columns, cut.first, cut.count);
}

// Why the arm declines (the incumbent then runs unchanged and the receipt line says why).
enum class Outcome : std::uint8_t {
    on, off, prerequisite, no_pack, open_failed, identity_unpinned, identity_malformed, identity_mismatch, payload_mismatch,
    entry_missing, entry_shape, not_sp4t
};
// `prerequisite` is the engine's own condition: its text comes from the engine.
[[nodiscard]] inline const char* outcome_text(Outcome o) noexcept {
    switch (o) {
    case Outcome::on: return "on";
    case Outcome::off: return "off";
    case Outcome::prerequisite: return "declined: engine prerequisite off";
    case Outcome::no_pack: return "declined: pack file absent";
    case Outcome::open_failed: return "declined: pack unreadable or not a strix-lowbit-pack";
    case Outcome::identity_unpinned: return "declined: no pinned pack identity";
    case Outcome::identity_malformed: return "declined: pinned pack identity is not 64 hex digits";
    case Outcome::identity_mismatch: return "declined: pack identity differs from the pinned one";
    case Outcome::payload_mismatch: return "declined: pack payload does not hash to its identity";
    case Outcome::entry_missing: return "declined: a projection has no pack entry";
    case Outcome::entry_shape: return "declined: a pack entry has the wrong shape or byte count";
    case Outcome::not_sp4t: return "declined: a pack entry is not fixed-table SP4";
    }
    return "declined";
}

struct PackFacts {
    bool exists = false, opened = false;
    bool pin_set = false, pin_wellformed = false, identity_equal = false, payload_ok = false;
    bool entries_present = false, entries_shaped = false, entries_sp4t = false;
};
[[nodiscard]] inline Outcome decide(bool requested, bool prerequisite_ok, const PackFacts& f) noexcept {
    if (!requested) return Outcome::off;
    if (!prerequisite_ok) return Outcome::prerequisite;
    if (!f.exists) return Outcome::no_pack;
    if (!f.opened) return Outcome::open_failed;
    if (!f.pin_set) return Outcome::identity_unpinned;
    if (!f.pin_wellformed) return Outcome::identity_malformed;
    if (!f.identity_equal) return Outcome::identity_mismatch;
    if (!f.payload_ok) return Outcome::payload_mismatch;
    if (!f.entries_present) return Outcome::entry_missing;
    if (!f.entries_shaped) return Outcome::entry_shape;
    if (!f.entries_sp4t) return Outcome::not_sp4t;
    return Outcome::on;
}

[[nodiscard]] inline std::string hex(const Digest256& d) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * d.bytes.size());
    for (const auto b : d.bytes) {
        out += kDigits[b >> 4U];
        out += kDigits[b & 15U];
    }
    return out;
}
// 64 hex digits -> digest; false when malformed (or all zero: never a pack identity).
[[nodiscard]] inline bool parse_pin(std::string_view text, Digest256& out) noexcept {
    const auto nibble = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
    };
    out = Digest256{};
    if (text.size() != 2 * out.bytes.size()) return false;
    bool nonzero = false;
    for (std::size_t i = 0; i < out.bytes.size(); ++i) {
        const int high = nibble(text[2 * i]), low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        out.bytes[i] = static_cast<std::uint8_t>(high * 16 + low);
        nonzero = nonzero || out.bytes[i] != 0U;
    }
    return nonzero;
}

// One tensor the arm needs and the cut the rank reads of it.
struct Need {
    std::string tensor;
    Cut cut;
};

// The facts of an inspected pack against the pin and the needs. `Pack` is strix::weight::LowbitPack (identity(),
// verify_identity(), entry(name) -> rows / columns / bytes / storage / sp_header); `sp4t` is the fixed-table SP4 format.
// The payload is hashed only once the identity equals the pin.
template <class Pack, class Format>
[[nodiscard]] PackFacts pack_facts(bool exists, bool opened, const Pack& pk, std::string_view pin, std::span<const Need> needs,
                                   const Format& sp4t) {
    PackFacts f;
    f.exists = exists;
    f.opened = exists && opened;
    if (!f.opened) return f;
    Digest256 pinned{};
    f.pin_set = !pin.empty();
    f.pin_wellformed = f.pin_set && parse_pin(pin, pinned);
    f.identity_equal = f.pin_wellformed && pinned.bytes == pk.identity().bytes;
    f.payload_ok = f.identity_equal && static_cast<bool>(pk.verify_identity());
    f.entries_present = f.entries_shaped = f.entries_sp4t = true;
    for (const Need& n : needs) {
        const auto* e = pk.entry(n.tensor);
        if (e == nullptr) {
            f.entries_present = false;
            continue;
        }
        if (e->rows != n.cut.rows || e->columns != n.cut.columns || !sp4_slice::admitted(e->bytes, e->rows, e->columns))
            f.entries_shaped = false;
        if (!(e->storage == sp4t) || !e->sp_header) f.entries_sp4t = false;
    }
    return f;
}

// Device bytes of all the rank's slices (each placed at a 256-byte aligned offset).
[[nodiscard]] inline std::uint64_t arena_bytes(std::span<const Need> needs) noexcept {
    std::uint64_t total = 0;
    for (const Need& n : needs) total += (n.cut.slice_bytes() + 255U) & ~std::uint64_t{255};
    return total;
}

}  // namespace strix::weight::sp4_rank

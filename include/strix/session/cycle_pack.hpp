#pragma once
// One completion per serve cycle (STRIX_SERVE_CYCLE_PACK).
//
// A batched decode cycle ends with small device records per member (status words, first-error records, union
// summaries). Read one by one, every record costs a copy, a host wait and a wake-up, member after member. A
// CyclePack collects the records of one cycle: add() reserves a slot per record, gather() copies all of them into
// one mapped host block with ONE kernel launch, wait() fences the stream ONCE, and the members then check their
// records from host memory. Scheduling only: the same records reach the same host checks, so tokens are unchanged.
//
// Shared by every batch runner (Qwen cohort window first). The header is host-only (streams are void*), so engines,
// servers and host tests include it without HIP.

#include "strix/lever.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strix::session {

inline constexpr lever::Decl serve_cycle_pack_lever{
    "serve-cycle-pack", "STRIX_SERVE_CYCLE_PACK", lever::Exactness::bitwise, false, {}, {},
    "one packed completion per batched decode cycle: same records, same checks, one wait"};

// The cycle's concurrency is the regime axis (members in the cycle).
[[nodiscard]] inline lever::Choice serve_cycle_pack_choice(std::uint64_t members) {
    return lever::select(serve_cycle_pack_lever, lever::Inputs{0, 0, members});
}

// Word layout of one cycle's records: each record is a 4-byte multiple placed at the next free word.
struct CyclePackLayout {
    static constexpr std::uint32_t max_records = 64;
    static constexpr std::uint32_t max_words = 2048;  // 8 KiB of records per cycle
    std::uint32_t records = 0;
    std::uint32_t words = 0;
    std::array<std::uint32_t, max_records> offset{};
    std::array<std::uint32_t, max_records> size{};

    void reset() noexcept { records = 0; words = 0; }
    // Returns the record's first word.
    std::uint32_t add(std::size_t bytes) {
        if (bytes == 0 || bytes % 4U != 0) throw std::invalid_argument("cycle pack record must be a non-empty 4-byte multiple");
        if (records == max_records) throw std::length_error("cycle pack holds at most 64 records per cycle");
        const auto n = static_cast<std::uint32_t>(bytes / 4U);
        if (n > max_words - words) throw std::length_error("cycle pack holds at most 8 KiB of records per cycle");
        offset[records] = words;
        size[records] = n;
        ++records;
        words += n;
        return words - n;
    }
};

// Device side (src/session/cycle_pack.hip). One pack per runner: records of one cycle at a time.
class CyclePack {
public:
    CyclePack();
    ~CyclePack();
    CyclePack(const CyclePack&) = delete;
    CyclePack& operator=(const CyclePack&) = delete;

    // Starts a cycle: drops the previous cycle's records.
    void begin() noexcept;
    // Reserves a slot for `bytes` (4-byte multiple) read from `device_source`; returns the host address the record
    // lands at. The slot is poisoned (0xA5A5A5A5 words) until gather() overwrites it and is valid after wait().
    [[nodiscard]] const void* add(const void* device_source, std::size_t bytes);
    // Copies every record of the cycle into the mapped host block: one kernel on `stream`.
    void gather(void* stream);
    // The single host wait of the cycle: synchronises `stream` (which must carry gather()).
    void wait(void* stream);
    [[nodiscard]] std::uint32_t records() const noexcept;
    // Counters for receipts: cycles packed and records gathered since creation.
    [[nodiscard]] std::uint64_t cycles() const noexcept;
    [[nodiscard]] std::uint64_t gathered() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::session

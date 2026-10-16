#include "strix/arithmetic_lineage.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {
void hash_word(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 255ULL;
        hash *= 1099511628211ULL;
    }
}
} // namespace

std::uint64_t arithmetic_history_identity(
    std::span<const ArithmeticRowInterval> intervals) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    hash_word(hash, 0x53545249584c494eULL); // lineage schema domain
    hash_word(hash, intervals.size());
    for (const auto& interval : intervals) {
        hash_word(hash, interval.begin);
        hash_word(hash, interval.end);
        hash_word(hash, static_cast<std::uint64_t>(interval.producer));
        hash_word(hash, interval.numerical_plan_identity);
        hash_word(hash, interval.reduction_plan_identity);
        hash_word(hash, interval.state_update_identity);
    }
    return hash;
}

bool arithmetic_lineage_valid(const ArithmeticLineage& lineage,
                              std::uint64_t resume_position) noexcept {
    if (lineage.tainted || lineage.history_identity == 0 || lineage.intervals.empty() ||
        lineage.history_identity != arithmetic_history_identity(lineage.intervals)) {
        return false;
    }
    std::uint64_t next = 0;
    for (const auto& interval : lineage.intervals) {
        if (interval.begin != next || interval.end <= interval.begin ||
            interval.end > resume_position || interval.numerical_plan_identity == 0 ||
            interval.reduction_plan_identity == 0 || interval.state_update_identity == 0 ||
            (interval.producer != ArithmeticProducer::serial_row &&
             interval.producer != ArithmeticProducer::prefill_chunk)) {
            return false;
        }
        next = interval.end;
    }
    return next == resume_position;
}

bool arithmetic_lineage_matches(const ArithmeticLineage& recorded,
                                const ArithmeticLineage& requested,
                                std::uint64_t resume_position) noexcept {
    return arithmetic_lineage_valid(recorded, resume_position) &&
        arithmetic_lineage_valid(requested, resume_position) &&
        recorded.history_identity == requested.history_identity &&
        recorded.intervals == requested.intervals;
}

bool arithmetic_lineage_serial_only(const ArithmeticLineage& lineage,
                                    std::uint64_t execution_identity,
                                    std::uint64_t resume_position) noexcept {
    if (!arithmetic_lineage_valid(lineage, resume_position) || execution_identity == 0) {
        return false;
    }
    for (const auto& interval : lineage.intervals) {
        if (interval.producer != ArithmeticProducer::serial_row ||
            interval.numerical_plan_identity != execution_identity ||
            interval.state_update_identity != execution_identity) {
            return false;
        }
    }
    return true;
}

ArithmeticLineage prepare_prefill_chunk_lineage(
    const ArithmeticLineage& previous, std::uint64_t begin, std::uint64_t end,
    std::uint64_t numerical_plan_identity, std::uint64_t reduction_plan_identity,
    std::uint64_t state_update_identity) {
    if (end <= begin || !numerical_plan_identity || !reduction_plan_identity ||
        !state_update_identity) {
        throw std::invalid_argument("prefill lineage requires a bounded chunk and exact identities");
    }
    if (previous.tainted) {
        throw std::invalid_argument("tainted arithmetic lineage requires independently valid restore");
    }
    if (previous.history_identity == 0 && previous.intervals.empty()) {
        if (begin != 0) return {};
    } else if (!arithmetic_lineage_valid(previous, begin)) {
        throw std::invalid_argument("prefill arithmetic lineage does not cover the current position");
    }
    auto next = previous;
    next.intervals.push_back({begin, end, ArithmeticProducer::prefill_chunk,
        numerical_plan_identity, reduction_plan_identity, state_update_identity});
    next.history_identity = arithmetic_history_identity(next.intervals);
    return next;
}

ArithmeticLineage prepare_serial_row_lineage(
    const ArithmeticLineage& previous, std::uint64_t position,
    std::uint64_t execution_identity, std::uint64_t reduction_plan_identity) {
    if (execution_identity == 0 || reduction_plan_identity == 0 ||
        position == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument("serial arithmetic lineage requires bounded position and exact arithmetic identities");
    }
    if (previous.tainted) {
        throw std::invalid_argument("tainted arithmetic lineage requires independently valid restore");
    }
    if (previous.history_identity == 0 && previous.intervals.empty()) {
        if (position != 0) { return {}; } // preserve execution, prohibit cache reuse
    } else if (!arithmetic_lineage_valid(previous, position)) {
        throw std::invalid_argument("serial arithmetic lineage does not cover the current position");
    }
    auto next = previous;
    ArithmeticRowInterval interval{position, position + 1, ArithmeticProducer::serial_row,
                                   execution_identity, reduction_plan_identity, execution_identity};
    if (!next.intervals.empty()) {
        auto& last = next.intervals.back();
        if (last.producer == interval.producer &&
            last.numerical_plan_identity == interval.numerical_plan_identity &&
            last.reduction_plan_identity == interval.reduction_plan_identity &&
            last.state_update_identity == interval.state_update_identity) {
            last.end = interval.end;
        } else { next.intervals.push_back(interval); }
    } else { next.intervals.push_back(interval); }
    next.history_identity = arithmetic_history_identity(next.intervals);
    return next;
}
} // namespace strix

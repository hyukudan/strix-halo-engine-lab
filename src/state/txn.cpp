#include "strix/state/txn.hpp"

#include "strix/session/kv_row_append.hpp"

#include <cstring>
#include <stdexcept>

namespace strix::state {
namespace {

std::uint64_t input_base(const InstanceLayout& piece, std::uint32_t slot, std::uint32_t row) {
    return piece.slot_offset(slot) + piece.input_offset + std::uint64_t{row} * piece.replay_input_bytes;
}

}  // namespace

Txn::Txn(CpuState& state, std::uint32_t slot) : state_(state), slot_(slot) {
    (void)state_.entry(0, slot_);
}

Txn::~Txn() { abort(); }

void Txn::begin(std::uint32_t rows) {
    const auto& stored = state_.entry(0, slot_);
    if (stored.pending) throw std::logic_error("state slot already has pending rows");
    const session::KvAppendChunk chunk{stored.frontier, rows};
    const auto& plan = state_.plan();
    if (!chunk.rows || chunk.rows > plan.max_pending || chunk.rows > stored.capacity - chunk.first_position)
        throw std::invalid_argument("state transaction exceeds admitted rows");
    for (std::uint32_t i = 0; i < plan.instances.size(); ++i) state_.mutable_entry(i, slot_).pending = rows;
    pending_ = rows;
}

std::uint64_t Txn::candidate(const InstanceLayout& piece, const SlotEntry& stored, std::uint32_t row) const {
    const auto start = piece.slot_offset(slot_);
    const auto committed = (stored.base - start) / piece.row_stride;
    std::uint64_t next = 1U + row % 2U;
    if (piece.rollback == Rollback::snapshot) next = std::uint64_t{row} + 1U;
    else if (piece.checkpoint_interval && (row + 1U) % piece.checkpoint_interval == 0)
        next = 3U + row / piece.checkpoint_interval;
    return start + ((committed + next) % piece.rows) * piece.row_stride;
}

void Txn::update(std::uint32_t state, std::uint32_t position, std::uint64_t previous, Row input,
                 std::uint64_t next) {
    const auto bytes = state_.plan().instances[state].payload_bytes();
    state_.updates_[state](position, state_.bytes(previous, bytes), input, state_.bytes(next, bytes));
}

void Txn::append(std::span<const Row> inputs) {
    const auto& pieces = state_.plan().instances;
    if (written_ >= pending_) throw std::logic_error("no unwritten pending state row");
    if (pieces.size() != inputs.size()) throw std::invalid_argument("wrong number of state epilogues");
    for (std::uint32_t i = 0; i < pieces.size(); ++i) {
        const auto& piece = pieces[i];
        const auto size = inputs[i].size();
        if (piece.storage != i || piece.rollback == Rollback::recompute) {
            if (size) throw std::invalid_argument("aliased/recomputed state takes no input bytes");
        } else if ((piece.rollback == Rollback::rewind && size != piece.payload_bytes()) ||
                   (piece.rollback == Rollback::replay && size != piece.replay_input_bytes)) {
            throw std::invalid_argument("state epilogue image has the wrong size");
        }
    }

    const auto position = state_.entry(0, slot_).frontier + written_;
    try {
        for (std::uint32_t i = 0; i < pieces.size(); ++i) {
            const auto& piece = pieces[i];
            if (piece.storage != i) continue;
            const auto& stored = state_.entry(i, slot_);
            auto input = inputs[i];
            if (piece.rollback == Rollback::rewind) {
                const auto offset = state_.append_base(piece, stored, position);
                const auto target = state_.bytes(offset, input.size() + sizeof(std::uint32_t));
                const auto tag = position + 1U;
                std::memmove(target.data(), input.data(), input.size());
                std::memcpy(target.data() + input.size(), &tag, sizeof(tag));
                continue;
            }
            if (piece.rollback == Rollback::replay && !input.empty()) {
                const auto target = state_.bytes(input_base(piece, slot_, written_), input.size());
                std::memmove(target.data(), input.data(), input.size());
                input = target;
            }
            const auto previous = written_ ? candidate(piece, stored, written_ - 1U) : stored.base;
            update(i, position, previous, input, candidate(piece, stored, written_));
        }
        ++written_;
    } catch (...) {
        abort();
        throw;
    }
}

void Txn::prepare_accept(std::uint32_t state, std::uint32_t accepted) {
    const auto& piece = state_.plan().instances[state];
    if (piece.storage != state || (piece.rollback != Rollback::replay && piece.rollback != Rollback::recompute))
        return;
    const auto& stored = state_.entry(state, slot_);
    const auto checkpoint = piece.checkpoint_interval ? accepted - accepted % piece.checkpoint_interval : 0U;
    auto previous = checkpoint ? candidate(piece, stored, checkpoint - 1U) : stored.base;
    for (auto row = checkpoint; row < accepted; ++row) {
        const auto next = candidate(piece, stored, row);
        const auto input = piece.rollback == Rollback::replay
            ? state_.bytes(input_base(piece, slot_, row), piece.replay_input_bytes) : Row{};
        update(state, stored.frontier + row, previous, input, next);
        previous = next;
    }
}

void Txn::commit(std::uint32_t accepted) {
    if (!pending_ || accepted > written_) throw std::logic_error("cannot commit unwritten state rows");
    if (accepted == 0) {
        abort();
        return;
    }
    const auto& pieces = state_.plan().instances;
    try {
        if (accepted < written_)
            for (std::uint32_t i = 0; i < pieces.size(); ++i) prepare_accept(i, accepted);
    } catch (...) {
        abort();
        throw;
    }
    // Resolve aliases against the old bases before changing their producers.
    for (std::uint32_t i = 0; i < pieces.size(); ++i) {
        const auto& producer = pieces[pieces[i].storage];
        if (producer.rollback != Rollback::rewind)
            state_.mutable_entry(i, slot_).base = candidate(producer, state_.entry(i, slot_), accepted - 1U);
    }
    for (std::uint32_t i = 0; i < pieces.size(); ++i) state_.mutable_entry(i, slot_).frontier += accepted;
    abort();
}

void Txn::abort() noexcept {
    if (!pending_) return;
    for (std::uint32_t i = 0; i < state_.plan().instances.size(); ++i) state_.mutable_entry(i, slot_).pending = 0;
    pending_ = 0;
    written_ = 0;
}

}  // namespace strix::state

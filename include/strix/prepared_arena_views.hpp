#pragma once

#include "strix/execution_runtime.hpp"

#include <cstddef>
#include <memory>

namespace strix {
struct CompiledSerialExecution;
class PreparedEagerExecution;
enum class PreparedBufferKind { read, write, temporary };

// Immutable allocation indices over the prepared execution's private arena.
// A checked metadata lookup; never permission to dispatch or commit. The
// caller retains the prepared execution through device quiescence.
class PreparedArenaViews final {
public:
    ~PreparedArenaViews();
    PreparedArenaViews(const PreparedArenaViews&) = delete;
    PreparedArenaViews& operator=(const PreparedArenaViews&) = delete;
    PreparedArenaViews(PreparedArenaViews&&) = delete;
    PreparedArenaViews& operator=(PreparedArenaViews&&) = delete;

    [[nodiscard]] ExecutionBufferAddress resolve(const ExecutionDispatchContext& context,
        PreparedBufferKind kind, std::size_t index, const ExecutionMemoryBinding& memory) const;

private:
    friend class PreparedEagerExecution;
    explicit PreparedArenaViews(const CompiledSerialExecution& immutable_owned);
    struct Impl;
    std::unique_ptr<const Impl> impl_;
};
} // namespace strix

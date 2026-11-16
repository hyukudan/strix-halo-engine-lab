#pragma once
// Shared VerifyTarget over any engine with the window contract (CUDA CudaEngine, HIP HipEngine):
//   forward_rows(tokens, rows, LogitRows) -> Out; read_top(Out, rows, host) (synchronizes); commit(k); sync().
// verify = forward_rows(all) + per-row argmax; commit(k) = engine commit + sync (the step wall is complete).
// Home: shared (host spec layer): no device types; each backend binds it with its own engine type.
#include <cstdint>
#include <span>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

template <class Engine>
class EngineVerifyTarget final : public VerifyTarget {
public:
    explicit EngineVerifyTarget(Engine& e) : e_(e) {}
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto out = e_.forward_rows(tokens.data(), rows, Engine::LogitRows::all);
        greedy.resize(rows);
        e_.read_top(out, rows, greedy.data());
    }
    void commit(std::uint32_t k) override {
        e_.commit(k);
        e_.sync();
    }
    Engine& engine() { return e_; }

private:
    Engine& e_;
};

}  // namespace strix::spec

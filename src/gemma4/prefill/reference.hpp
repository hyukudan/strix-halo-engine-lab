#pragma once
#include "emulator.hpp"
#include "attention.hpp"
#include "gemma4/ref/target.hpp"

namespace strix::gemma4::prefill {
// Slow independent host graph for synthetic identity gates and explicit offline
// references. Full-model work is never a ctest. Model/table owners outlive it.
class Reference {
  public:
    Reference(HostWeights weights, RopeTables tables, unsigned admitted_context, AttentionMode attention = attention_mode());
    ~Reference();
    Reference(const Reference&) = delete;
    Reference& operator=(const Reference&) = delete;
    // Commits the complete supplied chunk after successful evaluation. matrix=false
    // uses declared g4-v1 row dots for decode from this same prefilled state.
    ChunkResult forward(std::span<const std::uint32_t> tokens, bool matrix = true, bool all_logits = false);
    Result run(std::span<const std::uint32_t> tokens);
    [[nodiscard]] std::uint64_t frontier() const;
    [[nodiscard]] std::string digest() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::gemma4::prefill

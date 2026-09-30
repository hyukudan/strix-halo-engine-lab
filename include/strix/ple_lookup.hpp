#pragma once

#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

struct PleHashContract {
    static constexpr std::size_t maximum_heads = 32;
    static constexpr std::size_t maximum_ngram = 8;

    std::uint32_t ngram_size = 0;
    std::uint32_t heads_per_ngram = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t eos_token_id = 0;
    std::uint64_t row_bytes = 0;
    std::array<std::uint64_t, maximum_ngram> multipliers{};
    std::array<std::uint64_t, maximum_heads> head_offsets{};
    std::array<std::uint64_t, maximum_heads> head_vocab_sizes{};
    std::uint64_t minimum_table_rows = 0;
};

[[nodiscard]] PleHashContract build_ple_hash_contract(
    const ModelContract& model);

struct PleHashResult {
    std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
    std::uint32_t row_count = 0;
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> next_context{};
    std::uint32_t context_count = 0;
};

[[nodiscard]] PleHashResult hash_ple_token(
    const PleHashContract& contract,
    std::uint32_t token_id,
    std::span<const std::uint32_t> previous_context);

using PleLookupHandle = std::uint64_t;

struct PleLookupRequest {
    std::span<const std::uint64_t> row_ids;
    std::uint64_t row_bytes = 0;
    const ManifestTensor* table_manifest = nullptr;
    const RuntimeTensorBinding* table_runtime = nullptr;
};

class PleLookupBackend {
public:
    virtual ~PleLookupBackend() = default;

    [[nodiscard]] virtual std::uint64_t implementation_identity() const noexcept = 0;

    [[nodiscard]] virtual PleLookupHandle start(
        const PleLookupRequest& request) = 0;

    virtual void finish(
        PleLookupHandle handle,
        std::span<std::byte> destination) = 0;

    virtual void cancel(
        PleLookupHandle handle) noexcept = 0;
};

struct PleLookupSessionState {
    PleLookupHandle handle = 0;
    bool active = false;
    std::uint64_t sequence = 0;
    std::array<std::uint64_t, PleHashContract::maximum_heads> row_ids{};
    std::uint32_t row_count = 0;
};

}  // namespace strix

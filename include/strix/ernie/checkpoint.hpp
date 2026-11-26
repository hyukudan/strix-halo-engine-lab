#pragma once

#include "strix/ernie/inventory.hpp"

#include <functional>

namespace strix::ernie {

enum class TensorConsumer {
    first_row_reference, hca_history_candidate, mla_layer_candidate, multimax_pending,
    draft_pending, unknown
};

[[nodiscard]] std::string_view consumer_name(TensorConsumer consumer) noexcept;
[[nodiscard]] TensorConsumer tensor_consumer(std::string_view name, const Tensor& tensor) noexcept;

struct PayloadTotal {
    std::size_t tensors{};
    std::uint64_t bytes{};
};
struct PairPayload {
    PayloadTotal rank0, rank1, draft;
};

// Header-only placement accounting independent of numerical admission. Output
// globals go on the last rank and token embeddings on the first.
[[nodiscard]] PairPayload pair_payload(const Inventory& inventory, unsigned cut);

// Header-derived lazy weight source for the target reference. Quantized
// matrices are requested whole (two planes decoded through shared codecs).
// Plain tensors also support ranged reads, e.g. one embedding row.
struct CheckpointReadPolicy {
    bool table_mxfp4 = false;
    bool high_nibble_first = false;
    bool transpose_experts = false;
    bool transpose_fp8_scales = false;
};

class CheckpointWeights {
public:
    using RowRange = std::function<void(std::size_t first, std::size_t last)>;
    using RowExecutor = std::function<void(std::size_t rows, const RowRange&)>;

    explicit CheckpointWeights(Inventory inventory, CheckpointReadPolicy policy = {});
    [[nodiscard]] const Inventory& inventory() const { return inventory_; }
    // An executor may partition independent decoder rows. It joins all ranges
    // before returning or throwing.
    void read(std::string_view name, std::span<float> out, std::size_t first_element = 0,
              const RowExecutor& execute = {}) const;
private:
    Inventory inventory_;
    CheckpointReadPolicy policy_;
};

} // namespace strix::ernie

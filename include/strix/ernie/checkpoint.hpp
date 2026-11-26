#pragma once

#include "strix/ernie/inventory.hpp"

namespace strix::ernie {

enum class TensorConsumer {
    first_row_reference, hca_history_pending, mla_layer_pending, multimax_pending,
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

// Header-derived lazy weight source for the first-row reference. Quantized
// matrices are requested whole (two planes decoded through shared codecs).
// Plain tensors also support ranged reads, e.g. one embedding row.
class CheckpointWeights {
public:
    explicit CheckpointWeights(Inventory inventory);
    [[nodiscard]] const Inventory& inventory() const { return inventory_; }
    void read(std::string_view name, std::span<float> out, std::size_t first_element = 0) const;
private:
    Inventory inventory_;
};

} // namespace strix::ernie

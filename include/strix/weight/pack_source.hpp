#pragma once

#include "strix/weight/admission.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"

namespace strix::weight {

// One file (safetensors or GGUF), a checkpoint directory, or its safetensors index JSON. Opening
// indexes headers only; content identity is an explicit full checkpoint read.
class PackCheckpoint {
public:
    static Status open(const std::string& path, PackCheckpoint& out);
    [[nodiscard]] const std::vector<RawTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const RawTensor* find(std::string_view name) const;
    Status digest(Digest256& out) const;

private:
    std::shared_ptr<const Source> manifest_;
    std::vector<std::shared_ptr<const Source>> shards_;
    std::vector<RawTensor> tensors_;  // sorted by name
};

struct PackMatrixSource {
    // q8_0: a GGUF Q8_0 matrix (ggml type 8, rows of K/32 blocks of 34 bytes: FP16 d, 32 int8), codes = the whole
    // payload, no separate scale plane; value = float(d) * q (exact in FP32), the engines' Q8_0 dequantisation.
    enum class Encoding { bf16, mxfp4, fp8_block, nvfp4, mxfp8, exl3, q8_0 };
    Encoding encoding{};
    std::size_t rows{}, columns{};
    Slice codes, scales;
    Slice amax{}, denominator{};
    microscaled::RowOrder row_order{};
    // Row window of a larger microscaled matrix: `rows` logical rows starting at `first_row`
    // of a matrix with `matrix_rows` rows (0 means the window is the whole matrix).
    std::size_t first_row{}, matrix_rows{};
    // exl3 (a third-party trellis linear, formats/exl3_dense.hpp): `codes` are the trellis rings (input-tile-major,
    // 32 * K bytes per 16 x 16 tile, K = stored width / 16), `scales` the binary16 input scales suh (one per column)
    // and `output_scales` the binary16 output scales svh (one per row). rows = outputs, columns = inputs, both whole
    // 128-element blocks. decode() returns the dense FP32 weight the device computes. A tensor "<base>.trellis" binds
    // together with "<base>.suh" and "<base>.svh".
    Slice output_scales{};

    static Status bind(const PackCheckpoint&, const RawTensor&, PackMatrixSource&);
    static Status bind(const microscaled::MatrixView&, microscaled::RowOrder, std::size_t first,
                       std::size_t count, PackMatrixSource&);
    // When requested, hash the exact bytes already staged for this decode.
    Status decode(std::vector<float>& out, Digest256* digest = nullptr) const;
    Status digest(Digest256& out) const;
};

// Bind one rank-3 microscaled expert bank by its checkpoint plane names.
Status bind_pack_bank(const PackCheckpoint&, const RawTensor&, microscaled::RowOrder,
                      microscaled::ExpertSlabView&);
Status pack_expert(const microscaled::ExpertSlabView&, std::size_t expert, PackMatrixSource&);
std::string packed_expert_name(std::string_view bank, std::size_t expert);

}  // namespace strix::weight

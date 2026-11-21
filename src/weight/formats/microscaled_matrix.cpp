#include "strix/weight/formats/microscaled_matrix.hpp"

#include "strix/weight/tensor_access.hpp"

namespace strix::weight::microscaled {

Status ExpertSlabView::bind_sources(const TensorLookup& tensors, const ExpertSlabNames& names) {
    ExpertSlabView bound{format, row_order, experts, rows, columns, {}, {}, {}, {}};
    if (const auto status = tensors.slice(names.codes, bound.codes); !status) return status;
    if (const auto status = tensors.slice(names.block_scales, bound.block_scales); !status) return status;
    if (format == MatrixFormat::nvfp4) {
        if (const auto status = tensors.slice(names.amax, bound.amax); !status) return status;
        if (const auto status = tensors.slice(names.denominator, bound.denominator); !status) return status;
    }
    std::size_t code_stride = 0, scale_stride = 0;
    if (const auto status = bound.strides(code_stride, scale_stride); !status) return status;
    *this = std::move(bound);
    return {Error::ok, {}};
}

}  // namespace strix::weight::microscaled

#pragma once

#include "intel/arch/xe2/caps.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>

namespace strix::intel::detail {

namespace matrix = sycl::ext::oneapi::experimental::matrix;

inline bool query_xmx(const sycl::device& device) {
    using namespace arch::xe2;
    using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
    try {
        if (!device.has(sycl::aspect::fp16)) return false;
        const auto widths = device.get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(widths.begin(), widths.end(), kSubgroup) == widths.end()) return false;
        for (const auto& shape : device.get_info<matrix_combinations>()) {
            using matrix::matrix_type;
            if (shape.msize == kMatM && shape.nsize == kMatN && shape.ksize == kMatK &&
                shape.atype == matrix_type::fp16 && shape.btype == matrix_type::fp16 &&
                shape.ctype == matrix_type::fp32 && shape.dtype == matrix_type::fp32)
                return true;
        }
    } catch (const sycl::exception&) {
        // A driver without the optional query can still execute reference kernels.
    }
    return false;
}

struct XmxLinear {
    const sycl::half* input;
    const sycl::half* packed_weights;
    float* output;
    std::size_t rows, width;

    [[sycl::reqd_sub_group_size(arch::xe2::kSubgroup)]]
    void operator()(sycl::nd_item<2> item) const {
        using namespace arch::xe2;
        const auto group = item.get_sub_group();
        const auto m = item.get_group(0) * kMatM;
        const auto n = item.get_group(1) * kMatN;
        using matrix::layout;
        using matrix::use;
        matrix::joint_matrix<sycl::sub_group, sycl::half, use::a, kMatM, kMatK, layout::row_major> a;
        matrix::joint_matrix<sycl::sub_group, sycl::half, use::b, kMatK, kMatN, layout::ext_intel_packed> b;
        matrix::joint_matrix<sycl::sub_group, float, use::accumulator, kMatM, kMatN> c;
        matrix::joint_matrix_fill(group, c, 0.0F);
        for (std::size_t k = 0; k < width; k += kMatK) {
            const auto* a_base = input + m * width + k;
            const auto* b_base = packed_weights + packed_b_offset(k, n, rows);
            matrix::joint_matrix_load(group, a, sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(a_base), width);
            matrix::joint_matrix_load(group, b, sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(b_base), rows * 2U);
            matrix::joint_matrix_mad(group, c, a, b, c);
        }
        matrix::joint_matrix_store(group, c, sycl::address_space_cast<sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>(output + m * rows + n), rows, layout::row_major);
    }
};

} // namespace strix::intel::detail

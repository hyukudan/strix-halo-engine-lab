#pragma once

#include "strix/exec/descriptor.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRIX_STEP_ROWS_HD __host__ __device__
#else
#define STRIX_STEP_ROWS_HD
#endif

// Keep StepRows in the namespace used by the incumbent CUDA kernels.
namespace strix::cuda::core {

template <unsigned Rows>
struct StepRows {
    static constexpr unsigned width = Rows;
    const exec::StepBlock* descriptor;

    STRIX_STEP_ROWS_HD bool valid(unsigned row) const {
        return row < Rows && exec::step_row_valid(*descriptor, row);
    }

    STRIX_STEP_ROWS_HD unsigned position(unsigned row) const {
        return exec::step_row(*descriptor, row).position;
    }
};

template <class Status, class F>
constexpr Status dispatch_step_rows(unsigned width, Status invalid, F launch) {
    switch (width) {
    case 1: return launch.template operator()<1>();
    case 2: return launch.template operator()<2>();
    case 4: return launch.template operator()<4>();
    case 8: return launch.template operator()<8>();
    case 16: return launch.template operator()<16>();
    default: return invalid;
    }
}

}  // namespace strix::cuda::core

#undef STRIX_STEP_ROWS_HD

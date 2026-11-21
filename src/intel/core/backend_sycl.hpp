#pragma once

#include "backend.hpp"
#include <sycl/sycl.hpp>

namespace strix::intel::detail {

// For shared Intel kernel bodies only. The public resource handle is opaque.
sycl::queue& native_queue(Queue queue);

} // namespace strix::intel::detail

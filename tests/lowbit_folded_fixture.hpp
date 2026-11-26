#pragma once

#include <cstddef>

namespace strix::test {

template <class Poison, class Launch>
auto launch_poisoned_folded_group(float* first, float* second, std::size_t count,
                                  const Poison& poison, const Launch& launch) {
    poison(first, count);
    poison(second, count);
    return launch();
}

}  // namespace strix::test

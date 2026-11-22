#include "recur_cases.hpp"

#include "strix/graph/model_decl.hpp"

namespace strix::cuda::recur::test {

state::Layout layout(const std::array<state::Decl, 2>& decls, std::uint32_t steps) {
    return state::layout(decls, graph::ModelDecl{}, 256, 1, {.max_pending = steps});
}

}  // namespace strix::cuda::recur::test

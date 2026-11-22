// Serve engine row for Inkling 2 Mini (cmake: strix_serve_register_engine). The build compiles this
// file into the one device library it selected, with STRIX_INKLING2_SERVE_HIP or _CUDA, so the shell
// names no model and no backend.
#include "inkling2_mini_serve.hpp"

#include <utility>

namespace strix::inkling2::serve_engine {

void install_session_service(::strix::serve::ServiceFactories& factories) {
    MiniServeBackends backends;
#if defined(STRIX_INKLING2_SERVE_HIP)
    backends.hip = make_mini_hip_cohort;
#endif
#if defined(STRIX_INKLING2_SERVE_CUDA)
    backends.cuda = make_mini_cuda_cohort;
#endif
    install_mini_service(factories, std::move(backends));
}

} // namespace strix::inkling2::serve_engine

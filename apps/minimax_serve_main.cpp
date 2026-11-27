// Compatibility spelling; registry construction and ownership use the common shell.
#include "minimax/serve/session_service.hpp"
#ifdef STRIX_MINIMAX_SERVE_SHIPPED_DATA
#include "strix/cli/cli.hpp"
#endif
int main(int argc, char** argv) {
#ifdef STRIX_MINIMAX_SERVE_SHIPPED_DATA
    // Same data directory as `strix serve` and strix-serve: the shipped share/strix next to the executable.
    return strix::minimax::serve::legacy_serve(argc, argv, strix::cli::data_directory(strix::cli::executable_directory()));
#else
    return strix::minimax::serve::legacy_serve(argc, argv);
#endif
}

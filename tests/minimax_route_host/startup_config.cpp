#include "minimax/engine/mm_defaults.hpp"
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#ifdef STRIX_TEST_MINIMAX_SERVE
namespace defaults = strix::minimax::defaults;
struct ServePreparation { int exit_code = 2; };
ServePreparation prepare_serve(const std::vector<std::string>& arguments) {
#include "serve.inc"
    return p;
}
#endif

int main(int argc, char** argv) {
    try {
#ifdef STRIX_TEST_MINIMAX_SERVE
        return prepare_serve({argv + 1, argv + argc}).exit_code;
#else
        const struct { struct { std::string lines; } config; } cut_args{};
#include "rank.inc"
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
    }
    return 2;
}

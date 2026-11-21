#include "strix/cli/cli.hpp"

int main(int argc, char** argv) {
    return strix::cli::main({argv + 1, argv + argc});
}

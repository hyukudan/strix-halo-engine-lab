#include "mini_window_program_device_fixture.hpp"

#include <exception>
#include <iostream>
#include <string_view>

// strix-cuda-inkling2-mini-window-program-test MODEL.safetensors [--full-head]
// The verify window program against the eager verify pass, stage by stage (mini_window_program_device_fixture.hpp).
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("expected MODEL.safetensors [--full-head]");
        bool screened = true;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option{argv[i]};
            if (option == "--full-head") screened = false;
            else throw std::invalid_argument("unknown option: " + std::string(option));
        }
        return strix::inkling2::test::window_program_identity(argv[1], screened);
    } catch (const std::exception& error) {
        std::cout << "WINFIX ERROR " << error.what() << '\n';
        return 2;
    }
}

#include "strix/support/matrix.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace support = strix::support;
    enum class Mode { check, markdown, json };
    try {
        std::filesystem::path root = ".";
        support::Filter filter;
        std::optional<Mode> mode;
        const auto select = [&](Mode requested) {
            if (mode) throw std::invalid_argument("select one output mode");
            mode = requested;
        };
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto next = [&]() -> std::string_view {
                if (++i >= argc || std::string_view(argv[i]).empty() || std::string_view(argv[i]).starts_with("--"))
                    throw std::invalid_argument("missing value for " + std::string(arg));
                return argv[i];
            };
            if (arg == "--help") {
                std::cout << "usage: strix-support-matrix [--root DIR] (--check | --markdown | --json)\n"
                             "                            [--family NAME] [--backend NAME] [--state STATE]\n";
                return 0;
            }
            if (arg == "--check") select(Mode::check);
            else if (arg == "--markdown") select(Mode::markdown);
            else if (arg == "--json") select(Mode::json);
            else if (arg == "--root") root = next();
            else if (arg == "--family") filter.family = next();
            else if (arg == "--backend") filter.backend = next();
            else if (arg == "--state") {
                filter.state = support::parse_state(next());
                if (!filter.state) throw std::invalid_argument("unknown support state");
            } else throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
        if (!mode) throw std::invalid_argument("select --check, --markdown or --json");
        if (*mode == Mode::check && (!filter.family.empty() || !filter.backend.empty() || filter.state))
            throw std::invalid_argument("--check verifies the full matrix; filters apply to output modes");
        const auto matrix = support::load(root / "support/matrix.toml");
        const auto errors = support::check(matrix, root);
        for (const auto& error : errors) std::cerr << error << '\n';
        if (!errors.empty()) return 1;
        const auto rows = support::query(matrix, filter);
        if (*mode == Mode::markdown) std::cout << support::markdown(rows, matrix.consumers);
        if (*mode == Mode::json) std::cout << support::json(rows, matrix.consumers);
        if (*mode == Mode::check)
            std::cout << "Checked " << matrix.rows.size() << " support rows and " << matrix.consumers.size()
                      << " consumer rows\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strix-support-matrix: " << error.what() << '\n';
        return 1;
    }
}

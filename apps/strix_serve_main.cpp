#ifndef STRIX_SERVE_CLI_ONLY
#include "strix/serve/run.hpp"
#endif

#if defined(STRIX_SERVE_NATIVE_LAUNCH) || defined(STRIX_SERVE_CLI_ONLY)
#include "strix/cli/cli.hpp"
#include "strix/cli/registry_root.hpp"
#if defined(STRIX_SERVE_NATIVE_LAUNCH)
#include "strix/serve/native_handoff.hpp"
#endif
#include <filesystem>
#include <iostream>
#endif

int main(int argc, char** argv) {
#ifdef STRIX_SERVE_CLI_ONLY
    if (argc == 1 || std::string_view(argv[1]) == "--help") return strix::cli::main({"--help"});
    std::vector<std::string> args{"serve"};
    args.insert(args.end(), argv + 1, argv + argc);
    return strix::cli::main(args);
#elif defined(STRIX_SERVE_NATIVE_LAUNCH)
    namespace sv = strix::serve;
    namespace cli = strix::cli;
    sv::ServiceFactories factories;
    sv::install_service_factories(factories);
    sv::ServeShell shell;
#if defined(_WIN32) || defined(__linux__)
    shell.registry_root = cli::registry_root(".", cli::data_directory(cli::executable_directory())).string();
#endif
    shell.launch = [](const sv::ServiceSelection& selection, const sv::ServeShell::NativeRequest& request, std::ostream& out) {
        return cli::launch_native(sv::native_serve_request(selection, request), selection.model, selection.description,
                                  selection.root, out);
    };
    return sv::run(argc, argv, factories, std::cout, std::cerr, shell);
#else
    return strix::serve::run(argc, argv);
#endif
}

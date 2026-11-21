#include "qwen/session_service.hpp"

#include <iostream>

int main() {
    strix::serve::ServiceFactories factories;
    strix::qwen::install_session_service(factories,
        [](const std::vector<std::string>&, const strix::serve::ServiceFactory::ServeScoped&) { return 0; });

    const auto* factory = factories.find("qwen");
    if (!factory || !factory->create_scoped) {
        std::cerr << "Qwen adapter did not register a scoped service factory\n";
        return 1;
    }
    return 0;
}

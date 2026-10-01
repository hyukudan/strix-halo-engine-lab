#include "strix/model_content_hash.hpp"
#include "strix/serial_decode_reference.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) try {
    if (argc != 2) {
        throw std::invalid_argument(
            "usage: strix-validate-serial-reference REFERENCE.txt");
    }

    const auto reference =
        strix::read_serial_decode_reference(
            argv[1]);

    std::uint64_t probes = 0;
    for (const auto& step :
         reference.steps) {
        const auto count =
            static_cast<std::uint64_t>(
                step.logits.size());
        if (count >
            std::numeric_limits<std::uint64_t>::max() -
                probes) {
            throw std::overflow_error(
                "serial reference probe count overflow");
        }
        probes += count;
    }

    std::cout
        << "{\n"
        << "  \"model_content_identity\": \""
        << strix::model_content_identity_hex(
               reference.model_content_identity)
        << "\",\n"
        << "  \"reference_id\": "
        << reference.independent_reference_identity
        << ",\n"
        << "  \"steps\": "
        << reference.steps.size()
        << ",\n"
        << "  \"logit_probes\": "
        << probes
        << ",\n"
        << "  \"multi_step_ready\": "
        << (reference.steps.size() >= 2
                ? "true"
                : "false")
        << "\n"
        << "}\n";

    return reference.steps.size() >= 2
        ? 0
        : 2;
} catch (const std::exception& error) {
    std::cerr
        << "strix-validate-serial-reference: "
        << error.what()
        << '\n';
    return 1;
}

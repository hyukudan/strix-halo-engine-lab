#include "strix/model_content_hash.hpp"
#include "strix/model_contract.hpp"
#include "strix/serial_decode_reference.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 2 && argc != 3) {
        throw std::invalid_argument(
            "usage: strix-validate-serial-reference REFERENCE.txt [MODEL.gguf]");
    }

    const auto reference =
        strix::read_serial_decode_reference(
            argv[1]);

    const auto model =
        strix::ModelContract::qwen38_flash_next();

    std::uint64_t probes = 0;
    for (const auto& step : reference.steps) {
        if (step.input_token >= model.vocab_size ||
            step.expected_argmax_token >= model.vocab_size) {
            throw std::invalid_argument(
                "serial reference token exceeds target model vocabulary");
        }
        const auto count =
            static_cast<std::uint64_t>(step.logits.size());
        if (count >
            std::numeric_limits<std::uint64_t>::max() - probes) {
            throw std::overflow_error(
                "serial reference probe count overflow");
        }
        for (const auto& probe : step.logits) {
            if (probe.index >= model.vocab_size) {
                throw std::invalid_argument(
                    "serial reference logit probe exceeds target model vocabulary");
            }
        }
        probes += count;
    }

    const bool model_content_checked = argc == 3;
    bool model_content_match = false;
    std::string actual_model_content_identity;
    if (model_content_checked) {
        const std::array model_files{
            strix::ModelContentFile{"model", argv[2]},
        };
        const auto actual_identity =
            strix::build_model_content_identity(model_files);
        actual_model_content_identity =
            strix::model_content_identity_hex(actual_identity);
        model_content_match =
            actual_identity == reference.model_content_identity;
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
        << "  \"vocab_size\": "
        << model.vocab_size
        << ",\n"
        << "  \"model_content_checked\": "
        << (model_content_checked ? "true" : "false")
        << ",\n";

    if (model_content_checked) {
        std::cout
            << "  \"actual_model_content_identity\": \""
            << actual_model_content_identity
            << "\",\n"
            << "  \"model_content_match\": "
            << (model_content_match ? "true" : "false")
            << ",\n";
    } else {
        std::cout << "  \"model_content_match\": null,\n";
    }

    std::cout
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

    const bool ready =
        reference.steps.size() >= 2 &&
        (!model_content_checked || model_content_match);
    return ready ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr
        << "strix-validate-serial-reference: "
        << error.what()
        << '\n';
    return 1;
}

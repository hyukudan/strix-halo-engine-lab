#include "strix/execution_plan.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_tensor_index.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string json_escape(const std::string& value) {
    std::string result;
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            result.push_back('\\');
        }
        if (ch >= 0x20U && ch < 0x80U) {
            result.push_back(static_cast<char>(ch));
        }
    }
    return result;
}

const char* cardinality_name(
    strix::TensorRequirementCardinality value) {
    switch (value) {
    case strix::TensorRequirementCardinality::exactly_one:
        return "exactly_one";
    case strix::TensorRequirementCardinality::one_or_more:
        return "one_or_more";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        throw std::invalid_argument(
            "usage: strix-audit-execution-tensors MODEL.gguf");
    }

    const auto info =
        strix::gguf::read_file(argv[1]);
    const auto manifest =
        strix::build_model_manifest(info);
    const auto index =
        strix::build_model_tensor_index(manifest);
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    const auto requirements =
        strix::build_serial_execution_tensor_requirements(
            execution);

    std::uint32_t missing = 0;
    std::uint32_t ambiguous = 0;
    std::uint32_t satisfied = 0;

    std::cout << "{\n"
              << "  \"manifest_layout_identity\": "
              << manifest.layout_identity << ",\n"
              << "  \"requirements\": [\n";

    for (std::size_t position = 0;
         position < requirements.size();
         ++position) {
        const auto& requirement =
            requirements[position];
        const auto matches =
            strix::select_manifest_tensors(
                manifest,
                index,
                requirement.selector);

        const bool is_missing = matches.empty();
        const bool is_ambiguous =
            requirement.cardinality ==
                strix::TensorRequirementCardinality::exactly_one &&
            matches.size() > 1U;

        if (is_missing) {
            ++missing;
        } else if (is_ambiguous) {
            ++ambiguous;
        } else {
            ++satisfied;
        }

        std::cout
            << "    {\"step_id\":"
            << requirement.step_id
            << ",\"op\":\""
            << strix::to_string(requirement.op)
            << "\",\"slot\":"
            << requirement.binding_slot
            << ",\"cardinality\":\""
            << cardinality_name(
                   requirement.cardinality)
            << "\",\"role\":\""
            << strix::to_string(
                   *requirement.selector.role)
            << "\",\"layer\":";

        if (requirement.selector.layer.has_value()) {
            std::cout
                << *requirement.selector.layer;
        } else {
            std::cout << "null";
        }

        std::cout
            << ",\"name_contains\":\""
            << json_escape(
                   std::string(
                       requirement.selector.name_contains))
            << "\",\"match_count\":"
            << matches.size()
            << ",\"status\":\""
            << (is_missing
                    ? "missing"
                    : is_ambiguous
                        ? "ambiguous"
                        : "satisfied")
            << "\",\"tensor_indices\":[";

        for (std::size_t match = 0;
             match < matches.size();
             ++match) {
            if (match != 0) {
                std::cout << ',';
            }
            std::cout << matches[match];
        }
        std::cout << "]}";
        if (position + 1U !=
            requirements.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    std::cout << "  ],\n"
              << "  \"satisfied\": "
              << satisfied << ",\n"
              << "  \"missing\": "
              << missing << ",\n"
              << "  \"ambiguous\": "
              << ambiguous << ",\n";

    if (missing == 0 && ambiguous == 0) {
        const auto access =
            strix::compile_execution_tensor_access_plan(
                execution,
                manifest,
                index,
                requirements);
        std::cout
            << "  \"compiled\": true,\n"
            << "  \"tensor_access_plan_identity\": "
            << access.plan_identity << '\n';
    } else {
        std::cout
            << "  \"compiled\": false,\n"
            << "  \"tensor_access_plan_identity\": null\n";
    }

    std::cout << "}\n";
    return missing == 0 && ambiguous == 0
        ? 0
        : 2;
} catch (const std::exception& error) {
    std::cerr
        << "strix-audit-execution-tensors: "
        << error.what() << '\n';
    return 1;
}

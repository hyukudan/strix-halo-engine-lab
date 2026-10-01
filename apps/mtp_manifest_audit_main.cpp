#include "strix/gguf_reader.hpp"
#include "strix/model_manifest.hpp"
#include "strix/mtp_draft_tensor_audit.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
    if (argc != 2) {
        throw std::invalid_argument(
            "usage: strix-mtp-manifest-audit MTP.gguf");
    }

    const auto info =
        strix::gguf::read_file(argv[1]);
    const auto manifest =
        strix::build_model_manifest(info);
    const auto audit =
        strix::audit_mtp_draft_tensor_families(
            manifest);

    std::cout
        << "{\n"
        << "  \"manifest_layout_identity\": "
        << audit.manifest_layout_identity
        << ",\n"
        << "  \"structurally_complete\": "
        << (audit.structurally_complete()
                ? "true"
                : "false")
        << ",\n"
        << "  \"mtp_tensor_count\": ";

    std::size_t mtp_count = 0;
    for (const auto& tensor :
         manifest.tensors) {
        if (tensor.role ==
            strix::TensorRole::mtp) {
            ++mtp_count;
        }
    }
    std::cout << mtp_count << ",\n";

    std::cout
        << "  \"unclassified_mtp_tensor_count\": "
        << audit.unclassified_mtp_tensors.size()
        << ",\n"
        << "  \"families\": [\n";

    for (std::size_t index = 0;
         index < audit.families.size();
         ++index) {
        const auto& family =
            audit.families[index];
        std::cout
            << "    {\"name\":\""
            << strix::to_string(family.family)
            << "\",\"required\":"
            << (family.required ? "true" : "false")
            << ",\"tensor_count\":"
            << family.tensor_indices.size()
            << "}"
            << (index + 1 ==
                        audit.families.size()
                    ? "\n"
                    : ",\n");
    }

    std::cout
        << "  ],\n"
        << "  \"missing_required_families\": [";
    for (std::size_t index = 0;
         index <
             audit.missing_required_families.size();
         ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout
            << "\""
            << strix::to_string(
                   audit.missing_required_families[
                       index])
            << "\"";
    }
    std::cout << "]\n}\n";
    return audit.structurally_complete()
        ? 0
        : 2;
} catch (const std::exception& error) {
    std::cerr << "error: "
              << error.what()
              << '\n';
    return 1;
}

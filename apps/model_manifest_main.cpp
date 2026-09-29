#include "strix/gguf_reader.hpp"
#include "strix/model_manifest.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::string json_escape(const std::string& value) {
    std::string result;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(character));
        } else if (character >= 0x20U && character < 0x80U) {
            result.push_back(static_cast<char>(character));
        } else {
            std::ostringstream escaped;
            escaped << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
            result += escaped.str();
        }
    }
    return result;
}

void print_dimensions(const std::vector<std::uint64_t>& dimensions) {
    std::cout << '[';
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << dimensions[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        throw std::invalid_argument("usage: strix-model-manifest MODEL.gguf");
    }

    const auto info = strix::gguf::read_file(argv[1]);
    const auto manifest = strix::build_model_manifest(info);

    std::cout << "{\n"
              << "  \"architecture\": \"" << json_escape(manifest.architecture) << "\",\n"
              << "  \"layout_identity\": " << manifest.layout_identity << ",\n"
              << "  \"total_tensor_bytes\": " << manifest.total_tensor_bytes << ",\n"
              << "  \"file_backed_candidate_bytes\": " << manifest.file_backed_candidate_bytes << ",\n"
              << "  \"bounded_cache_candidate_bytes\": " << manifest.bounded_cache_candidate_bytes << ",\n"
              << "  \"unknown_tensor_count\": " << manifest.unknown_tensor_count << ",\n"
              << "  \"tensors\": [\n";

    for (std::size_t index = 0; index < manifest.tensors.size(); ++index) {
        const auto& tensor = manifest.tensors[index];
        std::cout << "    {\"name\":\"" << json_escape(tensor.name)
                  << "\",\"role\":\"" << strix::to_string(tensor.role)
                  << "\",\"residency\":\"" << strix::to_string(tensor.residency)
                  << "\",\"placement\":{\"kind\":\"" << strix::to_string(tensor.placement.kind)
                  << "\",\"node\":" << tensor.placement.node << "},\"layer\":";
        if (tensor.layer.has_value()) {
            std::cout << *tensor.layer;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"ggml_type\":" << tensor.ggml_type
                  << ",\"byte_size\":" << tensor.byte_size
                  << ",\"absolute_offset\":" << tensor.absolute_offset
                  << ",\"dimensions\":";
        print_dimensions(tensor.dimensions);
        std::cout << '}' << (index + 1 == manifest.tensors.size() ? '\n' : ',') << '\n';
    }

    std::cout << "  ]\n}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}

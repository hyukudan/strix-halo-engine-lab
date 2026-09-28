#include "strix/gguf_reader.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
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

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        throw std::invalid_argument("usage: strix-inspect-gguf MODEL.gguf");
    }
    const auto info = strix::gguf::read_file(argv[1]);
    const auto* architecture = info.find_metadata("general.architecture");
    const std::string architecture_value =
        architecture != nullptr && architecture->string_value.has_value()
            ? *architecture->string_value
            : "";
    std::uint64_t tensor_bytes = 0;
    for (const auto& tensor : info.tensors) {
        if (tensor.byte_size > std::numeric_limits<std::uint64_t>::max() - tensor_bytes) {
            throw std::overflow_error("aggregate tensor byte count overflow");
        }
        tensor_bytes += tensor.byte_size;
    }

    std::cout << "{\n"
              << "  \"version\": " << info.version << ",\n"
              << "  \"file_size\": " << info.file_size << ",\n"
              << "  \"alignment\": " << info.alignment << ",\n"
              << "  \"data_offset\": " << info.data_offset << ",\n"
              << "  \"metadata_count\": " << info.metadata.size() << ",\n"
              << "  \"tensor_count\": " << info.tensors.size() << ",\n"
              << "  \"tensor_bytes\": " << tensor_bytes << ",\n"
              << "  \"architecture\": \"" << json_escape(architecture_value) << "\"\n"
              << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}

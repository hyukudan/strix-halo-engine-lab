#include "strix/serial_decode_reference.hpp"

#include "strix/model_content_hash.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix {
namespace {

std::uint64_t parse_u64(
    std::string_view text,
    const char* field) {
    std::uint64_t value = 0;
    const auto result =
        std::from_chars(
            text.data(),
            text.data() + text.size(),
            value);
    if (result.ec != std::errc{} ||
        result.ptr !=
            text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(field) +
            " is not a valid uint64");
    }
    return value;
}

std::uint32_t parse_u32(
    std::string_view text,
    const char* field) {
    const auto value =
        parse_u64(text, field);
    if (value >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::out_of_range(
            std::string(field) +
            " exceeds uint32");
    }
    return static_cast<std::uint32_t>(
        value);
}

float parse_float(
    std::string_view text,
    const char* field) {
    std::string copy(text);
    char* end = nullptr;
    errno = 0;
    const auto value =
        std::strtof(
            copy.c_str(),
            &end);
    if (errno != 0 ||
        end == copy.c_str() ||
        *end != '\0' ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(field) +
            " is not a finite float");
    }
    return value;
}

std::uint8_t hex_nibble(char value) {
    if (value >= '0' &&
        value <= '9') {
        return static_cast<std::uint8_t>(
            value - '0');
    }
    if (value >= 'a' &&
        value <= 'f') {
        return static_cast<std::uint8_t>(
            value - 'a' + 10);
    }
    if (value >= 'A' &&
        value <= 'F') {
        return static_cast<std::uint8_t>(
            value - 'A' + 10);
    }
    throw std::invalid_argument(
        "content SHA-256 contains non-hex character");
}

ModelContentIdentity parse_identity(
    std::string_view hex) {
    ModelContentIdentity identity;
    if (hex.size() !=
        identity.bytes.size() * 2U) {
        throw std::invalid_argument(
            "content SHA-256 must contain exactly 64 hex characters");
    }

    for (std::size_t index = 0;
         index < identity.bytes.size();
         ++index) {
        identity.bytes[index] =
            static_cast<std::uint8_t>(
                (hex_nibble(hex[index * 2U]) << 4U) |
                hex_nibble(hex[index * 2U + 1U]));
    }
    if (!identity.valid()) {
        throw std::invalid_argument(
            "content SHA-256 must not be all zero");
    }
    return identity;
}

std::vector<std::string> words(
    const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> result;
    std::string word;
    while (input >> word) {
        result.push_back(
            std::move(word));
    }
    return result;
}

}  // namespace

SerialDecodeReference
read_serial_decode_reference(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open serial decode reference");
    }

    std::string line;
    if (!std::getline(input, line) ||
        line !=
            "STRIX_SERIAL_REPLAY_V1") {
        throw std::invalid_argument(
            "serial decode reference has unsupported or missing version");
    }

    SerialDecodeReference reference;

    if (!std::getline(input, line)) {
        throw std::invalid_argument(
            "serial decode reference is missing content SHA-256");
    }
    {
        const auto parts = words(line);
        if (parts.size() != 2 ||
            parts[0] !=
                "content_sha256") {
            throw std::invalid_argument(
                "serial decode reference expected content_sha256 line");
        }
        reference.model_content_identity =
            parse_identity(parts[1]);
    }

    if (!std::getline(input, line)) {
        throw std::invalid_argument(
            "serial decode reference is missing reference id");
    }
    {
        const auto parts = words(line);
        if (parts.size() != 2 ||
            parts[0] !=
                "reference_id") {
            throw std::invalid_argument(
                "serial decode reference expected reference_id line");
        }
        reference.independent_reference_identity =
            parse_u64(
                parts[1],
                "reference_id");
        if (reference.independent_reference_identity == 0) {
            throw std::invalid_argument(
                "reference_id must be non-zero");
        }
    }

    std::set<std::uint32_t> current_logit_indices;

    while (std::getline(input, line)) {
        if (line.empty() ||
            line[0] == '#') {
            continue;
        }

        const auto parts = words(line);
        if (parts.empty()) {
            continue;
        }

        if (parts[0] == "step") {
            if (parts.size() != 4) {
                throw std::invalid_argument(
                    "step line must contain input token, argmax token and probe count");
            }

            SerialDecodeReferenceStep step;
            step.input_token =
                parse_u32(
                    parts[1],
                    "step input token");
            step.expected_argmax_token =
                parse_u32(
                    parts[2],
                    "step argmax token");
            const auto probes =
                parse_u64(
                    parts[3],
                    "step probe count");
            if (probes == 0 ||
                probes >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
                throw std::invalid_argument(
                    "step probe count must be non-zero and fit size_t");
            }

            step.logits.reserve(
                static_cast<std::size_t>(
                    probes));
            current_logit_indices.clear();

            for (std::uint64_t probe = 0;
                 probe < probes;
                 ++probe) {
                if (!std::getline(input, line)) {
                    throw std::invalid_argument(
                        "serial decode reference ended inside logit probes");
                }
                const auto logit_parts =
                    words(line);
                if (logit_parts.size() != 3 ||
                    logit_parts[0] !=
                        "logit") {
                    throw std::invalid_argument(
                        "serial decode reference expected logit line");
                }

                const auto index =
                    parse_u32(
                        logit_parts[1],
                        "logit index");
                if (!current_logit_indices.insert(index).second) {
                    throw std::invalid_argument(
                        "duplicate logit index within reference step");
                }

                step.logits.push_back(
                    SerialDecodeReferenceLogit{
                        index,
                        parse_float(
                            logit_parts[2],
                            "logit value"),
                    });
            }

            reference.steps.push_back(
                std::move(step));
            continue;
        }

        throw std::invalid_argument(
            "unknown serial decode reference record");
    }

    if (reference.steps.empty()) {
        throw std::invalid_argument(
            "serial decode reference contains no steps");
    }
    return reference;
}

}  // namespace strix

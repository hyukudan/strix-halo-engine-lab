#pragma once

#include "strix/gguf_reader.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

// Registry of model families. The core (GGUF reader, residency, formats, windows, communicator, server) knows no
// model; each family is an optional module that registers itself here: the GGUF architectures it admits, its
// structural validation and a one-line description for logs and receipts. A program links the families it wants and
// calls their register functions (explicit calls, so static libraries keep their registration).
namespace strix {

struct ModelFamily {
    std::string name;                        // e.g. "deepseek41"
    std::vector<std::string> architectures;  // general.architecture values the family admits
    // Structural admission of a parsed file; throws std::invalid_argument naming the first violation.
    std::function<void(const gguf::FileInfo&)> validate;
    // Short human-readable summary of an admitted file (geometry, tensor count).
    std::function<std::string(const gguf::FileInfo&)> describe;
    // Reader limits the family needs (e.g. retained metadata arrays); default limits when empty.
    std::function<gguf::Limits()> reader_limits;
};

// Throws std::invalid_argument if the name or one of the architectures is already registered.
void register_model_family(ModelFamily family);
// Registering the same family twice is allowed when it is the identical registration (idempotent init calls).
[[nodiscard]] bool model_family_registered(std::string_view name);
// nullptr when no registered family admits the architecture.
[[nodiscard]] const ModelFamily* find_model_family(std::string_view architecture);
[[nodiscard]] std::vector<std::string> registered_model_families();

// Reads the file's architecture key with default limits, then re-reads with the family's limits and validates.
// Throws std::invalid_argument when no family admits the architecture.
struct AdmittedFile {
    const ModelFamily* family = nullptr;
    gguf::FileInfo info;
};
[[nodiscard]] AdmittedFile admit_model_file(const std::filesystem::path& path);

}  // namespace strix

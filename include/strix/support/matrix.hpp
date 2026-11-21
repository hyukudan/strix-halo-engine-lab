#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::support {

enum class State { declared, implemented, connected, validated, packaged };
std::string_view state_name(State state);
std::optional<State> parse_state(std::string_view name);

struct Row {
    std::string family, format, backend, arch, os, function;
    State state = State::declared;
    std::vector<std::string> sources;
    std::string entrypoint, receipt, device, toolchain, model, profile, component, note;
};

struct Consumer {
    std::string component, operation, file, symbol, test;
};

struct Matrix {
    std::vector<Row> rows;
    std::vector<Consumer> consumers;
};

struct Filter {
    std::string family, backend;
    std::optional<State> state;
};

// Load the v1 schema. Malformed data and I/O failures throw.
Matrix load(const std::filesystem::path& file);
// Return diagnostics for evidence and coverage against a repository checkout.
std::vector<std::string> check(const Matrix& matrix, const std::filesystem::path& root);
// Filters match exactly. Results borrow rows from the matrix in file order.
std::vector<const Row*> query(const Matrix& matrix, const Filter& filter = {});
// Both renderers preserve the supplied order. Consumer rows are independent of model filters.
std::string markdown(std::span<const Row* const> rows, std::span<const Consumer> consumers);
std::string json(std::span<const Row* const> rows, std::span<const Consumer> consumers);

} // namespace strix::support

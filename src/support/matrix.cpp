#include "strix/support/matrix.hpp"

#include "strix/model_registry.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace strix::support {
namespace {
namespace fs = std::filesystem;
namespace md = model_description;
constexpr std::string_view schema = "strix-support/v1";
constexpr std::array<std::string_view, 5> states{"declared", "implemented", "connected", "validated", "packaged"};

std::string read(const fs::path& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + file.generic_string());
    return {std::istreambuf_iterator<char>(input), {}};
}

template<class T>
T field(const md::Value::Table& row, std::string_view key) {
    const auto it = row.find(key);
    if (it == row.end()) return {};
    const auto* result = std::get_if<T>(&it->second.data);
    if (!result) throw std::invalid_argument("invalid type for " + std::string(key));
    return *result;
}

auto key(const Row& row) {
    return std::tie(row.family, row.format, row.backend, row.arch, row.os, row.function);
}

bool identifier_byte(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_';
}

bool contains_symbol(const std::string& content, const std::string& symbol) {
    std::istringstream lines(content);
    for (std::string line; std::getline(lines, line);) {
        const auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos || line[start] == '#') continue;
        const std::string_view code = std::string_view(line).substr(0, line.find("//"));
        for (auto pos = code.find(symbol); pos != std::string_view::npos; pos = code.find(symbol, pos + 1)) {
            const auto end = pos + symbol.size();
            if ((pos == 0 || !identifier_byte(code[pos - 1])) &&
                (end == code.size() || !identifier_byte(code[end]))) return true;
        }
    }
    return false;
}

bool has_anchor(const std::string& content, const std::string& anchor) {
    if (content.find("id=\"" + anchor + "\"") != std::string::npos) return true;
    std::istringstream lines(content);
    for (std::string line; std::getline(lines, line);) {
        const auto start = line.find_first_not_of('#');
        if (start == std::string::npos || start == 0 || line[start] != ' ') continue;
        const auto heading = line.find_first_not_of(' ', start);
        if (heading == std::string::npos) continue;
        std::string slug;
        for (const char c : std::string_view(line).substr(heading)) {
            if (c >= 'A' && c <= 'Z') slug += static_cast<char>(c - 'A' + 'a');
            else if (identifier_byte(c) || c == '-') slug += c;
            else if (c == ' ') slug += '-';
        }
        if (slug == anchor) return true;
    }
    return false;
}

// Read the literal lists owned by packaging. Refuse an absent, repeated or computed list.
std::set<std::string> cmake_list(const std::string& content, const std::string& name) {
    const std::regex declaration(R"((^|\n)[ \t]*set\s*\(\s*)" + name + R"(\s+([^)]+)\))");
    std::sregex_iterator match(content.begin(), content.end(), declaration);
    const std::sregex_iterator end;
    if (match == end) throw std::invalid_argument("missing packaging list " + name);
    std::istringstream values((*match)[2].str());
    if (++match != end) throw std::invalid_argument("repeated packaging list " + name);
    const std::regex literal("[A-Za-z0-9_-]+");
    std::set<std::string> list;
    for (std::string value; values >> value;) {
        if (!std::regex_match(value, literal) || !list.insert(value).second)
            throw std::invalid_argument("invalid item in packaging list " + name);
    }
    return list;
}

void json_string(std::ostream& out, std::string_view value) {
    constexpr std::string_view hex = "0123456789abcdef";
    out << '"';
    for (const char c : value) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte < 0x20U) out << "\\u00" << hex[byte >> 4U] << hex[byte & 0xfU];
        else {
            if (c == '"' || c == '\\') out << '\\';
            out << c;
        }
    }
    out << '"';
}

std::string md_cell(std::string_view value) {
    std::string result;
    for (const char c : value) {
        if (c == '\n' || c == '\r') result += ' ';
        else {
            if (c == '|' || c == '`' || c == '<' || c == '>') result += '\\';
            result += c;
        }
    }
    return result;
}

std::string link(const std::string& path) {
    if (path.empty()) return {};
    return "[`" + md_cell(path) + "`](../" + path + ")";
}
} // namespace

std::string_view state_name(State state) {
    return states.at(static_cast<std::size_t>(state));
}

std::optional<State> parse_state(std::string_view name) {
    const auto it = std::find(states.begin(), states.end(), name);
    if (it == states.end()) return std::nullopt;
    return static_cast<State>(it - states.begin());
}

Matrix load(const fs::path& file) {
    const auto parsed = md::load(file.string(), md::TableSyntax::root_array_tables);
    if (field<std::string>(parsed.fields, "schema") != schema)
        throw std::invalid_argument("support matrix schema must be strix-support/v1");
    Matrix matrix;
    const auto parse_rows = [&](std::string_view group, auto add) {
        const auto items = field<md::Value::Array>(parsed.fields, group);
        for (std::size_t i = 0; i < items.size(); ++i) {
            try {
                add(std::get<md::Value::Table>(items[i].data));
            } catch (const std::exception& error) {
                throw std::invalid_argument(std::string(group) + " " + std::to_string(i + 1) + ": " + error.what());
            }
        }
    };
    parse_rows("support", [&](const md::Value::Table& table) {
        const auto get = [&](std::string_view name) { return field<std::string>(table, name); };
        const auto state = parse_state(get("state"));
        if (!state) throw std::invalid_argument("missing or unknown state");
        Row row{get("family"), get("format"), get("backend"), get("arch"), get("os"), get("function"),
            *state, {}, get("entrypoint"), get("receipt"), get("device"), get("toolchain"), get("model"),
            get("profile"), get("component"), get("note")};
        for (const auto& source : field<md::Value::Array>(table, "sources"))
            row.sources.push_back(std::get<std::string>(source.data));
        matrix.rows.push_back(std::move(row));
    });
    parse_rows("consumer", [&](const md::Value::Table& table) {
        const auto get = [&](std::string_view name) { return field<std::string>(table, name); };
        matrix.consumers.push_back({get("component"), get("operation"), get("file"), get("symbol"), get("test")});
    });
    return matrix;
}

std::vector<std::string> check(const Matrix& matrix, const fs::path& root) {
    std::vector<std::string> errors;
    std::map<std::string, std::set<std::string>> families;
    for (const auto& model : model_registry::load((root / "models/registry.toml").string()))
        families[model.family].insert(model.name);
    const auto support = read(root / "packaging/StrixSupportMatrix.cmake");
    const auto packaging = read(root / "packaging/StrixPackaging.cmake");
    struct Archs {
        std::string_view os;
        std::set<std::string> packaging, registered;
    };
    std::map<std::string, Archs> archs{
        {"hip", {"linux", cmake_list(support, "STRIX_HIP_QUALIFIED_ARCHS"), {}}},
        {"cuda", {"windows", cmake_list(support, "STRIX_CUDA_QUALIFIED_ARCHS"), {}}}};
    std::map<std::string, std::set<std::string>> components;

    std::set<std::string> covered;
    std::set<decltype(key(Row{}))> seen;
    const auto exists = [&](const std::string& value) {
        const fs::path path(value);
        return !value.empty() && !path.has_root_path() && value.find_first_of("\\:#") == std::string::npos &&
            std::none_of(path.begin(), path.end(), [](const auto& part) { return part == ".."; }) &&
            fs::is_regular_file(root / path);
    };
    for (const auto& row : matrix.rows) {
        const std::string id = row.family + "/" + row.format + "/" + row.backend + "/" + row.arch + "/" +
            row.os + "/" + row.function;
        const auto fail = [&](const std::string& reason) { errors.push_back(id + ": " + reason); };
        if (!seen.insert(key(row)).second) fail("duplicate support key");
        for (const auto* field_value : {&row.family, &row.format, &row.backend, &row.arch, &row.os, &row.function})
            if (field_value->empty()) fail("empty support key field");
        const auto family = families.find(row.family);
        if (family == families.end()) fail("unknown model family");
        covered.insert(row.family);
        if (row.state >= State::implemented && row.sources.empty()) fail("missing sources");
        for (const auto& file : row.sources)
            if (!exists(file)) fail("source file missing or invalid: " + file);
        if (row.state >= State::connected && row.entrypoint.empty()) fail("missing entrypoint");
        if (const auto hash = row.entrypoint.find('#'); hash != std::string::npos) {
            const auto file = row.entrypoint.substr(0, hash);
            const auto symbol = row.entrypoint.substr(hash + 1);
            if (!exists(file) || symbol.empty() || !contains_symbol(read(root / file), symbol))
                fail("factory entrypoint file or symbol missing: " + row.entrypoint);
        }
        const auto arch = archs.find(row.backend);
        const bool pending = row.note.starts_with("PENDING-HARDWARE");
        if (arch != archs.end() && (row.os != arch->second.os || !arch->second.packaging.contains(row.arch)) &&
            !pending)
            fail("unqualified architecture or OS requires a PENDING-HARDWARE note");
        if (row.state > State::implemented && pending)
            fail("PENDING-HARDWARE rows cannot be connected or higher");

        if (row.state >= State::validated) {
            const auto hash = row.receipt.find('#');
            const auto file = row.receipt.substr(0, hash);
            if (!exists(file)) fail("receipt file missing or invalid: " + file);
            else if (hash == std::string::npos && fs::path(file).extension() == ".md")
                fail("Markdown receipt requires a section anchor: " + row.receipt);
            else if (hash != std::string::npos &&
                     !has_anchor(read(root / file), row.receipt.substr(hash + 1)))
                fail("receipt anchor missing: " + row.receipt);
            if (row.device.empty() || row.toolchain.empty() || row.model.empty() || row.profile.empty())
                fail("validated evidence requires device, toolchain, model and profile");
            if (!row.model.empty() && family != families.end() && !family->second.contains(row.model))
                fail("receipt model is outside the family: " + row.model);
            if (arch != archs.end()) {
                if (row.os != arch->second.os) fail("OS disagrees with architecture qualification in packaging");
                else arch->second.registered.insert(row.arch);
            }
        }
        if (row.state == State::packaged) {
            if (row.component != "hip" && row.component != "cuda" && row.component != "core" &&
                row.component != "tools") {
                fail("missing or unknown package component: " + row.component);
                continue;
            }
            if (!components.contains(row.component)) {
                auto upper = row.component;
                for (char& c : upper) c = static_cast<char>(c - 'a' + 'A');
                components.emplace(row.component, cmake_list(packaging, "STRIX_PACKAGE_" + upper + "_TARGETS"));
            }
            if (!components.at(row.component).contains(row.entrypoint.substr(0, row.entrypoint.find(' '))))
                fail("entrypoint is outside the package component: " + row.component);
        }
    }
    for (const auto& [family, models] : families) {
        (void)models;
        if (!covered.contains(family)) errors.push_back("missing support rows for " + family);
    }
    for (const auto& [backend, lists] : archs)
        if (lists.packaging != lists.registered) errors.push_back(backend + ": qualified arches disagree with packaging");

    std::set<std::tuple<std::string, std::string, std::string, std::string>> consumers;
    if (matrix.consumers.empty()) errors.emplace_back("missing consumer rows");
    for (const auto& row : matrix.consumers) {
        const auto id = row.component + "/" + row.operation + "/" + row.file + "#" + row.symbol;
        if (!consumers.emplace(row.component, row.operation, row.file, row.symbol).second)
            errors.push_back(id + ": duplicate consumer");
        if (row.component.empty() || row.operation.empty() || row.symbol.empty())
            errors.push_back(id + ": missing consumer component, operation or symbol");
        if (!exists(row.file)) errors.push_back(id + ": consumer file missing or invalid");
        else if (!row.symbol.empty() && !contains_symbol(read(root / row.file), row.symbol))
            errors.push_back(id + ": consumer symbol absent");
        if (!exists(row.test)) errors.push_back(id + ": consumer test file missing or invalid: " + row.test);
    }
    return errors;
}

std::vector<const Row*> query(const Matrix& matrix, const Filter& filter) {
    std::vector<const Row*> rows;
    for (const auto& row : matrix.rows)
        if ((filter.family.empty() || filter.family == row.family) &&
            (filter.backend.empty() || filter.backend == row.backend) &&
            (!filter.state || *filter.state == row.state)) rows.push_back(&row);
    return rows;
}

std::string markdown(std::span<const Row* const> rows, std::span<const Consumer> consumers) {
    std::ostringstream out;
    out << "<!-- Generated by strix-support-matrix --markdown from the repository root. -->\n"
        "# Support matrix\n\n"
        "Rows describe one family, weight format, backend, architecture, OS and function. "
        "States are cumulative: declared, implemented, connected, validated, packaged. "
        "Validation applies to the model and profile in the evidence. Packages contain programs and require separate weights.\n\n"
        "| Family | Weight format | Backend | Arch | OS | Function | State | Evidence |\n"
        "|---|---|---|---|---|---|---|---|\n";
    for (const auto* row : rows) {
        out << "| ";
        for (const auto* value : {&row->family, &row->format, &row->backend, &row->arch, &row->os, &row->function})
            out << md_cell(*value) << " | ";
        out << state_name(row->state) << " |";
        for (const auto& file : row->sources) out << ' ' << link(file);
        if (!row->entrypoint.empty()) out << " · `" << md_cell(row->entrypoint) << '`';
        if (!row->receipt.empty())
            out << " · " << link(row->receipt) << " (" << md_cell(row->model) << ", " << md_cell(row->profile)
                << ", " << md_cell(row->device) << ", " << md_cell(row->toolchain) << ')';
        if (!row->component.empty()) out << " · Package `" << md_cell(row->component) << '`';
        if (!row->note.empty()) out << ' ' << md_cell(row->note);
        out << " |\n";
    }
    out << "\n## Shared component consumers\n\n"
        "Each row points to a use in source and a test for the shared operation.\n\n"
        "| Component | Operation | Consumer file | Symbol | Test |\n|---|---|---|---|---|\n";
    for (const auto& row : consumers)
        out << "| " << md_cell(row.component) << " | " << md_cell(row.operation) << " | " << link(row.file)
            << " | `" << md_cell(row.symbol) << "` | " << link(row.test) << " |\n";
    return out.str();
}

std::string json(std::span<const Row* const> rows, std::span<const Consumer> consumers) {
    using Field = std::pair<std::string_view, std::string_view>;
    std::ostringstream out;
    const auto fields = [&](std::initializer_list<Field> entries) {
        bool first = true;
        for (const auto& [name, value] : entries) {
            if (!std::exchange(first, false)) out << ',';
            json_string(out, name);
            out << ':';
            json_string(out, value);
        }
    };
    out << "{\"schema\":";
    json_string(out, schema);
    out << ",\"support\":[";
    bool first = true;
    for (const auto* row : rows) {
        if (!std::exchange(first, false)) out << ',';
        out << "\n{";
        fields({{"family", row->family}, {"format", row->format}, {"backend", row->backend}, {"arch", row->arch},
            {"os", row->os}, {"function", row->function}, {"state", state_name(row->state)},
            {"entrypoint", row->entrypoint}, {"receipt", row->receipt}, {"device", row->device},
            {"toolchain", row->toolchain}, {"model", row->model}, {"profile", row->profile},
            {"component", row->component}, {"note", row->note}});
        out << ",\"sources\":[";
        bool first_source = true;
        for (const auto& source : row->sources) {
            if (!std::exchange(first_source, false)) out << ',';
            json_string(out, source);
        }
        out << "]}";
    }
    out << "],\"consumer\":[";
    first = true;
    for (const auto& row : consumers) {
        if (!std::exchange(first, false)) out << ',';
        out << "\n{";
        fields({{"component", row.component}, {"operation", row.operation}, {"file", row.file},
            {"symbol", row.symbol}, {"test", row.test}});
        out << '}';
    }
    out << "]}\n";
    return out.str();
}
} // namespace strix::support

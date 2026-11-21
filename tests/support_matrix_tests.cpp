#include "strix/support/matrix.hpp"

#include "strix/model_description.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
namespace support = strix::support;
namespace md = strix::model_description;

void require(bool condition, std::string_view why) {
    if (!condition) throw std::runtime_error(std::string(why));
}

void check_parser() {
    constexpr auto rows = "[[row]]\nitem.key = 'first'\n[[row]]\nitem.key = 'second'\n";
    const auto parsed = md::parse(rows, md::TableSyntax::root_array_tables);
    const auto& array = std::get<md::Value::Array>(parsed.find("row")->data);
    require(array.size() == 2, "repeated headers should append rows");
    const auto value = [&](std::size_t i) {
        const auto& row = std::get<md::Value::Table>(array.at(i).data);
        return std::get<std::string>(std::get<md::Value::Table>(row.at("item").data).at("key").data);
    };
    require(value(0) == "first" && value(1) == "second", "array table values should be independent");
    const auto rejects = [](std::string_view data, md::TableSyntax syntax) {
        try {
            (void)md::parse(data, syntax);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    require(rejects(rows, md::TableSyntax::standard), "legacy parser unexpectedly accepted an array table");
    require(rejects("row = []\n[[row]]\n", md::TableSyntax::root_array_tables), "array table redefined a key");
    require(rejects("[[row]]\na = 1\na = 2\n", md::TableSyntax::root_array_tables), "duplicate row field");
}

void check_validation(const std::filesystem::path& root) {
    const auto good = support::load(root / "support/matrix.toml");
    const auto errors = support::check(good, root);
    for (const auto& error : errors) std::cerr << error << '\n';
    require(errors.empty(), "checked-in matrix fails validation");
    const auto selected = support::query(good, {"", "hip", support::State::packaged});
    require(!selected.empty(), "expected a packaged HIP row");
    const auto index = static_cast<std::size_t>(selected.front() - good.rows.data());
    require(support::query(good, {good.rows.at(index).family, "hip", support::State::packaged}).front() ==
        selected.front(), "query should intersect family, backend and state");

    const auto invalid = [&](auto mutate, std::string_view diagnostic) {
        auto bad = good;
        mutate(bad, bad.rows.at(index));
        const auto problems = support::check(bad, root);
        require(std::any_of(problems.begin(), problems.end(), [&](const auto& item) {
            return item.find(diagnostic) != std::string::npos;
        }), diagnostic);
    };
    invalid([](auto&, auto& row) { row.sources.clear(); }, "missing sources");
    invalid([](auto&, auto& row) { row.sources = {"src/no-such-support-file.cpp"}; }, "source file missing");
    invalid([](auto&, auto& row) { row.entrypoint.clear(); }, "missing entrypoint");
    invalid([](auto&, auto& row) { row.entrypoint = "src/serve/run.cpp#missing_factory"; }, "factory entrypoint");
    invalid([](auto&, auto& row) { row.device.clear(); }, "requires device");
    invalid([](auto&, auto& row) { row.receipt = "docs/no-such-receipt.md"; }, "receipt file missing");
    invalid([](auto&, auto& row) { row.receipt += "-missing"; }, "receipt anchor missing");
    invalid([](auto&, auto& row) { row.component = "tools"; }, "outside the package component");
    invalid([](auto&, auto& row) { row.arch = "unknown-arch"; }, "qualified arches disagree");
    invalid([](auto&, auto& row) { row.arch = "unknown-arch"; }, "requires a PENDING-HARDWARE note");
    invalid([](auto&, auto& row) { row.note = "PENDING-HARDWARE"; }, "cannot be connected or higher");
    invalid([](auto& matrix, auto&) { matrix.consumers.at(0).file = "src/no-such-consumer.cpp"; }, "consumer file missing");
    invalid([](auto& matrix, auto&) { matrix.consumers.at(0).symbol = "strix_missing_consumer_symbol"; }, "consumer symbol absent");
    invalid([](auto& matrix, auto&) { matrix.consumers.at(0).test.clear(); }, "consumer test file missing");
    invalid([](auto& matrix, auto& row) {
        const auto family = row.family;
        std::erase_if(matrix.rows, [&](const auto& item) { return item.family == family; });
    }, "missing support rows");
    invalid([](auto& matrix, auto& row) {
        const auto duplicate = row;
        matrix.rows.push_back(duplicate);
    }, "duplicate support key");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected repository root");
        check_parser();
        check_validation(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include "strix/model_description.hpp"

#include <cstdio>
#include <stdexcept>

int main() try {
    using namespace strix::model_description;
    constexpr std::size_t count = 512;
    std::string input = "stages = [\n";
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = std::to_string(i);
        input += "{ shape = { nested = { id = " + value + " } }, route.id = " + value + " },\n";
    }
    input += "]\n[metadata]\nroute.id = 7\n[[visits]]\nroute.id = 8\n[[visits]]\nroute.id = 9\n";
    const auto decl = parse(input, TableSyntax::root_array_tables);
    const auto& stages = std::get<Value::Array>(decl.fields.at("stages").data);
    if (stages.size() != count) throw std::runtime_error("incorrect stage count");
    for (std::size_t i = 0; i < count; ++i) {
        const auto& stage = std::get<Value::Table>(stages.at(i).data);
        const auto& route = std::get<Value::Table>(stage.at("route").data);
        const auto& shape = std::get<Value::Table>(stage.at("shape").data);
        const auto& nested = std::get<Value::Table>(shape.at("nested").data);
        const auto expected = static_cast<std::int64_t>(i);
        if (std::get<std::int64_t>(route.at("id").data) != expected ||
            std::get<std::int64_t>(nested.at("id").data) != expected)
            throw std::runtime_error("incorrect inline table ID");
    }
    const auto* metadata = decl.find("metadata.route.id");
    if (!metadata || std::get<std::int64_t>(metadata->data) != 7)
        throw std::runtime_error("incorrect metadata route ID");
    const auto& visits = std::get<Value::Array>(decl.fields.at("visits").data);
    if (visits.size() != 2U) throw std::runtime_error("incorrect visit count");
    for (std::size_t i = 0; i < visits.size(); ++i) {
        const auto& visit = std::get<Value::Table>(visits[i].data);
        const auto& route = std::get<Value::Table>(visit.at("route").data);
        if (std::get<std::int64_t>(route.at("id").data) != 8 + static_cast<std::int64_t>(i))
            throw std::runtime_error("incorrect visit route ID");
    }

    for (const std::string_view invalid : {
             "a = { b = { c = 1 }, b.d = 2 }\n",
             "a = { b.c = 1 }\na.b.d = 2\n",
             "a = { b = { c = 1 } }\n[a.b]\n",
             "a.b = 1\n[a]\n",
         }) {
        try {
            (void)parse(invalid);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("accepted an invalid inline extension or duplicate table");
    }
    std::printf("model_decl_inline_tables PASS tables=%zu\n", count);
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}

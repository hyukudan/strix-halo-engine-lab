#include "gemma4/engine/coordinator.hpp"
#include "gemma4/exl3/format.hpp"
#include <iostream>

namespace g4 = strix::gemma4;
namespace {
class Backend final : public g4::engine::Backend {
public:
    bool exl3 = true;
    std::unique_ptr<g4::engine::StreamState> create_stream() override { throw std::logic_error("metadata-only fixture"); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan>) override { throw std::logic_error("metadata-only fixture"); }
    const g4::ModelConfig& config() const override { return config_; }
    strix::serve::Json identity() const override {
        auto out = strix::serve::Json::object();
        if (exl3) out.set("identity", g4::exl3::identity);
        return out;
    }
private:
    g4::ModelConfig config_;
};
}
int main() {
    try {
        Backend backend; g4::engine::Request request;
        auto record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::exl3::identity) throw std::runtime_error("EXL3 receipt identity");
        backend.exl3 = false;
        record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::arithmetic_id) throw std::runtime_error("legacy metadata fallback");
        std::cout << "EXL3 and legacy receipt identities PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "strix/shared_expert_rows_diagnostic_contract.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace strix::shared_expert_rows_diagnostic;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid shared rows contract was admitted");
}
struct Case {
    Shape shape;
    Span input, workspace, output;
    Weights weights;
    explicit Case(unsigned rows, unsigned hidden = 2560U, unsigned intermediate = 640U, unsigned padding = 0U)
        : shape{hidden, intermediate, rows, 2ULL * hidden + padding,
                2ULL * hidden + padding, 8ULL * intermediate + padding} {
        const auto sizes = layout(shape, Arm::shared_q8_rows);
        input = {0x100000000ULL, sizes.input_bytes};
        workspace = {0x200000000ULL, sizes.workspace_bytes};
        output = {0x300000000ULL, sizes.output_bytes};
        weights = {{0x400000000ULL, sizes.gate_up_weight_bytes},
                   {0x500000000ULL, sizes.gate_up_weight_bytes},
                   {0x600000000ULL, sizes.down_weight_bytes},
                   {0x700000000ULL, sizes.scalar_weight_bytes}};
    }
    void valid(Arm arm = Arm::shared_q8_rows) const {
        const auto l = validate(shape, input, weights, workspace, output, arm);
        require(l.launches == (arm == Arm::reference_rows ? 5U * shape.rows : 3U + 2U * shape.rows),
                "wrong actual shared rows launch count");
        require(l.workspace_bytes == (shape.rows - 1U) * shape.workspace_stride_bytes + 8ULL * shape.intermediate,
                "workspace does not preserve gate/up per-row layout");
    }
};
} // namespace
int main() try {
    for (unsigned rows : {2U, 4U, 8U}) for (unsigned hidden : {32U, 96U, 288U, 2560U})
        for (unsigned intermediate : {32U, 96U, 640U}) for (unsigned pad : {0U, 16U}) {
            Case c(rows, hidden, intermediate, pad); c.valid(); c.valid(Arm::reference_rows);
        }
    Case canonical(2U);
    for (unsigned rows : {0U, 1U, 3U, 9U, std::numeric_limits<unsigned>::max()}) {
        auto c = canonical; c.shape.rows = rows; rejects([&] { c.valid(); });
    }
    for (unsigned which = 0; which < 2U; ++which) for (unsigned value : {0U, 31U, 33U, 10272U}) {
        auto c = canonical;
        (which ? c.shape.intermediate : c.shape.hidden) = value;
        rejects([&] { c.valid(); });
    }
    rejects([&] { canonical.valid(static_cast<Arm>(123)); });
    for (unsigned which = 0; which < 3U; ++which) for (unsigned kind = 0; kind < 3U; ++kind) {
        auto c = canonical;
        auto& stride = which == 0U ? c.shape.input_stride_bytes :
                       which == 1U ? c.shape.output_stride_bytes : c.shape.workspace_stride_bytes;
        if (kind == 0U) stride -= 16U;
        if (kind == 1U) ++stride;
        if (kind == 2U) stride = std::numeric_limits<std::uint64_t>::max() - 15U;
        rejects([&] { c.valid(); });
    }
    for (unsigned which = 0; which < 7U; ++which) for (unsigned kind = 0; kind < 5U; ++kind) {
        auto c = canonical;
        Span* spans[]{&c.input, &c.weights.gate_q8, &c.weights.up_q8, &c.weights.down_q8,
                      &c.weights.shared_gate_f32, &c.workspace, &c.output};
        auto& span = *spans[which];
        if (kind == 0U) span.address = 0;
        if (kind == 1U) ++span.address;
        if (kind == 2U) --span.bytes;
        if (kind == 3U) ++span.bytes;
        if (kind == 4U) span.address = std::numeric_limits<std::uintptr_t>::max() - 15U;
        rejects([&] { c.valid(); });
    }
    for (unsigned write = 0; write < 2U; ++write) for (unsigned read = 0; read < 5U; ++read) {
        auto c = canonical;
        const Span reads[]{c.input, c.weights.gate_q8, c.weights.up_q8, c.weights.down_q8, c.weights.shared_gate_f32};
        (write ? c.output : c.workspace).address = reads[read].address;
        rejects([&] { c.valid(); });
    }
    auto c = canonical; c.output.address = c.workspace.address + 16U;
    rejects([&] { c.valid(); });
    // Read-only aliases are legal; the wrapper never writes any input matrix.
    c = canonical; c.weights.up_q8 = c.weights.gate_q8; c.valid();
    std::cout << "shared rows CPU contract source fixtures passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }

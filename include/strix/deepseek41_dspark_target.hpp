#pragma once

// The trimmed host forward as a DSpark speculative target. The drafter reads the inputs of the
// model's last three layers (layers 37, 38, 39 of the full model; layers N-3..N-1 of a trimmed one):
// the input of layer L is the residual stream after layer L-1 (none of those layers carries Engram).

#include "strix/deepseek41_host_forward.hpp"
#include "strix/ds41_dspark.hpp"
#include "strix/ds41_dspark_driver.hpp"

#include <span>
#include <stdexcept>

namespace strix::deepseek41 {

class HostForwardTarget final : public ds41::dspark::SpeculativeTarget {
public:
    explicit HostForwardTarget(HostForward& forward) : forward_(forward) {}
    std::uint32_t step(std::uint32_t token) override {
        (void)forward_.step(token);
        return forward_.next_token();
    }
    void main_hidden(std::span<float> out) const override {
        const auto& streams = forward_.last_step_streams();
        if (streams.size() < 4U) throw std::runtime_error("the DSpark target needs at least 4 layers");
        const std::size_t n = streams.size();
        const auto row = ds41::dspark::main_hidden_row(streams[n - 4U], streams[n - 3U], streams[n - 2U]);
        std::copy(row.begin(), row.end(), out.begin());
    }

private:
    HostForward& forward_;
};

} // namespace strix::deepseek41

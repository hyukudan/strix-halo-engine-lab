#include "../row_epilogue_fixture_cases.hpp"
#include "cuda/core/chain/launch.cuh"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

namespace chain = strix::cuda::core::chain;
namespace data = strix::test::row_epilogue;
using chain::Form;
using Row = std::array<float, data::input.size()>;
constexpr auto width = static_cast<unsigned>(data::input.size());
constexpr auto bytes = sizeof(float) * data::input.size();

void check(cudaError_t result, const char* action) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(result));
}

class DeviceRow {
public:
    DeviceRow() { check(cudaMalloc(&values, bytes), "allocate row"); }
    ~DeviceRow() { cudaFree(values); }
    DeviceRow(const DeviceRow&) = delete;
    DeviceRow& operator=(const DeviceRow&) = delete;

    void upload(const Row& row) {
        check(cudaMemcpy(values, row.data(), bytes, cudaMemcpyHostToDevice), "upload row");
    }
    [[nodiscard]] Row download() const {
        Row row{};
        check(cudaMemcpy(row.data(), values, bytes, cudaMemcpyDeviceToHost), "download row");
        return row;
    }

    float* values = nullptr;
};

void compare_chain() {
    using FirstAdd = chain::Step<Form::residual_add, 2>;
    using Round = chain::Step<Form::store_bf16, 0>;
    using LastAdd = chain::Step<Form::residual_add, 1>;

    DeviceRow input, first, last, sum, rounded, incumbent, fused;
    Row addend{};
    input.upload(data::input);
    addend.fill(data::first_addend);
    first.upload(addend);
    addend.fill(data::last_addend);
    last.upload(addend);

    chain::ChainArgs args{};
    args.aux[2] = first.values;
    args.aux[1] = last.values;
    check(chain::launch<1>(chain::List<FirstAdd>{}, {input.values, sum.values, 1, width}, args, nullptr),
          "launch incumbent combine");
    check(chain::launch<1>(chain::List<Round>{}, {sum.values, rounded.values, 1, width}, args, nullptr),
          "launch incumbent round");
    check(chain::launch<1>(chain::List<LastAdd>{}, {rounded.values, incumbent.values, 1, width}, args, nullptr),
          "launch incumbent residual add");
    check(chain::launch<1>(chain::List<FirstAdd, Round, LastAdd>{}, {input.values, fused.values, 1, width},
                           args, nullptr), "launch fused chain");

    const auto reference = incumbent.download();
    const auto actual = fused.download();
    if (std::memcmp(reference.data(), actual.data(), bytes) != 0)
        throw std::runtime_error("fused chain changed the incumbent bits");
}

}  // namespace

int main() {
    try {
        compare_chain();
        std::puts("PASS B3 CUDA exact add-round-add chain bits");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "B3 CUDA fixture failed: %s\n", error.what());
        return 1;
    }
}

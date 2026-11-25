#include "attn_position_split_cases.hpp"
#include "cuda/core/attn_decode.cuh"

namespace {

namespace cuda_attn = strix::cuda::core::attn;
using namespace b6_fixture;

void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "b6 FAIL %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

template <class T>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
    }
    explicit Device(const std::vector<T>& data) : Device(data.size()) {
        check(cudaMemcpy(ptr_, data.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { (void)cudaFree(ptr_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] T* get() const { return ptr_; }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> data(count_);
        check(cudaMemcpy(data.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return data;
    }

private:
    std::size_t count_;
    T* ptr_ = nullptr;
};

std::vector<__nv_bfloat16> bf16(const std::vector<float>& input) {
    std::vector<__nv_bfloat16> result;
    result.reserve(input.size());
    for (float value : input) result.push_back(__float2bfloat16(value));
    return result;
}

cuda_attn::DecodePlan plan_for(const Case& c, unsigned rows, unsigned context) {
    cuda_attn::DecodeSpec spec{c.dim, c.heads, c.kv_heads, rows, context, c.slots, c.window};
    spec.layout = c.layout == Layout::head ? cuda_attn::KvLayout::head_major :
        c.layout == Layout::slot ? cuda_attn::KvLayout::slot_major : cuda_attn::KvLayout::ring;
    const auto plan = cuda_attn::make_decode_plan(spec, cuda_attn::kNativeSplitHeadDim);
    require(plan.has_value(), "decode plan");
    return *plan;
}

void run(const Case& c, bool general_scores) {
    const auto data = make_data(c, general_scores);
    const Device<float> q(data.q);
    const Device<__nv_bfloat16> k(bf16(data.k)), v(bf16(data.v));
    std::vector<std::uint32_t> positions, planes, selves;
    for (const auto& row : c.rows) {
        positions.push_back(row.pos);
        planes.push_back(row.plane);
        selves.push_back(row.self);
    }
    const Device<std::uint32_t> pos(positions), plane(planes), self(selves);
    const auto batch_plan = plan_for(c, static_cast<unsigned>(c.rows.size()), c.context);
    const auto incumbent_plan = plan_for(c, 1, c.context);
    const Device<float> batch(c.rows.size() * c.row_size()), single(c.row_size());
    const Device<float> scratch(batch_plan.scratch_floats);
    const bool mapped = c.layout != Layout::slot;
    const bool tree = c.layout == Layout::head;
    cuda_attn::DecodeInput in{q.get(), k.get(), v.get()};
    in.first_position = c.rows.front().pos;
    if (mapped) in.row_map = {pos.get(), plane.get(), c.plane_size(), tree ? self.get() : nullptr};
    check(cuda_attn::decode_position_split(batch_plan, in, batch.get(), scratch.get(),
                                          batch_plan.scratch_floats, nullptr), "batch candidate");
    const auto batched = batch.download();
    for (std::size_t i = 0; i < c.rows.size(); ++i) {
        const auto& row = c.rows[i];
        in.q = q.get() + row.query * c.row_size();
        in.first_position = row.pos;
        if (mapped) in.row_map = {pos.get() + i, plane.get() + i, c.plane_size(), tree ? self.get() + i : nullptr};
        const auto single_plan = plan_for(c, 1, row.pos + 1U);
        check(cuda_attn::decode_position_split(single_plan, in, single.get(), scratch.get(),
                                              batch_plan.scratch_floats, nullptr), "singleton candidate");
        const float* start = batched.data() + i * c.row_size();
        const std::vector<float> batch_row(start, start + c.row_size());
        compare("batch vs singleton", batch_row, single.download());
        if (!general_scores) {
            compare("CPU serial order", serial_row(c, data, row), batch_row);
            continue;
        }

        auto reference = in;
        if (!tree) {
            // The incumbent generic path takes consecutive positions in one plane.
            reference.row_map = {};
            reference.k += row.plane * c.plane_size();
            reference.v += row.plane * c.plane_size();
        }
        check(cuda_attn::decode(incumbent_plan, reference, single.get(), scratch.get(),
                               batch_plan.scratch_floats, nullptr), "incumbent");
        compare("incumbent bits", batch_row, single.download());
    }
}

}  // namespace

int main() {
    for (const auto& c : b6_fixture::cases) {
        run(c, false);
        std::printf("b6 CUDA %s singleton and CPU serial bits PASS\n", c.name);
        run(c, true);
        std::printf("b6 CUDA %s singleton and incumbent bits PASS\n", c.name);
    }
}

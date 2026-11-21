#include <strix/weight/admission.hpp>
#include <strix/weight/qwen_qsa_output.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

using namespace strix::weight;

struct FixtureSource final : Source {
    explicit FixtureSource(std::uint64_t bytes) : bytes_(bytes) {}
    Status read(ByteRange range, std::span<Byte> output) const override {
        if (range.offset > bytes_ || range.bytes > bytes_ - range.offset || range.bytes != output.size())
            return {Error::malformed, "fixture read outside extent"};
        std::fill(output.begin(), output.end(), Byte{});
        return {Error::ok, {}};
    }
    std::uint64_t size() const noexcept override { return bytes_; }
    Digest256 identity() const override { return {}; }

private:
    std::uint64_t bytes_;
};

struct TensorType {
    std::uint32_t wire_type;
    std::uint64_t width, outputs, block_elements, block_bytes;
};
constexpr TensorType native{8, 6144, 2560, 32, 34};
constexpr TensorType unsupported{12, 6144, 2560, 256, 144};

void add_tensor(strix::gguf::FileInfo& file, std::string name, const TensorType& type) {
    const auto bytes = (type.width / type.block_elements) * type.block_bytes * type.outputs;
    const auto offset = file.file_size;
    file.file_size += bytes;
    file.tensors.push_back({std::move(name), {type.width, type.outputs}, type.wire_type,
                            offset - file.data_offset, offset, bytes, file.file_size});
}

void admit_fixture(const strix::gguf::FileInfo& file, TensorIndex& index, WeightMap& admitted) {
    auto status = index_gguf(file, std::make_shared<FixtureSource>(file.file_size), index);
    if (status) status = bind_qwen(index, admitted);
    if (!status) throw std::runtime_error(status.detail);
}

// The previous HIP launcher chose dense A8 before wave8, then its typed linear kernel.
// CUDA used native storage for type 8 and the dense A8 projection after upload.
KernelId previous_kernel(Backend backend, bool dense_a8, bool wave8) {
    if (backend == Backend::cuda) return 0x515304;
    if (dense_a8) return 0x515303;
    if (wave8) return 0x515302;
    return 0x515301;
}

struct Route {
    Backend backend;
    bool dense_a8, wave8;
};

} // namespace

bool wfmt_qwen_qsa_output_test() {
    strix::gguf::FileInfo file;
    file.data_offset = file.file_size = 4096;
    for (unsigned layer = 3; layer < 48; layer += 4)
        add_tensor(file, "blk." + std::to_string(layer) + ".attn_output.weight", native);
    add_tensor(file, "blk.0.ssm_out.weight", native);
    TensorIndex index;
    WeightMap admitted;
    admit_fixture(file, index, admitted);

    constexpr Route routes[] = {{Backend::hip, false, false}, {Backend::hip, false, true},
                                {Backend::hip, true, true}, {Backend::cuda, true, false}};
    for (const auto& route : routes) {
        const auto incumbent = route.backend == Backend::hip
            ? hip_qwen_qsa_output_kernel(route.dense_a8, route.wave8) : QwenQsaOutputKernel::cuda_dense_a8;
        QwenQsaOutputPlans plans;
        const auto status = resolve_qwen_qsa_outputs(index, admitted, route.backend, incumbent, plans);
        if (!status) throw std::runtime_error(status.detail);
        if (plans.size() != 12U || plans.contains("blk.0.ssm_out.weight")) return false;
        for (const auto& tensor : file.tensors) {
            if (tensor.name == "blk.0.ssm_out.weight") continue;
            const auto it = plans.find(tensor.name);
            if (it == plans.end()) return false;
            const auto& plan = it->second;
            const auto& selected = plan.launch.selected;
            const auto [width, outputs] = std::pair{tensor.dimensions[0], tensor.dimensions[1]};
            if (!(selected.storage == FormatId{"gguf.q8_0", 1}) || !(selected.image == ImageId{"image.raw", 1}) ||
                selected.strategy != Strategy::native || selected.backend != route.backend ||
                selected.kernel != previous_kernel(route.backend, route.dense_a8, route.wave8) ||
                plan.payload.bytes != (width / 32U) * 34U * outputs || plan.payload.offset != tensor.absolute_offset ||
                plan.layout.block_elements != 32U || plan.layout.block_bytes != 34U || plan.layout.group_elements != 32U ||
                plan.layout.tile_k != 1U || plan.layout.tile_n != 1U || plan.layout.row_alignment != 32U ||
                plan.layout.payload_alignment != 1U || !plan.layout.row_addressable)
                return false;
        }
        const auto json = qwen_qsa_output_plans_json(plans);
        if (json.find("\"tensor\":\"blk.3.attn_output.weight\"") == std::string::npos ||
            json.find("\"kernel\":" + std::to_string(previous_kernel(route.backend, route.dense_a8, route.wave8))) == std::string::npos ||
            json.find("\"bytes\":16711680") == std::string::npos)
            return false;
    }

    strix::gguf::FileInfo alternate;
    alternate.data_offset = alternate.file_size = 4096;
    add_tensor(alternate, "blk.3.attn_output.weight", unsupported);
    admit_fixture(alternate, index, admitted);
    if (admitted.size() != 1U) return false;
    for (const auto& route : routes) {
        const auto incumbent = route.backend == Backend::hip
            ? hip_qwen_qsa_output_kernel(route.dense_a8, route.wave8) : QwenQsaOutputKernel::cuda_dense_a8;
        QwenQsaOutputPlan plan{};
        const auto status = resolve_qwen_qsa_output(admitted.front(), route.backend, incumbent, plan);
        if (status.code != Error::unknown_format || status.detail.find("candidate storage differs from admitted tensor") == std::string::npos ||
            plan.launch.selected.kernel != 0U || plan.payload.bytes != 0U)
            return false;
    }
    return true;
}

#include "qwen35/cuda/qwen27_program_cuda.hpp"
#include "strix/qwen35/config.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace model = strix::qwen35;
namespace adapter = model::cuda_program;
namespace leaf = strix::exec::cuda_step_rows;
namespace exec = strix::exec;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void offline(std::span<const std::byte>, const exec::Grid&, void*) {
    throw std::logic_error("cannot execute a host capture fixture");
}

const auto offline_thunks = [] {
    leaf::Thunks result;
    result.fill(offline);
    return result;
}();

class ResidentFixture {
public:
    explicit ResidentFixture(std::uint32_t context, bool mixed) : position(context), parity(context & 1U) {
        const model::Config c;
        storage = {c.hidden, c.vocab, c.rotary_dim, c.rms_eps,
            {c.heads, c.kv_heads, c.head_dim, 8192, 0.0625F},
            matrix(c.hidden, c.vocab), matrix(c.hidden, c.vocab), cell(), &token, &position, cell(), cell(),
            cell(), cell(), q_gate.data(), cell(), cell(), cell(), cell(), cell(), cell(), cell(), &top, cell()};

        const auto value_width = c.gdn_v_width();
        auto* z = cell();
        auto* gout = cell();
        for (std::uint32_t index = 0; index < c.layers; ++index) {
            auto& visit = visits.at(index);
            visit.input_gain = cell();
            visit.post_gain = cell();
            visit.gate = matrix(c.hidden, c.intermediate);
            visit.up = matrix(c.hidden, c.intermediate);
            visit.down = matrix(c.intermediate, c.hidden);
            if (c.kind(index) == model::LayerKind::gated_delta_net) {
                auto* raw = cell();
                auto* alpha = cell();
                auto* beta = cell();
                adapter::GdnVisit g{
                    {matrix(c.hidden, c.gdn_conv_channels()), matrix(c.hidden, value_width),
                     matrix(c.hidden, c.gdn_value_heads), matrix(c.hidden, c.gdn_value_heads)},
                    {raw, z, beta, alpha},
                    {c.gdn_key_heads, c.gdn_key_dim, c.gdn_value_heads, c.gdn_value_dim, c.gdn_conv_kernel,
                     cell(), cell(), cell(), cell(), cell(), cell(), raw, cell(), alpha, beta, z, gout,
                     &parity, c.rms_eps},
                    matrix(value_width, c.hidden)};
                if (mixed && index == 0) g.inputs[2].format = leaf::Format::bf16;
                visit.mixer = g;
            } else {
                visit.mixer = adapter::AttentionVisit{
                    {matrix(c.hidden, c.q_gate_width()), matrix(c.hidden, c.kv_width()), matrix(c.hidden, c.kv_width())},
                    cell(), cell(), cell(), cell(), matrix(c.heads * c.head_dim, c.hidden)};
            }
        }
    }

    adapter::DecodeStorage storage{};
    std::array<adapter::Visit, 64> visits{};
    std::uint32_t token = 123, position, top = 0, parity;

private:
    float* cell() { return &cells.at(next++); }
    leaf::Matrix matrix(std::uint32_t in, std::uint32_t out) { return {cell(), in, out, leaf::Format::q8}; }
    alignas(16) std::array<float, 2048> cells{};
    std::array<float, 257> q_gate{};
    std::size_t next = 0;
};

// Source-recorded capture, engine.cu:632–663. Use independent geometry and
// compare each payload by member, excluding unspecified C++ padding bytes.
class Capture {
public:
    Capture(const exec::CudaRecipeProgram& result, bool pdl) : result_(result), pdl_(pdl) {}

    template <class Args>
    void node(std::int32_t visit, std::string_view id, const Args& args, std::uint32_t blocks,
              std::uint32_t threads = 256, std::uint32_t blocks_y = 1) {
        const auto& launch = result_.program.launches.at(index_);
        const auto& node = result_.binding.nodes.at(index_);
        require(node.kind == exec::CudaNodeKind::kernel && launch.kernels == 1 && launch.fn == offline &&
                launch.identity == id && launch.layer == visit, "native symbol, node kind or visit differs");
        require(launch.grid.blocks == std::array{blocks, blocks_y, 1U} &&
                launch.grid.threads == std::array{threads, 1U, 1U} &&
                launch.grid.shared_bytes == 0, "native CUDA launch geometry differs");
        require(launch.args_bytes == sizeof(Args), "native argument size differs");
        Args bound;
        std::memcpy(&bound, result_.program.args.data() + launch.args_offset, sizeof(bound));
        require(bound == args, "native argument values differ");
        if (index_ == 0) {
            require(node.after.empty() && launch.after == exec::no_launch, "capture root differs");
        } else {
            require(node.after.size() == 1 && node.after.front() == exec::CudaDependency{index_ - 1,
                    pdl_ ? exec::CudaEdgeKind::programmatic : exec::CudaEdgeKind::completion} &&
                    launch.after == index_ - 1, "capture predecessor edge differs");
        }
        ++index_;
    }

    void finish(std::size_t nodes) const {
        require(index_ == nodes && result_.program.launches.size() == nodes &&
                result_.binding.nodes.size() == nodes && result_.program.predicted.launches == nodes,
                "capture node count differs");
        require(result_.program.capturable && result_.program.key.ctx_bucket == 0, "fixed-grid admission differs");
        require(result_.binding.readback == exec::CudaReadback::resident, "resident output contract differs");
    }

private:
    const exec::CudaRecipeProgram& result_;
    bool pdl_;
    std::size_t index_ = 0;
};

void check_capture(const ResidentFixture& fixture, const exec::CudaRecipeProgram& result, bool pdl, bool mixed) {
    const auto& s = fixture.storage;
    require(exec::cuda_detail::is_completion_chain(result.binding.nodes) == !pdl,
            "native stream capture requires a serial completion chain");
    Capture capture(result, pdl);
    constexpr std::string_view embed = "cuda-sm120-embedding-q8-r1-v1";
    constexpr std::string_view rms = "cuda-sm120-norm-rms-v1";
    constexpr std::string_view mma = "cuda-sm120-dense-rows-mma3-v1";
    constexpr std::string_view ffn = "cuda-sm120-dense-gate-up-mma3-r1-v1";
    constexpr std::string_view rope = "cuda-sm120-rope-device-position-r1-v1";
    constexpr leaf::View h{1, 5120, 5120, 5120};
    constexpr leaf::View qsrc{24, 256, 12288, 512}, qdst{24, 256, 6144, 256}, kv{4, 256, 1024, 256};

    auto dense = [&](std::int32_t visit, const leaf::Matrix& w, const float* x, float* y, const float* residual,
                     std::uint32_t tiles, const void* up = nullptr) {
        capture.node(visit, up ? ffn : mma, leaf::Dense{{w}, {y}, 1, x, residual, up, true}, tiles);
    };

    capture.node(-1, embed, leaf::Embedding{s.embedding, &fixture.token, s.x}, 1);
    for (std::int32_t index = 0; index < 64; ++index) {
        const auto& visit = fixture.visits.at(static_cast<std::size_t>(index));
        capture.node(index, rms, leaf::Rms{s.x, h, visit.input_gain, s.xn, h, 1e-6F}, 1);
        if ((index + 1) % 4 != 0) {
            const auto& g = std::get<adapter::GdnVisit>(visit.mixer);
            if (index == 0 && mixed) {
                constexpr std::array tiles{640U, 384U, 3U, 3U};
                for (std::size_t i = 0; i < tiles.size(); ++i)
                    dense(index, g.inputs[i], s.xn, g.projected[i], nullptr, tiles[i]);
            } else {
                capture.node(index, mma, leaf::Dense{g.inputs, g.projected, 4, s.xn, nullptr, nullptr, true}, 1030);
            }
            capture.node(index, "cuda-sm120-gdn-convolution-parity-r1-v1", g.window, 80, 128);
            capture.node(index, "cuda-sm120-gdn-recurrence-parity-r1-v1", g.window, 48, 128);
            require(g.window.parity == &fixture.parity, "GDN parity must remain device driven");
            dense(index, g.output, g.window.output, s.x, s.x, 320);
        } else {
            const auto& a = std::get<adapter::AttentionVisit>(visit.mixer);
            capture.node(index, mma, leaf::Dense{{a.inputs[0], a.inputs[1], a.inputs[2]}, {s.q_gate, s.k, s.v},
                         3, s.xn, nullptr, nullptr, true}, 896);
            capture.node(index, rms, leaf::Rms{s.q_gate, qsrc, a.query_gain, s.q, qdst, 1e-6F}, 24);
            capture.node(index, rms, leaf::Rms{s.k, kv, a.key_gain, s.k, kv, 1e-6F}, 4);
            capture.node(index, rope, leaf::Rope{s.q, qdst, 64, s.cosines, s.sines, &fixture.position}, 24, 32);
            capture.node(index, rope, leaf::Rope{s.k, kv, 64, s.cosines, s.sines, &fixture.position}, 4, 32);
            const leaf::AttentionShape shape{24, 4, 256, 8192, 0.0625F};
            capture.node(index, "cuda-sm120-kv-store-bf16-linear-r1-v1",
                         leaf::KvStore{shape, s.k, s.v, a.key_cache, a.value_cache, &fixture.position}, 4);
            const leaf::Attention args{shape, s.q, qdst, s.q_gate + 256, qsrc, a.key_cache, a.value_cache,
                                       &fixture.position, s.attention_output, s.scratch};
            capture.node(index, "cuda-sm120-attention-decode-bf16-scores-r1-v1", args, 32, 256, 4);
            capture.node(index, "cuda-sm120-attention-decode-bf16-values-r1-v1", args, 32, 256, 4);
            capture.node(index, "cuda-sm120-attention-decode-bf16-combine-r1-v1", args, 24);
            dense(index, a.output, s.attention_output, s.x, s.x, 320);
        }
        capture.node(index, rms, leaf::Rms{s.x, h, visit.post_gain, s.xn, h, 1e-6F}, 1);
        dense(index, visit.gate, s.xn, s.ffn_hidden, nullptr, 1088, visit.up.data);
        dense(index, visit.down, s.ffn_hidden, s.x, s.x, 320);
    }
    capture.node(-1, rms, leaf::Rms{s.x, h, s.output_gain, s.xn, h, 1e-6F}, 1);
    dense(-1, s.head, s.xn, s.logits, nullptr, 15520);
    const leaf::Pick pick{s.logits, s.argmax_workspace, 248320, s.top};
    capture.node(-1, "cuda-sm120-greedy-partial-pdl-r1-v1", pick, 128);
    capture.node(-1, "cuda-sm120-greedy-final-pdl-index-r1-v1", pick, 1);
    capture.finish(mixed ? 616 : 613);
}

void check(std::uint32_t context, bool pdl, bool mixed) {
    ResidentFixture fixture(context, mixed);
    const auto lower = [&] {
        return adapter::lower(fixture.storage, fixture.visits, leaf::Bindings(offline_thunks), pdl)
            .compile(exec::ProgramKey{}, exec::CudaReadback::resident);
    };
    const auto before = lower();
    check_capture(fixture, before, pdl, mixed);

    // Change the contents of all replay scalars at their original addresses.
    // Compare each typed native argument against the independent expected
    // packet in both passes. This excludes unspecified C++ padding bytes.
    ++fixture.token;
    ++fixture.position;
    fixture.parity ^= 1U;
    fixture.top = 97;
    const auto after = lower();
    require(before.binding == after.binding && before.program.key == after.program.key,
            "replay scalars changed physical plan or program key");
    check_capture(fixture, after, pdl, mixed);
}

}  // namespace

int main() {
    try {
        check(32, false, false);
        check(4096, false, false);
        check(4096, true, true);
        std::cout << "qwen27_program_host PASS contexts=32,4096 nodes=613 mixed_nodes=616\n";
    } catch (const std::exception& error) {
        std::cerr << "qwen27_program_host FAIL " << error.what() << '\n';
        return 1;
    }
}

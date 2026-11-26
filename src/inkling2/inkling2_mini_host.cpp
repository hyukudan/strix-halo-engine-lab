#include "inkling2_mini_host.hpp"
#include "inkling2_mini_expert_layout.hpp"

#include "attn_core/causal_gqa_reference.hpp"
#include "core/linear_microscaled_reference.hpp"
#include "core/linear_reference.hpp"
#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_forward.hpp"
#include "inkling2_mini_dspark_reference.hpp"
#include "inkling2_mini_moe_reference.hpp"
#include "inkling2_mini_state.hpp"
#include "moe_core/grouped_strixpack.hpp"
#include "strix/serial_request.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/pack_plan.hpp"
#include "strix/session/growing_visit_checkpoint.hpp"

#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string_view>

namespace strix::inkling2 {
namespace {

void require_checkpoint(const weight::Status& status) {
    if (!status) throw std::runtime_error(status.detail);
}

std::string text_parameter(std::string_view suffix) {
    return "model.llm." + std::string(suffix);
}

namespace grouped = moe_core::grouped_strixpack;

using VerifyWindow = session::GrowingVisitWindow<MiniCacheTraits, float>;

}  // namespace

struct MiniHostDecoder::Impl {
    Impl(const std::string& path, unsigned capacity, const char* dspark_path, const char* host_pack,
         const char* hot_routes) : cache_plan(capacity, 1U) {
        require_checkpoint(weights.open(path));
        if (host_pack) bind_pack(host_pack);
        if (hot_routes) warm_pack(hot_routes);
        if (dspark_path) {
            if (capacity > kMaxContext - MiniDSparkCheckpoint::block + 1U)
                throw std::invalid_argument("DSpark needs room for a complete block at the context limit");
            dspark.emplace(dspark_path, [this](auto name, auto out, auto first) {
                require_checkpoint(weights.read(text_parameter(name), out, first));
            });
            chain.emplace(MiniDSparkReference::policy(), *dspark);
            tap_row.resize(tap_elements);
        }
    }

    void bind_pack(const char* path) {
        const auto checkpoint = weights.find("model.llm.embed.weight")->payload.source;
        weight::Digest256 digest{};
        require_checkpoint(weight::digest_slice({checkpoint, {0, checkpoint->size()}}, digest));
        require_checkpoint(weight::LowbitPack::open_host(path, digest, pack));
        if (pack.version() != weight::PackVersion::checked)
            throw std::invalid_argument("--host-pack requires Strix pack v3");
        bool selected = false;
        for (auto role : {"attention", "head", "controller_down"})
            selected = selected || !pack.role_entries(role).empty();
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
            for (const auto projection : {MiniExpertProjection::gate_up, MiniExpertProjection::down}) {
                const auto bank_name = mini_expert_slab_names(layer, projection).codes;
                const auto* role = projection == MiniExpertProjection::gate_up ? "expert_gate_up" : "expert_down";
                if (!pack.entry(weight::packed_expert_name(bank_name, 0))) continue;
                std::vector<std::string> names;
                names.reserve(kExperts);
                const auto geometry = mini_expert_geometry(projection);
                for (unsigned expert = 0; expert < kExperts; ++expert) {
                    auto name = weight::packed_expert_name(bank_name, expert);
                    const auto* entry = pack.entry(name);
                    if (!entry || entry->coding.role != role || entry->columns != geometry.inputs ||
                        entry->rows != geometry.outputs || !entry->sp_header)
                        throw std::invalid_argument("--host-pack has an incomplete or incompatible expert bank: " + name);
                    names.push_back(std::move(name));
                }
                const auto index = projection == MiniExpertProjection::gate_up ? 0U : 1U;
                expert_banks[layer][index] = std::make_unique<grouped::ExpertBank>(pack, role, std::move(names));
                selected = true;
            }
        }
        if (!selected) throw std::invalid_argument("--host-pack has no bound Inkling 2 Mini roles");
    }

    void warm_pack(const char* path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::invalid_argument("cannot read --host-pack-hot route counts");
        constexpr std::size_t limit = 1U << 20U;
        std::string text(limit + 1U, '\0');
        file.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!file.eof() || file.bad()) throw std::invalid_argument("--host-pack-hot input exceeds 1 MiB or cannot be read");
        text.resize(static_cast<std::size_t>(file.gcount()));
        const auto document = serve::parse_json_unique(text);
        const auto* routes = document.find("routes");
        if (!routes) throw std::invalid_argument("--host-pack-hot needs a routes array");
        const auto ordered = weight::pack_route_counts(*routes);
        for (const auto& route : ordered)
            if (route.layer >= kPhysicalLayers || route.expert >= kExperts)
                throw std::invalid_argument("--host-pack-hot route outside Inkling 2 Mini geometry");
        for (const auto& route : ordered) {
            if (!route.count) break;
            for (const auto& bank : expert_banks[route.layer]) {
                if (!bank) continue;
                grouped::MatrixView view;
                require_checkpoint(bank->load(static_cast<std::size_t>(route.expert), view));
            }
        }
    }

    bool packed_linear(std::string_view role, std::string_view name, std::span<const float> input,
                       std::span<float> output, std::size_t rows) const {
        const auto tensor = text_parameter(name);
        const auto* entry = pack.entry(tensor);
        if (!entry) return false;
        if (entry->coding.role != role || entry->rows != rows || entry->columns != input.size())
            throw std::invalid_argument("incompatible packed role or matrix dimensions for " + tensor);
        grouped::Matrix matrix;
        require_checkpoint(matrix.load(pack, tensor));
        require_checkpoint(grouped::project_row(matrix.view(), input, output));
        return true;
    }

    void linear(std::string_view role, std::string_view name, std::span<const float> input,
                std::span<float> output) const {
        if (!packed_linear(role, name, input, output, output.size()))
            core::linear_reference(parameter(name, input.size() * output.size()), input, output);
    }

    std::vector<float> parameter(std::string_view name, std::size_t elements, std::size_t first = 0) const {
        std::vector<float> result(elements);
        require_checkpoint(weights.read(text_parameter(name), result, first));
        return result;
    }

    std::vector<float> attention(LayerVisit visit, unsigned position, std::span<const float> q,
                                 std::span<const float> k, std::span<const float> v) {
        auto& state = storage[visit.logical_layer];
        const auto& plan = cache_plan.visit(visit);
        const bool full = is_full_attention(visit.physical_layer);
        const unsigned window = plan.kv.ring;

        state.prepare(plan, position);

        const auto layer = visit.physical_layer;
        const auto query_gain = parameter(mini_layer_parameter(layer, "attn.q_l2_clip.scale"), kQueryWidth);
        const auto relative_gain =
            parameter(mini_layer_parameter(layer, "attn.rel_l2_clip.scale"), kQueryHeads * kRelativeWidth);
        auto key_conv = parameter(mini_layer_parameter(layer, full ? "attn.k_sconv.static_w" : "attn.k_sconv.weight"),
                                  kConvTaps * kKvWidth);
        auto value_conv = parameter(mini_layer_parameter(layer, full ? "attn.v_sconv.static_w" : "attn.v_sconv.weight"),
                                    kConvTaps * kKvWidth);
        if (!full) {
            key_conv = mini_short_conv_weights(key_conv, kKvWidth);
            value_conv = mini_short_conv_weights(value_conv, kKvWidth);
        }
        constexpr auto relative = mini_attention_projection_spec(MiniAttentionProjection::relative_bias);
        const auto relative_source = parameter(mini_layer_parameter(layer, relative.tensor_suffix),
                                               kRelativeWidth * kRelativeExtent);
        std::array<float, kRelativeWidth * kRelativeExtent> relative_table;
        for (std::size_t i = 0; i < kRelativeWidth; ++i) {
            for (std::size_t lag = 0; lag < kRelativeExtent; ++lag) {
                relative_table[lag * kRelativeWidth + i] = relative_source[i * kRelativeExtent + lag];
            }
        }
        const auto project = [&](MiniAttentionProjection kind, std::span<const float> input, std::span<float> output) {
            const auto spec = mini_attention_projection_spec(kind);
            if (input.size() != spec.input_width || output.size() != spec.output_width) {
                throw std::invalid_argument("invalid Inkling 2 Mini projection operands");
            }
            if (spec.transpose_for_rows) {
                core::linear_reference(relative_table, input, output);
            } else {
                linear("attention", mini_layer_parameter(layer, spec.tensor_suffix), input, output);
            }
        };

        const auto inputs = prepare_mini_attention(visit, position, q, k, v,
            {query_gain, relative_gain, key_conv, value_conv}, state.history[0], state.history[1], project);
        std::array<float, kQueryWidth> attended{};
        attn_core::causal_gqa_reference(
            {kQueryHeads, kKvHeads, kHeadWidth, window, attn_core::ReferenceAttentionArithmetic::bf16_eager},
            {position, inputs.query, inputs.key, inputs.value, kRelativeExtent, inputs.relative_logits, inputs.sinks},
            {state.keys, state.values}, attended);
        std::vector<float> hidden(kHiddenWidth);
        project_mini_attention_output(attended, inputs.output_gate, project, hidden);
        return hidden;
    }

    std::vector<float> convolve(LayerVisit visit, bool after_moe, std::span<const float> input) {
        auto& state = storage[visit.logical_layer];
        constexpr core::ShortConvolution plan{kHiddenWidth, kConvTaps, 0U, 1.0F, 2.2360679774997898F};
        const auto suffix = after_moe ? "mlp_sconv.weight" : "attn_sconv.weight";
        const auto weights_by_lag = mini_short_conv_weights(
            parameter(mini_layer_parameter(visit.physical_layer, suffix), kConvTaps * kHiddenWidth), kHiddenWidth);
        std::vector<float> output(kHiddenWidth);
        core::causal_short_convolution(plan, input, weights_by_lag, {}, state.history[after_moe ? 3U : 2U], output);
        mini_round_bf16(output);
        return output;
    }

    std::vector<float> moe(LayerVisit visit, std::span<const float> input) const {
        const auto layer = visit.physical_layer;
        const auto router = parameter(mini_layer_parameter(layer, "mlp.gate.weight"), kExperts * kHiddenWidth);
        const auto bias = parameter(mini_layer_parameter(layer, "mlp.gate.bias"), kExperts);
        const auto project = [&](unsigned expert, strix::moe::ExpertProjection kind,
                                 std::span<const float> row, std::span<float> output) {
            const auto projection = kind == strix::moe::ExpertProjection::gate_up
                ? MiniExpertProjection::gate_up : MiniExpertProjection::down;
            const auto index = projection == MiniExpertProjection::gate_up ? 0U : 1U;
            if (auto* bank = expert_banks[layer][index].get()) {
                grouped::MatrixView view;
                require_checkpoint(bank->load(expert, view));
                require_checkpoint(grouped::project_row(view, row, output));
                return;
            }
            weight::microscaled::MatrixView view;
            require_checkpoint(weights.expert_matrix(layer, expert, projection, view));
            weight::microscaled::StagedMatrix matrix;
            require_checkpoint(matrix.load(view, mini_expert_row_order(projection)));
            using weight::microscaled::MatrixFormat;
            if (matrix.format() == MatrixFormat::nvfp4) {
                core::microscaled_matvec_reference<MatrixFormat::nvfp4>(matrix.packed(), row, output);
            } else {
                core::microscaled_matvec_reference<MatrixFormat::mxfp8>(matrix.packed(), row, output);
            }
        };
        std::vector<float> output(kHiddenWidth);
        mini_moe_reference(input, router, bias, project, output);
        return output;
    }

    std::vector<float> logits(std::span<const float> hidden) const {
        constexpr std::size_t rows_per_read = 128;
        std::vector<float> output(kOutputVocabulary);
        if (!packed_linear("head", "unembed.weight", hidden, output, kVocabulary)) {
            for (std::size_t first = 0; first < output.size(); first += rows_per_read) {
                const auto rows = std::min(rows_per_read, output.size() - first);
                const auto matrix = parameter("unembed.weight", rows * kHiddenWidth, first * kHiddenWidth);
                core::linear_reference(matrix, hidden, std::span<float>(output).subspan(first, rows));
            }
        }
        mini_round_bf16(output);
        for (float& value : output) {
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite Inkling 2 Mini logits");
            value = kLogitSoftcap * std::tanh(value / kLogitSoftcap);
        }
        return output;
    }

    static constexpr auto tap_elements = MiniDSparkTapPlan::width;

    void capture_draft_tap(unsigned visit, std::span<const float> hidden) {
        if (!dspark) return;
        const auto index = MiniDSparkTapPlan::index(visit);
        if (index != MiniDSparkTapPlan::count)
            std::ranges::copy(hidden, std::span(tap_row).subspan(index * kHiddenWidth, kHiddenWidth).begin());
    }

    std::vector<float> step(unsigned token, bool want_logits, bool stage_taps = false) {
        if (failed) throw std::runtime_error("Inkling 2 Mini session failed during a prior step");
        if (!cache_plan.admits(frontier, 1U) || token >= kVocabulary) {
            throw std::out_of_range("Inkling 2 Mini session capacity or token id exceeded");
        }

        failed = true;
        return advance(parameter("embed.weight", kHiddenWidth, std::size_t{token} * kHiddenWidth), want_logits,
                       stage_taps);
    }

    // A provided row (BF16, e.g. a projected image row) in place of a token row: the same embed_norm and forward.
    std::vector<float> step_row(std::span<const std::uint16_t> row, bool want_logits) {
        if (failed) throw std::runtime_error("Inkling 2 Mini session failed during a prior step");
        if (!cache_plan.admits(frontier, 1U) || row.size() != kHiddenWidth)
            throw std::out_of_range("Inkling 2 Mini session capacity or provided row width exceeded");
        failed = true;
        std::vector<float> embedding(kHiddenWidth);
        for (std::size_t i = 0; i < kHiddenWidth; ++i)
            embedding[i] = std::bit_cast<float>(static_cast<std::uint32_t>(row[i]) << 16U);
        return advance(embedding, want_logits);
    }

    std::vector<float> advance(std::span<const float> embedding, bool want_logits, bool stage_taps = false) {
        const auto normalized = mini_rms_norm(embedding, parameter("embed_norm.weight", kHiddenWidth));
        const auto hidden = mini_forward_row(*this, frontier, normalized);
        auto result = want_logits ? logits(hidden) : std::vector<float>{};
        if (dspark) {
            if (stage_taps) staged_taps.insert(staged_taps.end(), tap_row.begin(), tap_row.end());
            else dspark->append_context(tap_row);
        }
        ++frontier;
        failed = false;
        return result;
    }

    MiniTextWeights weights;
    weight::LowbitPack pack;
    std::array<std::array<std::unique_ptr<grouped::ExpertBank>, 2>, kPhysicalLayers> expert_banks;
    MiniSessionStatePlan cache_plan;
    unsigned frontier = 0;
    bool failed = false;
    VerifyWindow::Storage storage;
    VerifyWindow verify_window;
    std::optional<MiniDSparkReference> dspark;
    std::optional<spec::ParallelBlockDrafter> chain;
    std::vector<float> tap_row;
    std::vector<float> staged_taps;
};

MiniHostDecoder::MiniHostDecoder(const std::string& path, unsigned max_positions, const char* dspark,
                                 const char* host_pack, const char* hot_routes)
    : impl_(std::make_unique<Impl>(path, max_positions, dspark, host_pack, hot_routes)) {}
MiniHostDecoder::~MiniHostDecoder() = default;

std::vector<float> MiniHostDecoder::step(unsigned token, bool want_logits) {
    if (impl_->verify_window.pending()) throw std::logic_error("Mini verify window needs a commit");
    return impl_->step(token, want_logits);
}

std::vector<float> MiniHostDecoder::step_row(std::span<const std::uint16_t> row, bool want_logits) {
    if (impl_->verify_window.pending()) throw std::logic_error("Mini verify window needs a commit");
    return impl_->step_row(row, want_logits);
}

void MiniHostDecoder::verify(std::span<const std::uint32_t> inputs, std::vector<std::uint32_t>& greedy) {
    if (impl_->failed || impl_->verify_window.pending() ||
        !impl_->cache_plan.admits_sequence(impl_->frontier, inputs.size()))
        throw std::invalid_argument("invalid Mini CPU verification frontier");
    if (std::ranges::any_of(inputs, [](std::uint32_t id) { return id >= kVocabulary; }))
        throw std::out_of_range("token id exceeds the Mini vocabulary");
    try {
        impl_->verify_window.verify(impl_->cache_plan, impl_->storage, impl_->frontier, inputs, greedy,
            [&](std::uint32_t input) {
                return select_greedy_token(impl_->step(input, true, true), kOutputVocabulary);
            });
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}

void MiniHostDecoder::commit(std::uint32_t rows) {
    if (impl_->failed) throw std::logic_error("Mini session failed during a prior step");
    impl_->frontier -= impl_->verify_window.commit(impl_->storage, rows);
    if (impl_->dspark) {
        impl_->failed = true;
        if (rows != 0U)
            impl_->dspark->append_context(std::span(impl_->staged_taps).first(rows * Impl::tap_elements));
        impl_->staged_taps.clear();
        impl_->failed = false;
    }
}

bool MiniHostDecoder::has_dspark() const noexcept { return impl_->chain.has_value(); }

spec::ChainDrafter* MiniHostDecoder::start_dspark(unsigned anchor, unsigned position) {
    if (!impl_->chain) return nullptr;
    if (position != impl_->frontier || impl_->failed || impl_->verify_window.pending())
        throw std::logic_error("invalid DSpark start frontier");
    impl_->chain->start(anchor, position);
    return &*impl_->chain;
}

}  // namespace strix::inkling2

#include "strix/ernie/calibration.hpp"

#include "core/multimax_reference.hpp"
#include "moe_core/reference.hpp"
#include "strix/ernie/inventory.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/calib_moments.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_plan.hpp"
#include "strix/weight/pack_source.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

namespace {

namespace ernie = strix::ernie;
namespace wei = strix::weight;
using strix::moe::ExpertProjection;
using Path = std::filesystem::path;
using strix::model::narrow;
constexpr auto kLatent = narrow<unsigned>(ernie::data::moe::moe_latent_size);
constexpr auto kInner = narrow<unsigned>(ernie::data::moe::moe_intermediate_size);
constexpr std::string_view kTensor = "model.layers.1.block_sparse_moe.experts.3.w1.weight";

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Compute>
void rejects(Compute compute, const char* message) {
    try {
        compute();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::string read_text(const Path& path) {
    std::ifstream file(path, std::ios::binary);
    check(static_cast<bool>(file), "fixture open");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void calibration_test(const Path& dir) {
    ernie::ExpertCalibration capture;
    const std::array<std::uint32_t, 2> ids{35, 51};
    const std::array<std::uint32_t, 1> new_ids{69};
    const std::array<std::uint32_t, 3> all_ids{35, 51, 69};
    const std::array<std::uint32_t, 1> other_ids{99};
    const auto window = capture.add_window(ids);
    const auto other = capture.add_window(other_ids);
    std::vector<float> latent(kLatent), inner(kInner);
    latent[0] = 2.0F;
    latent[1] = -3.0F;
    inner[0] = -5.0F;
    inner[1] = 6.0F;
    for (const auto step : {window, other}) {
        capture.observe(step, 1, 3, ExpertProjection::gate_up, latent);
        capture.observe(step, 1, 3, ExpertProjection::down, inner);
    }
    capture.observe(other, 2, 2, ExpertProjection::gate_up, latent);
    capture.observe(other, 2, 2, ExpertProjection::down, inner);
    capture.extend_window(window, new_ids);
    check(capture.route_count(1, 3) == 2 && capture.route_count(2, 2) == 1 && capture.route_count(1, 2) == 0,
          "counts for selected experts");
    check(capture.moment_bytes() == 2U * (kLatent + kInner) * sizeof(double), "diagonal bytes for observed inputs");

    const auto description = dir / "base.json";
    std::ofstream(description, std::ios::binary) << R"JSON({"schema":"strix-pack-model/v1","retained_bytes":0,
      "expert_groups":[{"first_layer":1,"layers":2,"experts":4,"projections":[
      {"name":"model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight","rows":1,"columns":128}]}]})JSON";
    const auto path = dir / "capture";
    capture.write(path, description);
    const auto moments = wei::calib::MomentSet::read((path / "layer-1.moments").string(), 1U << 20U);
    check(moments.layout() == wei::calib::MomentLayout::diagonal && moments.inputs().size() == 2, "moment partition");
    check(moments.provenance().windows()[window].digest.bytes == wei::calib::window_identity(all_ids).bytes,
          "actual processed decode positions in provenance");
    const auto* gate = moments.find_consumer(kTensor);
    const auto* up = moments.find_consumer("model.layers.1.block_sparse_moe.experts.3.w3.weight");
    const auto* down = moments.find_consumer("model.layers.1.block_sparse_moe.experts.3.w2.weight");
    check(gate && gate == up && down, "moment inputs by consumer");
    check(gate->rows == 2 && gate->window_rows == std::vector<std::uint64_t>{1, 1} &&
          gate->diagonal(0) == 8.0 && gate->diagonal(1) == 18.0 && gate->diagonal(2) == 0.0, "latent second moments");
    check(down->rows == 2 && down->diagonal(0) == 50.0 && down->diagonal(1) == 72.0, "GLU second moments");
    const auto second = wei::calib::MomentSet::read((path / "layer-2.moments").string(), 1U << 20U);
    check(second.find_consumer(kTensor) == nullptr && second.inputs().front().window_rows ==
          std::vector<std::uint64_t>{0, 1}, "other layer partition");

    const auto routed_description = strix::serve::parse_json_unique(read_text(path / "measured-pack-description.json"));
    const auto plan = wei::plan_strix_pack(routed_description, wei::kSP2WeightStorage, 1, 4000, 0);
    check(plan.hot_experts == 1 && plan.tensors[3].tensor == kTensor &&
          plan.tensors[3].storage == wei::kSP3WeightStorage, "pack planner promotes measured hot expert");
    const std::array<unsigned, 2> layers{1, 2};
    const auto pilot = dir / "pilot.tsv";
    capture.write_slice_plan(pilot, layers, 1, "sp2");
    const auto tsv = read_text(pilot);
    check(tsv.starts_with("tensor\tformat\tbytes\n") && std::ranges::count(tsv, '\n') == 7 &&
          tsv.find("model.layers.2.block_sparse_moe.experts.2.w2.weight\tsp2\t") != std::string::npos,
          "measurement-selected pilot matrices");
    rejects([&] { capture.write(path, description); }, "capture directory cannot be overwritten");
    rejects([&] { capture.observe(window, 0, 0, ExpertProjection::gate_up, latent); }, "only routed layers captured");
    capture.observe(window, 2, 2, ExpertProjection::down, inner);
    rejects([&] { capture.write_slice_plan(dir / "partial.tsv", layers, 1, "sp2"); }, "unpaired capture admitted");
}

void write_checkpoint(const Path& path, char packed_code) {
    auto head = std::string("{\"") + std::string(kTensor) +
                "\":{\"dtype\":\"I8\",\"shape\":[1,64],\"data_offsets\":[0,64]},\"" + std::string(kTensor) +
                "_scale_inv\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,4],\"data_offsets\":[64,68]}}";
    while (head.size() % 8U) head.push_back(' ');
    std::ofstream out(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i)
        out.put(static_cast<char>((static_cast<std::uint64_t>(head.size()) >> (8U * i)) & 0xffU));
    out << head << std::string(64, packed_code) << std::string(4, 127);
    out.close();
    check(static_cast<bool>(out), "fixture checkpoint write");
}

template <class Storage>
void check_override(const Path& checkpoint, const Path& sidecar, wei::FormatId format) {
    wei::PackCheckpoint source;
    check(static_cast<bool>(wei::PackCheckpoint::open(checkpoint.string(), source)), "pack checkpoint open");
    wei::Digest256 checkpoint_id{};
    check(static_cast<bool>(source.digest(checkpoint_id)), "fixture checkpoint digest");
    const auto* raw = source.find(kTensor);
    wei::PackMatrixSource matrix;
    wei::Digest256 matrix_id{};
    check(raw && wei::PackMatrixSource::bind(source, *raw, matrix) && matrix.digest(matrix_id), "expert digest");
    const std::array names{kTensor};
    wei::sp::Header header;
    header.bits = Storage::bits;
    header.table = wei::sp::default_table<Storage::bits>();
    {
        wei::LowbitPackWriter writer(checkpoint_id, sidecar.string(), names);
        writer.add(std::string(kTensor), format, 1, 128, matrix_id, std::vector<std::uint8_t>(Storage::bytes), header);
        check(static_cast<bool>(writer.finish()), "fixture sidecar publish");
    }
    const ernie::PackedExpertOverrides overrides(checkpoint, sidecar, Storage::bytes);
    std::vector<float> input(128, 1.0F);
    std::array<float, 1> output{-1.0F};
    check(!overrides.project("other.weight", input, output) && output[0] == -1.0F, "non-packed projection falls through");
    check(overrides.matrix_count() == 1 && overrides.project(kTensor, input, output) &&
          output[0] == static_cast<float>(128U * header.table[0]), "packed projection arithmetic");
    rejects([&] { const ernie::PackedExpertOverrides limit(checkpoint, sidecar, Storage::bytes - 1U); },
            "packed residency budget");
    rejects([&] { const ernie::PackedExpertOverrides stale(checkpoint.parent_path() / "different.safetensors", sidecar); },
            "checkpoint identity not enforced");
}

void logits_test() {
    const std::array<float, 2> a{1000.0F, 1000.0F};
    const std::array<float, 2> b{0.0F, std::log(3.0F)};
    const double expected = std::log(4.0 / 3.0) * 0.5;
    const auto comparison = ernie::compare_logits(a, b);
    check(std::abs(comparison.kl_reference_to_candidate - expected) < 1.0e-8 &&
          comparison.reference_top1 == 0 && comparison.candidate_top1 == 1 && !comparison.top1_agrees(),
          "hand-computed KL or low-id tie differs");
    check(ernie::compare_logits(a, a).top1_agrees() && ernie::compare_logits(a, a).kl_reference_to_candidate == 0.0,
          "identity score");
    const std::array<float, 2> nonfinite{0.0F, std::numeric_limits<float>::infinity()};
    rejects([&] { static_cast<void>(ernie::compare_logits(a, nonfinite)); }, "nonfinite logits accepted");
}

void multimax_test() {
    const std::array ranges{-1.5078125F, 3.75F, -2.125F, 6.125F};
    const std::array strengths{0.31640625F, -0.2265625F, 0.07177734375F, -0.048095703125F};
    const std::array expected{-1.53125F, -1.9296875F, -1.921875F, -1.5078125F, 1.0F,
                              3.75F, 5.59375F, 6.5F};
    std::array logits{-12.25F, -2.125F, -2.109375F, -1.5078125F, 1.00390625F, 3.75F, 6.125F, 7.4375F};
    strix::core::multimax_bf16_reference(logits, ranges, strengths);
    check(logits == expected, "MultiMax thresholds, input tie, or intermediate BF16 stores");

    const auto inf = std::numeric_limits<float>::infinity();
    std::array nonfinite{std::numeric_limits<float>::quiet_NaN(), inf, -inf};
    strix::core::multimax_bf16_reference(nonfinite, ranges, strengths);
    check(std::ranges::all_of(nonfinite, [](float value) { return std::isnan(value); }),
          "MultiMax must propagate NaN and preserve infinity arithmetic");

    const std::array model_ranges{15.625F, 15.9375F, 8.75F, 0.3671875F};
    const std::array model_strengths{-0.1337890625F, 0.279296875F, -0.09765625F, -0.0301513671875F};
    constexpr std::array<std::uint32_t, 16> input_bits{
        0, 0x80000000, 1, 0x80000001, 0x417a0000, 0x417f0000, 0x410c0000, 0x3ebc0000,
        0x4179ffff, 0x417effff, 0x410bffff, 0x3ebbffff, 0x417a0001, 0x417f0001, 0x410c0001, 0x3ebc0001};
    constexpr std::array<std::uint32_t, 16> output_bits{
        0xc1191380, 0xc1191380, 0xc1191380, 0xc1191380, 0x4109b118, 0x410a0b52, 0x40b6c3ee, 0xc10895e6,
        0x4109b118, 0x410a0b52, 0x40b6c3ec, 0xc10895e6, 0x4109b118, 0x410a0b52, 0x40b6c3ee, 0xc10895e6};
    std::array<float, input_bits.size()> values;
    std::ranges::transform(input_bits, values.begin(), [](auto bits) { return std::bit_cast<float>(bits); });
    strix::core::multimax_f32_reference(values, model_ranges, model_strengths);
    for (std::size_t i = 0; i < values.size(); ++i)
        check(std::bit_cast<std::uint32_t>(values[i]) == output_bits[i], "MultiMax FP32 threshold or rounding");
    nonfinite = {std::numeric_limits<float>::quiet_NaN(), inf, -inf};
    strix::core::multimax_f32_reference(nonfinite, model_ranges, model_strengths);
    check(std::isnan(nonfinite[0]) && std::isnan(nonfinite[1]) && nonfinite[2] == -inf,
          "MultiMax FP32 exceptional values");
}

}  // namespace

int main() {
    const Path dir = "ernie-calibration-fixture";
    try {
        check(std::filesystem::create_directory(dir), "fixture directory already exists");
        try {
            calibration_test(dir);
            const auto checkpoint = dir / "model.safetensors";
            write_checkpoint(checkpoint, 17);
            write_checkpoint(dir / "different.safetensors", 34);
            check_override<wei::sp::SP2Storage>(checkpoint, dir / "sp2.pack", wei::kSP2WeightStorage);
            check_override<wei::sp::SP3Storage>(checkpoint, dir / "sp3.pack", wei::kSP3WeightStorage);
            logits_test();
            multimax_test();
        } catch (...) {
            std::filesystem::remove_all(dir);
            throw;
        }
        std::filesystem::remove_all(dir);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

#include "strix/ernie/calibration.hpp"

#include "moe_core/reference.hpp"
#include "strix/ernie/inventory.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight/calib_moments.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_plan.hpp"
#include "strix/weight/pack_source.hpp"

#include <array>
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

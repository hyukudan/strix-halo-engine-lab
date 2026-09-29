#include "strix/model_manifest.hpp"
#include "strix/platform_probe.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    using strix::ResidencyPolicy;
    using strix::TensorRole;

    expect(strix::classify_tensor_role("token_embd.weight") == TensorRole::embedding,
           "embedding role is classified");
    expect(strix::classify_tensor_role("blk.7.attn_q.weight") == TensorRole::attention_projection,
           "attention role is classified");
    expect(strix::classify_tensor_role("blk.7.experts.12.down_proj.weight") == TensorRole::moe_expert,
           "expert role is classified");
    expect(strix::classify_tensor_role("ple.ngram_embedding.shard_3") == TensorRole::ngram_table,
           "n-gram role is classified");
    expect(strix::classify_tensor_role("blk.7.shared_expert.gate_proj.weight") ==
               TensorRole::dense_projection,
           "shared expert is not misclassified as routed expert");
    expect(strix::classify_tensor_role("blk.7.ffn_gate_up_exps.weight") ==
               TensorRole::moe_expert,
           "Qwen4Exp packed expert tensor is classified as routed expert");
    expect(strix::classify_tensor_role("blk.7.ffn_gate_inp_shexp.weight") ==
               TensorRole::dense_projection,
           "Qwen4Exp shared-expert gate is not classified as router");
    expect(strix::classify_tensor_role("blk.7.hc_attn_down.weight") ==
               TensorRole::dense_projection,
           "Qwen4Exp hyper-connection projection is classified");
    expect(strix::classify_tensor_role("blk.1.ple_key.weight") ==
               TensorRole::ple_parameter,
           "Qwen4Exp PLE parameter is not treated as n-gram table");
    expect(strix::classify_tensor_role("per_layer_token_embd.weight") ==
               TensorRole::ngram_table,
           "per-layer PLE table is classified as n-gram storage");
    expect(strix::classify_tensor_role("blk.3.attn_gate.weight") ==
               TensorRole::attention_projection,
           "Qwen4Exp attention gate is classified with attention projections");
    expect(strix::default_residency(TensorRole::moe_expert) == ResidencyPolicy::resident,
           "experts remain resident by default");
    expect(strix::default_residency(TensorRole::ngram_table) ==
               ResidencyPolicy::file_backed_candidate,
           "n-gram tables are file-backed candidates");

    strix::gguf::FileInfo file;
    strix::gguf::MetadataEntry architecture;
    architecture.key = "general.architecture";
    architecture.type = strix::gguf::ValueType::string;
    architecture.string_value = "test_arch";
    file.metadata.push_back(architecture);

    strix::gguf::TensorInfo attention;
    attention.name = "blk.7.attn_q.weight";
    attention.dimensions = {128, 128};
    attention.ggml_type = 1;
    attention.absolute_offset = 4096;
    attention.byte_size = 32768;
    attention.absolute_end = 36864;
    file.tensors.push_back(attention);

    strix::gguf::TensorInfo ngram;
    ngram.name = "ple.ngram_embedding.shard_3";
    ngram.dimensions = {128, 128};
    ngram.ggml_type = 1;
    ngram.relative_offset = 32768;
    ngram.absolute_offset = 36864;
    ngram.byte_size = 32768;
    ngram.absolute_end = 69632;
    file.tensors.push_back(ngram);

    const auto manifest = strix::build_model_manifest(file);
    expect(manifest.architecture == "test_arch", "architecture is copied into manifest");
    expect(manifest.layout_identity != 0 &&
               manifest.layout_identity ==
                   strix::model_manifest_layout_identity(manifest),
           "manifest layout identity is stable");
    expect(manifest.total_tensor_bytes == 65536, "manifest totals tensor bytes");
    expect(manifest.file_backed_candidate_bytes == 32768,
           "manifest totals file-backed candidates");
    expect(manifest.tensors[0].layer.has_value() && *manifest.tensors[0].layer == 7,
           "manifest extracts layer index");
    expect(manifest.tensors[0].placement.kind == strix::PlacementKind::local &&
               manifest.tensors[0].placement.node == 0,
           "manifest defaults to local node zero");

    auto changed_manifest = manifest;
    ++changed_manifest.tensors[0].absolute_offset;
    expect(strix::model_manifest_layout_identity(changed_manifest) !=
               manifest.layout_identity,
           "tensor-layout change invalidates manifest layout identity");

    const auto root = std::filesystem::temp_directory_path() / "strix-platform-probe-test";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root / "proc");
    std::filesystem::create_directories(root / "opt/rocm/.info");
    std::filesystem::create_directories(root / "opt/rocm/bin");
    std::filesystem::create_directories(root / "sys/class/kfd/kfd/topology/nodes/1");
    std::filesystem::create_directories(root / "sys/bus/thunderbolt/devices/domain0");
    std::filesystem::create_directories(root / "sys/bus/thunderbolt/devices/0-0");
    std::filesystem::create_directories(root / "sys/module/thunderbolt_stream");
    std::filesystem::create_directories(root / "sys/kernel/config/thunderbolt/stream/0-1.0/data");
    std::filesystem::create_directories(root / "dev");

    {
        std::ofstream out(root / "proc/meminfo");
        out << "MemTotal: 131072 kB\n";
    }
    {
        std::ofstream out(root / "opt/rocm/.info/version");
        out << "7.2.0\n";
    }
    {
        std::ofstream out(root / "opt/rocm/bin/hipcc");
        out << '\n';
    }
    {
        std::ofstream out(root / "sys/class/kfd/kfd/topology/nodes/1/properties");
        out << "gfx_target_version 110501\n"
            << "simd_count 80\n"
            << "simd_per_cu 2\n";
    }
    {
        std::ofstream out(root / "dev/tbstream0");
        out << '\n';
    }

    {
        const auto stream = root / "sys/kernel/config/thunderbolt/stream/0-1.0/data";
        std::ofstream(stream / "index") << "0\n";
        std::ofstream(stream / "busy_poll") << "1\n";
        std::ofstream(stream / "ring_size") << "512\n";
        std::ofstream(stream / "throttling") << "4096\n";
        std::ofstream(stream / "in_hopid") << "8\n";
        std::ofstream(stream / "out_hopid") << "9\n";
    }


    const auto platform = strix::probe_platform(root);
    expect(platform.mem_total_bytes == 131072ULL * 1024ULL,
           "platform probe parses MemTotal");
    expect(platform.rocm_version.has_value() && *platform.rocm_version == "7.2.0",
           "platform probe reads ROCm version");
    expect(platform.hipcc_present, "platform probe sees hipcc");
    expect(platform.gpu_nodes.size() == 1 && platform.gpu_nodes[0].gfx_arch == "gfx1151",
           "platform probe maps KFD target to gfx1151");
    expect(platform.gpu_nodes[0].compute_units.has_value() &&
               *platform.gpu_nodes[0].compute_units == 40,
           "platform probe derives compute units");
    expect(platform.usb4.domains.size() == 1 && platform.usb4.host_routers.size() == 1,
           "platform probe discovers USB4 domain and host router");
    expect(platform.usb4.stream_devices.size() == 1 &&
               platform.usb4.stream_devices[0] == "tbstream0",
           "platform probe discovers USB4 stream device");
    expect(platform.usb4.stream_driver_present,
           "platform probe detects USB4 stream driver");
    expect(platform.usb4.configured_streams.size() == 1,
           "platform probe discovers configured USB4 stream");
    if (platform.usb4.configured_streams.size() == 1) {
        const auto& stream = platform.usb4.configured_streams[0];
        expect(stream.service == "0-1.0" && stream.name == "data" &&
                   stream.index.has_value() && *stream.index == 0 &&
                   stream.busy_poll.has_value() && *stream.busy_poll &&
                   stream.ring_size.has_value() && *stream.ring_size == 512 &&
                   stream.throttling_ns.has_value() && *stream.throttling_ns == 4096 &&
                   stream.in_hopid.has_value() && *stream.in_hopid == 8 &&
                   stream.out_hopid.has_value() && *stream.out_hopid == 9,
               "configured USB4 stream attributes are parsed");
    }

    std::filesystem::remove_all(root, error);

    if (failures == 0) {
        std::cout << "manifest/platform tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

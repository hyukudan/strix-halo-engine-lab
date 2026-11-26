#pragma once
#include "source.hpp"
#include "strix/gguf_reader.hpp"
#include "shard_name.hpp"
#include <limits>
namespace strix::weight {
enum class Container : std::uint8_t { gguf, safetensors, pack };
struct RawTensor {
  TensorId id{}; std::string name; std::string wire_dtype;
  std::uint32_t wire_type{}; Shape shape{}; Slice payload;
  // Full logical extents as stored by the container (any rank). A tensor whose rank exceeds the
  // WeightView rank (Shape::dims) is indexed opaquely: name, dtype, extents and byte range are
  // kept, shape stays empty, and it can never be admitted as a WeightView. Engines that need such
  // a tensor (for example a Conv3d patch embedding) read it by its byte range.
  std::vector<std::uint64_t> extents;
  bool opaque() const noexcept { return extents.size()>shape.dims.size(); }
};
struct TensorIndex {
  Container container{}; std::string schema;
  std::vector<RawTensor> tensors;
  std::vector<NamedSlice> metadata_blobs;
  std::shared_ptr<const Source> owner;
  Digest256 header_identity{};
  std::vector<std::pair<std::string,std::string>> metadata;
};
struct LegacyConversion { TensorId source; FormatId target; std::string recipe; };
struct AdmissionRequest {
  const TensorIndex* index{}; std::span<const TensorBinding> bindings;
  std::span<const Parameters> tensor_parameters;
};
Status index_gguf(std::shared_ptr<const Source>, TensorIndex&);
Status namespace_index(TensorIndex&, std::string shard);
Status index_gguf(const gguf::FileInfo&, std::shared_ptr<const Source>, TensorIndex&);
Status index_safetensors(std::shared_ptr<const Source>, TensorIndex&);
struct SafetensorsIndexLimits {
  static constexpr std::size_t default_header_bytes = 64U * 1024U * 1024U;
  std::size_t max_header_bytes = default_header_bytes;
  std::size_t max_json_values = std::numeric_limits<std::size_t>::max();
  std::size_t max_tensors = std::numeric_limits<std::size_t>::max();
  std::size_t max_metadata_entries = std::numeric_limits<std::size_t>::max();
};
Status index_safetensors(std::shared_ptr<const Source>, TensorIndex&, const SafetensorsIndexLimits&);
struct SafetensorsManifestEntry {
  std::string_view name, shard;
  std::uint64_t bytes;
};
// Reconcile all indexed payloads with a weight_map. For a partial download,
// only entries belonging to explicitly absent shards can be missing.
Status validate_safetensors_manifest(std::span<const SafetensorsManifestEntry>, std::string_view json,
                                     std::span<const std::string> missing_shards = {});
// Verify the supported single-shard manifest contract without reading any payload.
Status validate_safetensors_shard_manifest(const TensorIndex&, std::string_view json, std::string_view shard);
Status index_pack(std::shared_ptr<const Source>, TensorIndex&);
Status admit(const Registry&, const AdmissionRequest&, WeightMap&);
Status validate_binding(const Registry&, const WeightMapEntry&);
std::string describe_admission(const WeightMap&);
FormatId safetensors_format(std::string_view) noexcept;
Status bind_qwen(const TensorIndex&, WeightMap&);
Status bind_ds41(const TensorIndex&, WeightMap&);
Status bind_minimax(const TensorIndex&, WeightMap&);
Status bind_glm(const TensorIndex&, WeightMap&);
Status bind_gemma4(const TensorIndex&, WeightMap&);
Status bind_h3(const TensorIndex&, WeightMap&);
Status bind_vision(const TensorIndex&, WeightMap&);
}

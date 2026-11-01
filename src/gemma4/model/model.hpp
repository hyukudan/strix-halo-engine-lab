#pragma once

#include "gemma4/contracts/g4_types.hpp"
#include "strix/serve/json.hpp"
#include <filesystem>
#include <map>

namespace strix::gemma4 {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] const std::uint8_t* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }
private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};
struct SafeTensor {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0, bytes = 0;
};
struct SafeFile {
    std::shared_ptr<MappedFile> storage;
    std::vector<SafeTensor> tensors;
};
[[nodiscard]] SafeFile read_safetensors(const std::filesystem::path& path);
[[nodiscard]] ModelConfig parse_text_config(const std::filesystem::path& path);
[[nodiscard]] std::map<std::string, std::vector<std::uint64_t>> required_tensors(const ModelConfig& config);
[[nodiscard]] bool frozen_compatibility_tensor(const std::string& name);
[[nodiscard]] ModelManifest load_safetensors_model(const std::filesystem::path& directory);
// GGUF norms are already direct multipliers; no offset adjustment or Q/K permutation.
[[nodiscard]] ModelManifest load_gguf_model(const std::filesystem::path& path);
[[nodiscard]] std::string gguf_tensor_name(const std::string& name);
[[nodiscard]] const TensorRecord& tensor(const ModelManifest& model, const std::string& name);
[[nodiscard]] float weight_at(const LinearView& view, std::uint32_t row, std::uint32_t column);
[[nodiscard]] std::size_t packed_row_bytes(WeightFormat format, std::uint32_t inputs);
void validate_finite(const LinearView& view);
[[nodiscard]] std::string hash_bytes(const void* bytes, std::size_t size);
[[nodiscard]] serve::Json read_json(const std::filesystem::path& path);

namespace model_detail {
[[nodiscard]] const serve::Json& require(const serve::Json& value, const char* key);
[[nodiscard]] std::uint64_t integer(const serve::Json& value);
[[nodiscard]] LinearView make_view(const std::uint8_t* data, WeightFormat format, const std::vector<std::uint64_t>& shape);
void validate_manifest(ModelManifest& model);
}
} // namespace strix::gemma4

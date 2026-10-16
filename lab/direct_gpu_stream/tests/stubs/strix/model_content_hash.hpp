#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <string>
namespace strix {
struct ModelContentIdentity { std::array<std::uint8_t,32> bytes{}; };
ModelContentIdentity sha256_bytes(std::span<const std::byte>);
std::string model_content_identity_hex(const ModelContentIdentity&);
}

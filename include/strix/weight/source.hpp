#pragma once
#include "types.hpp"
namespace strix::weight {
struct Source {
  virtual ~Source() = default;
  virtual Status read(ByteRange, std::span<Byte>) const = 0;
  virtual std::uint64_t size() const noexcept = 0;
  virtual Digest256 identity() const = 0;
};
Status open_source(const std::string&, std::shared_ptr<const Source>&);
}

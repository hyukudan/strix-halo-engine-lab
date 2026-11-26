#pragma once
#include "types.hpp"
#include <bit>
#include <istream>
#include <optional>
namespace strix::weight {
// Decode one little-endian F32 from four source bytes, at any byte alignment.
inline float decode_f32_le(const Byte* bytes) {
    std::uint32_t bits = 0;
    for (unsigned i = 0; i < 4; ++i) bits |= std::to_integer<std::uint32_t>(bytes[i]) << (8U * i);
    return std::bit_cast<float>(bits);
}

// The owner pins the immutable file view across pack/source destruction.
struct MappedBytes {
    std::shared_ptr<const void> owner;
    std::span<const Byte> bytes;
};
struct Source {
  virtual ~Source() = default;
  virtual Status read(ByteRange, std::span<Byte>) const = 0;
  virtual std::uint64_t size() const noexcept = 0;
  virtual Digest256 identity() const = 0;
  virtual Status map(ByteRange, MappedBytes&) const { return {Error::unsupported_arithmetic, "source is not mappable"}; }
  // Stable shard namespace, independent of a tensor name. Not a content digest.
  virtual std::string shard_name() const { return {}; }
  // Optional positioned-I/O cache hint; it never changes the stored bytes.
  virtual void discard(ByteRange) const noexcept {}
};
Digest256 digest_bytes(std::span<const Byte>);
Status open_source(const std::string&, std::shared_ptr<const Source>&, std::string shard = {});
// Device and file number of an opened file: equal keys name one file, whatever its names.
struct FileKey {
  std::uint64_t device{}, file{};
  bool operator==(const FileKey&) const = default;
};
// The key of the file a source opened from a file holds (nullopt: not a file source).
std::optional<FileKey> source_file_key(const Source&);
// While alive, open_source of exactly `path` returns `source` (a file the caller already holds open, under the shard
// name open_source gives that path) instead of looking the name up again: two loaders of one run read one file. One
// pin per process at a time; uses() counts the opens it served.
class SourcePin {
public:
  SourcePin(std::string path, std::shared_ptr<const Source> source);
  ~SourcePin();
  SourcePin(const SourcePin&) = delete;
  SourcePin& operator=(const SourcePin&) = delete;
  [[nodiscard]] std::size_t uses() const;
};
std::shared_ptr<const Source> memory_source(std::span<const Byte>, std::string shard = {});
// The span must remain immutable; owner retains the complete referenced allocation.
std::shared_ptr<const Source> view_source(std::span<const Byte>, std::shared_ptr<const void>, std::string shard = {});
std::unique_ptr<std::istream> source_stream(std::shared_ptr<const Source>);
TensorId tensor_id(const Source&, std::string_view, const Digest256& header = {});
Status digest_slice(const Slice&, Digest256&);
std::shared_ptr<const Source> namespace_source(std::shared_ptr<const Source>, std::string shard);
Status read_slice(const Slice&, std::span<Byte>);
// Create a positioned view relative to a slice without reading its contents.
inline Status subslice(const Slice& parent, ByteRange range, Slice& out) {
    if (!parent.source) return {Error::stale_source, "missing source ownership"};
    if (parent.range.offset > parent.source->size() ||
        parent.range.bytes > parent.source->size() - parent.range.offset ||
        range.offset > parent.range.bytes || range.bytes > parent.range.bytes - range.offset)
        return {Error::malformed, "subslice outside its source range"};
    out = {parent.source, {parent.range.offset + range.offset, range.bytes}};
    return {Error::ok, {}};
}
}

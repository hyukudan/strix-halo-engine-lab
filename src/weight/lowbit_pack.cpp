#include "strix/weight/lowbit_pack.hpp"

#include "core/sha256.hpp"
#include "strix/weight/host_q8.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <set>

namespace strix::weight {
namespace {

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kMaxHeader = 64ULL << 20;

Status fail(Error code, std::string detail) { return {code, std::move(detail)}; }
const Status kOk{Error::ok, {}};

std::uint64_t align_up(std::uint64_t v) { return (v + kAlign - 1U) / kAlign * kAlign; }

bool supported(FormatId id) { return id == kQ4G64WeightStorage || id == kQ6G64WeightStorage; }

std::optional<FormatId> parse_format(std::string_view name, std::uint32_t version) {
    for (const auto id : {kQ4G64WeightStorage, kQ6G64WeightStorage})
        if (name == id.name && version == id.revision) return id;
    return std::nullopt;
}

std::string hex(const Digest256& d) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    for (const auto b : d.bytes) {
        s.push_back(digits[b >> 4U]);
        s.push_back(digits[b & 15U]);
    }
    return s;
}

bool parse_hex(std::string_view s, Digest256& d) {
    if (s.size() != 64U) return false;
    const auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (std::size_t i = 0; i < 32U; ++i) {
        const int hi = nib(s[2U * i]), lo = nib(s[2U * i + 1U]);
        if (hi < 0 || lo < 0) return false;
        d.bytes[i] = static_cast<std::uint8_t>(hi * 16 + lo);
    }
    return true;
}

template <class T>
bool parse_num(std::string_view s, T& v) {
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return ec == std::errc{} && p == s.data() + s.size();
}

std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t at = 0;
    while (at <= line.size()) {
        const auto next = line.find(' ', at);
        if (next == std::string_view::npos) {
            out.push_back(line.substr(at));
            break;
        }
        out.push_back(line.substr(at, next - at));
        at = next + 1U;
    }
    return out;
}

Status expected_bytes(FormatId storage, std::uint64_t rows, std::uint64_t columns, std::uint64_t& bytes) {
    Shape shape{};
    shape.rank = 2;
    shape.dims = {rows, columns, 0, 0};
    MatrixLoadGeometry geometry{};
    if (const auto status = matrix_load_geometry(storage, shape, geometry); !status) return status;
    bytes = geometry.bytes;
    return kOk;
}

}  // namespace

void LowbitPackWriter::add(std::string tensor, FormatId storage, std::uint64_t rows, std::uint64_t columns,
                           const Digest256& source, std::vector<std::uint8_t> bytes) {
    Item item;
    item.entry.tensor = std::move(tensor);
    item.entry.storage = storage;
    item.entry.rows = rows;
    item.entry.columns = columns;
    item.entry.bytes = bytes.size();
    item.entry.source = source;
    item.payload = std::move(bytes);
    items_.push_back(std::move(item));
}

Status LowbitPackWriter::write(const std::string& path) const {
    if (checkpoint_.bytes == Digest256{}.bytes) return fail(Error::malformed, "low-bit pack without a checkpoint identity");
    std::set<std::string> names;
    for (const auto& item : items_) {
        const auto& e = item.entry;
        if (e.tensor.empty() || std::ranges::any_of(e.tensor, [](char c) { return c <= ' ' || c == 127; }))
            return fail(Error::malformed, "invalid tensor name in low-bit pack: " + e.tensor);
        if (!names.insert(e.tensor).second) return fail(Error::malformed, "duplicate tensor " + e.tensor);
        if (!supported(e.storage)) return fail(Error::unknown_format, "unsupported pack format for " + e.tensor);
        std::uint64_t want = 0;
        if (const auto status = expected_bytes(e.storage, e.rows, e.columns, want); !status) return status;
        if (want != e.bytes) return fail(Error::malformed, "payload size mismatch for " + e.tensor);
    }
    // Header size depends on offsets only through digit counts; iterate to a fixed point.
    std::vector<std::uint64_t> offsets(items_.size(), 0);
    std::string header;
    const auto build = [&] {
        header = std::string(kLowbitPackTag) + " " + std::to_string(items_.size()) + "\ncheckpoint " + hex(checkpoint_) + "\n";
        for (std::size_t i = 0; i < items_.size(); ++i) {
            const auto& e = items_[i].entry;
            header += "tensor " + e.tensor + " " + std::string(e.storage.name) + " " +
                      std::to_string(e.storage.revision) + " " + std::to_string(e.rows) + " " +
                      std::to_string(e.columns) + " " + std::to_string(offsets[i]) + " " +
                      std::to_string(e.bytes) + " " + hex(e.source) + "\n";
        }
        header += "end\n";
    };
    for (int pass = 0; pass < 8; ++pass) {
        build();
        auto at = align_up(header.size());
        std::vector<std::uint64_t> next(items_.size());
        for (std::size_t i = 0; i < items_.size(); ++i) {
            next[i] = at;
            at = align_up(at + items_[i].entry.bytes);
        }
        if (next == offsets) break;
        offsets = std::move(next);
    }
    build();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return fail(Error::malformed, "cannot create " + path);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    auto pos = static_cast<std::uint64_t>(header.size());
    const auto pad_to = [&](std::uint64_t target, char fill) {
        const std::string pad(static_cast<std::size_t>(target - pos), fill);
        out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
        pos = target;
    };
    pad_to(align_up(pos), '\n');
    for (std::size_t i = 0; i < items_.size(); ++i) {
        pad_to(offsets[i], '\0');
        out.write(reinterpret_cast<const char*>(items_[i].payload.data()),
                  static_cast<std::streamsize>(items_[i].payload.size()));
        pos += items_[i].payload.size();
    }
    out.flush();
    if (!out) return fail(Error::malformed, "write failed for " + path);
    return kOk;
}

Status LowbitPack::open(const std::string& path, const Digest256& checkpoint, LowbitPack& out) {
    if (checkpoint.bytes == Digest256{}.bytes)
        return fail(Error::stale_source, "low-bit pack " + path + " needs the checkpoint identity it is bound to");
    // One open file for the framing, the identity and the resident slices handed out by find().
    std::shared_ptr<const Source> file;
    if (const auto status = open_source(path, file); !status)
        return fail(Error::malformed, "cannot open low-bit pack " + path + ": " + status.detail);
    const auto file_size = file->size();
    const auto read_at = [&](std::uint64_t at, std::span<Byte> into) { return file->read({at, into.size()}, into); };
    // Read the header block by block until the terminating "end" line is complete.
    std::string text;
    std::size_t header_end = std::string::npos;
    std::vector<Byte> block;
    while (header_end == std::string::npos) {
        if (text.size() >= file_size || text.size() >= kMaxHeader)
            return fail(Error::malformed, "low-bit pack header not terminated: " + path);
        block.resize(static_cast<std::size_t>(std::min<std::uint64_t>(kAlign, file_size - text.size())));
        if (const auto status = read_at(text.size(), block); !status) return fail(Error::malformed, "short read of " + path);
        text.append(reinterpret_cast<const char*>(block.data()), block.size());
        const auto at = text.find("\nend\n");
        if (at != std::string::npos) header_end = at + 5U;
    }
    const auto header = std::string_view(text).substr(0, header_end);
    const auto padded = align_up(header_end);
    if (padded > file_size) return fail(Error::malformed, "low-bit pack header padding truncated");
    for (std::size_t i = header_end; i < text.size() && i < padded; ++i)
        if (text[i] != '\n') return fail(Error::malformed, "low-bit pack header padding is not newline");

    std::vector<std::string_view> lines;
    for (std::size_t at = 0; at < header.size();) {
        const auto nl = header.find('\n', at);
        lines.push_back(header.substr(at, nl - at));
        at = nl + 1U;
    }
    const auto first = split(lines.front());
    std::uint64_t count = 0;
    if (first.size() != 2U || first[0] != kLowbitPackTag || !parse_num(first[1], count))
        return fail(Error::malformed, "bad low-bit pack tag in " + path + " (expected " + std::string(kLowbitPackTag) + ")");
    if (lines.size() != count + 3U || lines.back() != "end")
        return fail(Error::malformed, "low-bit pack entry count mismatch in " + path);
    const auto bound = split(lines[1]);
    Digest256 declared{};
    if (bound.size() != 2U || bound[0] != "checkpoint" || !parse_hex(bound[1], declared))
        return fail(Error::malformed, "low-bit pack " + path + " lacks its checkpoint line");
    if (declared.bytes != checkpoint.bytes)
        return fail(Error::stale_source, "low-bit pack " + path + " was coded from checkpoint " + hex(declared).substr(0, 16) +
                                             ", the model is " + hex(checkpoint).substr(0, 16));

    std::vector<LowbitPackEntry> entries;
    std::set<std::string, std::less<>> names;
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto f = split(lines[static_cast<std::size_t>(i) + 2U]);
        std::uint32_t version = 0;
        LowbitPackEntry e{};
        if (f.size() != 9U || f[0] != "tensor" || f[1].empty() || !parse_num(f[3], version) ||
            !parse_num(f[4], e.rows) || !parse_num(f[5], e.columns) || !parse_num(f[6], e.offset) ||
            !parse_num(f[7], e.bytes) || !parse_hex(f[8], e.source))
            return fail(Error::malformed, "bad low-bit pack entry line " + std::to_string(i + 3U));
        e.tensor = std::string(f[1]);
        const auto storage = parse_format(f[2], version);
        if (!storage) return fail(Error::unknown_format, "unsupported low-bit pack format for " + e.tensor);
        e.storage = *storage;
        if (!names.insert(e.tensor).second) return fail(Error::malformed, "duplicate tensor " + e.tensor);
        std::uint64_t want = 0;
        if (const auto status = expected_bytes(e.storage, e.rows, e.columns, want); !status)
            return fail(status.code, e.tensor + ": " + status.detail);
        if (want != e.bytes) return fail(Error::malformed, "payload size mismatch for " + e.tensor);
        if (e.offset % kAlign != 0U || e.offset < padded || e.offset > file_size || e.bytes > file_size - e.offset)
            return fail(Error::malformed, "payload outside file for " + e.tensor);
        entries.push_back(std::move(e));
    }

    // The identity names the content: the header and every payload byte, so packs coded from other statistics or
    // settings differ even when their tensors, shapes and sources match.
    core::Sha256 hash;
    hash.update(header.data(), header.size());
    std::vector<Byte> chunk(std::size_t{1} << 20);
    for (const auto& e : entries)
        for (std::uint64_t done = 0; done < e.bytes;) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(e.bytes - done, chunk.size()));
            if (const auto status = read_at(e.offset + done, std::span(chunk).first(n)); !status)
                return fail(Error::malformed, "short read of " + path);
            hash.update(chunk.data(), n);
            done += n;
        }
    out.path_ = path;
    out.file_ = std::move(file);
    out.entries_ = std::move(entries);
    out.identity_ = Digest256{hash.finish()};
    out.checkpoint_ = declared;
    return kOk;
}

Status LowbitPack::find(std::string_view tensor, FormatId storage, const RawTensor& source,
                        std::optional<Slice>& out) const {
    out.reset();
    const auto it = std::ranges::find(entries_, tensor, &LowbitPackEntry::tensor);
    if (it == entries_.end()) return kOk;
    const std::string name(tensor);
    if (it->storage != storage) return fail(Error::malformed, "pack storage mismatch for " + name);
    if (source.shape.rank != 2U || it->rows != source.shape.dims[0] || it->columns != source.shape.dims[1])
        return fail(Error::malformed, "pack shape mismatch for " + name);
    std::uint64_t want = 0;
    if (const auto status = expected_bytes(storage, source.shape.dims[0], source.shape.dims[1], want); !status)
        return fail(status.code, "pack geometry for " + name + ": " + status.detail);
    if (want != it->bytes) return fail(Error::malformed, "pack byte count mismatch for " + name);
    Digest256 digest{};
    if (const auto status = digest_slice(source.payload, digest); !status) return status;
    if (digest.bytes != it->source.bytes) return fail(Error::stale_source, "pack source digest mismatch for " + name);
    out = Slice{file_, {it->offset, it->bytes}};
    return kOk;
}

}  // namespace strix::weight

#include "strix/weight/lowbit_pack.hpp"

#include "core/sha256.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/pack_source.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <set>

namespace strix::weight {
namespace {

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kMaxHeader = 64ULL << 20;

Status fail(Error code, std::string detail) { return {code, std::move(detail)}; }
const Status kOk{Error::ok, {}};

std::uint64_t align_up(std::uint64_t v) { return (v + kAlign - 1U) / kAlign * kAlign; }

bool supported(FormatId id) {
    return id == kQ4G64WeightStorage || id == kQ6G64WeightStorage || id == kSP4TWeightStorage || is_strixpack_storage(id);
}

std::optional<FormatId> parse_format(std::string_view name, std::uint32_t version) {
    for (const auto id : {kQ4G64WeightStorage, kQ6G64WeightStorage, kSP4TWeightStorage,
                          kSP4WeightStorage, kSP3WeightStorage, kSP2WeightStorage})
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

std::string header_hex(const sp::Header& h) {
    std::uint8_t bytes[sp::kHeaderBytes];
    h.write(bytes);
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    for (const auto b : bytes) {
        s.push_back(digits[b >> 4U]);
        s.push_back(digits[b & 15U]);
    }
    return s;
}

// 32 lowercase hex digits -> a header of the storage's bit width: "" on success, otherwise the first violated rule.
// storage_bits 0 (a storage that carries no header) checks the header against its own width, so the storage rule
// below names the defect.
std::string parse_header(std::string_view s, unsigned storage_bits, sp::Header& out) {
    if (s.size() != 2U * sp::kHeaderBytes) return "bad header hex";
    std::uint8_t bytes[sp::kHeaderBytes];
    for (std::size_t i = 0; i < sp::kHeaderBytes; ++i) {
        unsigned v = 0;
        for (std::size_t d = 0; d < 2U; ++d) {
            const char c = s[2U * i + d];
            if (c >= '0' && c <= '9') v = v * 16U + static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v = v * 16U + static_cast<unsigned>(c - 'a' + 10);
            else return "bad header hex";
        }
        bytes[i] = static_cast<std::uint8_t>(v);
    }
    return sp::Header::read(bytes, storage_bits != 0U ? storage_bits : unsigned{bytes[1]}, out);
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
    const auto status = is_strixpack_storage(storage) ? strixpack_matrix_geometry(storage, shape, geometry)
                                                      : matrix_load_geometry(storage, shape, geometry);
    if (!status) return status;
    bytes = geometry.bytes;
    return kOk;
}

Status pack_identity(const Source& file, std::size_t header_bytes, std::span<const LowbitPackEntry> entries,
                     Digest256& result) {
    core::Sha256 hash;
    std::vector<Byte> chunk(std::size_t{1} << 20);
    const auto add = [&](ByteRange range) -> Status {
        for (std::uint64_t done = 0; done < range.bytes;) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(range.bytes - done, chunk.size()));
            if (const auto status = file.read({range.offset + done, n}, std::span(chunk).first(n)); !status)
                return status;
            hash.update(chunk.data(), n);
            done += n;
        }
        return kOk;
    };
    if (const auto status = add({0, header_bytes}); !status) return status;
    for (const auto& entry : entries)
        if (const auto status = add({entry.offset, entry.bytes}); !status) return status;
    result = Digest256{hash.finish()};
    return kOk;
}

std::string pack_header(std::size_t count, const Digest256& checkpoint) {
    return std::string(kLowbitPackTag) + " " + std::to_string(count) + "\ncheckpoint " + hex(checkpoint) + "\n";
}

std::string entry_line(const LowbitPackEntry& e, std::uint64_t offset) {
    return "tensor " + e.tensor + " " + std::string(e.storage.name) + " " +
           std::to_string(e.storage.revision) + " " + std::to_string(e.rows) + " " +
           std::to_string(e.columns) + " " + std::to_string(offset) + " " + std::to_string(e.bytes) +
           " " + hex(e.source) + (e.sp_header ? " " + header_hex(*e.sp_header) : std::string{}) + "\n";
}

Status validate_entry(const LowbitPackEntry& e, std::set<std::string>& names) {
    if (e.tensor.empty() || std::ranges::any_of(e.tensor, [](unsigned char c) { return c <= ' ' || c == 127; }))
        return fail(Error::malformed, "invalid tensor name in low-bit pack: " + e.tensor);
    if (!supported(e.storage)) return fail(Error::unknown_format, "unsupported pack format for " + e.tensor);
    if (const auto why = lowbit_header_defect(e.storage, e.sp_header); !why.empty())
        return fail(Error::malformed, e.tensor + ": " + why);
    std::uint64_t want = 0;
    if (const auto status = expected_bytes(e.storage, e.rows, e.columns, want); !status) return status;
    if (want != e.bytes) return fail(Error::malformed, "payload size mismatch for " + e.tensor);
    if (!names.insert(e.tensor).second) return fail(Error::malformed, "duplicate tensor " + e.tensor);
    return kOk;
}

unsigned storage_bits(FormatId storage) {
    return storage == kSP4TWeightStorage ? 4U : strixpack_bits(storage);
}

}  // namespace

bool lowbit_storage_has_header(FormatId storage) noexcept { return storage_bits(storage) != 0U; }

std::string lowbit_header_defect(FormatId storage, const std::optional<sp::Header>& header) {
    if (!lowbit_storage_has_header(storage)) return header ? "a q4_g64 / q6_g64 entry carries no header" : "";
    if (!header) return "a Strix pack entry needs its tensor header";
    if (const char* why = header->invalid(); *why != '\0') return why;
    if (header->bits != storage_bits(storage)) return "Strix pack header width disagrees with storage";
    if (storage == kSP4TWeightStorage && header->table != sp::default_table<4>())
        return "strix.sp4t needs 4 bits and the fixed table {1,3,5,8,10,14,18,24}";
    return {};
}

struct LowbitPackWriter::Staging {
    std::filesystem::path destination, directory, file;
    std::ofstream stream;
    std::set<std::string> remaining, written;
    std::uint64_t reserve{}, pos{};
    bool owns_directory = false;

    ~Staging() {
        if (stream.is_open()) stream.close();
        if (!owns_directory) return;
        std::error_code error;
        std::filesystem::remove(file, error);
        std::filesystem::remove(directory, error);
    }
};

LowbitPackWriter::LowbitPackWriter(const Digest256& checkpoint) : checkpoint_(checkpoint) {}

LowbitPackWriter::LowbitPackWriter(const Digest256& checkpoint, const std::string& path,
                                   std::span<const std::string_view> names)
    : checkpoint_(checkpoint), staging_(std::make_unique<Staging>()) {
    auto& stage = *staging_;
    if (checkpoint.bytes == Digest256{}.bytes || names.empty())
        throw std::invalid_argument("streaming pack needs a checkpoint identity and tensor names");
    std::uint64_t header = pack_header(names.size(), checkpoint).size() + 4U;
    for (const auto name : names) {
        // Space for all decimal fields at their largest width and a Strix header.
        if (name.size() > kMaxHeader || header + name.size() + 256U > kMaxHeader)
            throw std::invalid_argument("streaming pack header exceeds 64 MiB");
        header += name.size() + 256U;
        if (!stage.remaining.emplace(name).second)
            throw std::invalid_argument("duplicate pack tensor: " + std::string(name));
    }
    stage.destination = path;
    if (std::filesystem::exists(std::filesystem::symlink_status(stage.destination)))
        throw std::invalid_argument("pack output already exists: " + path);
    stage.directory = path + ".writing";
    stage.file = stage.directory / "payload";
    stage.owns_directory = std::filesystem::create_directory(stage.directory);
    if (!stage.owns_directory) throw std::runtime_error("pack staging directory already exists: " + path);
    stage.stream.open(stage.file, std::ios::binary);
    stage.reserve = align_up(header);
    stage.pos = stage.reserve;
    stage.stream.seekp(static_cast<std::streamoff>(stage.pos));
    if (!stage.stream) throw std::runtime_error("cannot open pack staging file: " + path);
}

LowbitPackWriter::~LowbitPackWriter() = default;

void LowbitPackWriter::add(std::string tensor, FormatId storage, std::uint64_t rows, std::uint64_t columns,
                           const Digest256& source, std::vector<std::uint8_t> bytes,
                           std::optional<sp::Header> sp_header) {
    Item item;
    item.entry.tensor = std::move(tensor);
    item.entry.storage = storage;
    item.entry.rows = rows;
    item.entry.columns = columns;
    item.entry.bytes = bytes.size();
    item.entry.source = source;
    item.entry.sp_header = sp_header;
    if (staging_) {
        auto& stage = *staging_;
        if (!stage.remaining.contains(item.entry.tensor))
            throw std::invalid_argument("pack tensor was not planned or is duplicated: " + item.entry.tensor);
        if (const auto status = validate_entry(item.entry, stage.written); !status)
            throw std::invalid_argument(status.detail);
        item.entry.offset = align_up(stage.pos);
        const std::string padding(static_cast<std::size_t>(item.entry.offset - stage.pos), '\0');
        stage.stream.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        stage.stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stage.stream) throw std::runtime_error("cannot write pack payload for " + item.entry.tensor);
        stage.pos = item.entry.offset + item.entry.bytes;
        stage.remaining.erase(item.entry.tensor);
    } else {
        item.payload = std::move(bytes);
    }
    items_.push_back(std::move(item));
}

Status LowbitPackWriter::write(const std::string& path) const {
    if (staging_) return fail(Error::malformed, "use finish for a streaming pack");
    if (checkpoint_.bytes == Digest256{}.bytes) return fail(Error::malformed, "low-bit pack without a checkpoint identity");
    std::set<std::string> names;
    for (const auto& item : items_)
        if (const auto status = validate_entry(item.entry, names); !status) return status;
    // Header size depends on offsets only through digit counts; iterate to a fixed point.
    std::vector<std::uint64_t> offsets(items_.size(), 0);
    std::string header;
    const auto build = [&] {
        header = pack_header(items_.size(), checkpoint_);
        for (std::size_t i = 0; i < items_.size(); ++i)
            header += entry_line(items_[i].entry, offsets[i]);
        header += "end\n";
    };
    for (;;) {
        build();
        if (header.size() > kMaxHeader) return fail(Error::malformed, "low-bit pack header exceeds 64 MiB");
        auto at = align_up(header.size());
        std::vector<std::uint64_t> next(items_.size());
        for (std::size_t i = 0; i < items_.size(); ++i) {
            next[i] = at;
            if (items_[i].entry.bytes > std::numeric_limits<std::uint64_t>::max() - at - (kAlign - 1U))
                return fail(Error::memory_budget, "low-bit pack offset overflow");
            at = align_up(at + items_[i].entry.bytes);
        }
        if (next == offsets) break;
        offsets = std::move(next);
    }
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

Status LowbitPackWriter::finish() {
    if (!staging_ || !staging_->stream) return fail(Error::malformed, "pack staging is not writable");
    auto& stage = *staging_;
    if (!stage.remaining.empty()) return fail(Error::malformed, "pack missing planned tensor " + *stage.remaining.begin());
    std::string header = pack_header(items_.size(), checkpoint_);
    for (const auto& item : items_) header += entry_line(item.entry, item.entry.offset);
    header += "end\n";
    if (header.size() > stage.reserve) return fail(Error::malformed, "pack header exceeded its reservation");
    header.resize(static_cast<std::size_t>(stage.reserve), '\n');
    stage.stream.seekp(0);
    stage.stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stage.stream.close();
    if (!stage.stream) return fail(Error::malformed, "could not finish staged pack");
    std::error_code error;
    std::filesystem::create_hard_link(stage.file, stage.destination, error);
    if (error) return fail(Error::malformed, "could not publish staged pack: " + error.message());
    staging_.reset();
    items_.clear();
    return kOk;
}

Status LowbitPack::open(const std::string& path, const Digest256& checkpoint, LowbitPack& out) {
    return open_impl(path, checkpoint, false, out);
}

Status LowbitPack::open_host(const std::string& path, const Digest256& checkpoint, LowbitPack& out) {
    return open_impl(path, checkpoint, true, out);
}

Status LowbitPack::open_impl(const std::string& path, const Digest256& checkpoint, bool host, LowbitPack& out) {
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
        const auto search_from = text.size() > 4U ? text.size() - 4U : 0U;
        text.append(reinterpret_cast<const char*>(block.data()), block.size());
        const auto at = text.find("\nend\n", search_from);
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
    if (lines.size() < 3U || count != lines.size() - 3U || lines.back() != "end")
        return fail(Error::malformed, "low-bit pack entry count mismatch in " + path);
    const auto bound = split(lines[1]);
    Digest256 declared{};
    if (bound.size() != 2U || bound[0] != "checkpoint" || !parse_hex(bound[1], declared))
        return fail(Error::malformed, "low-bit pack " + path + " lacks its checkpoint line");
    if (declared.bytes != checkpoint.bytes)
        return fail(Error::stale_source, "low-bit pack " + path + " was coded from checkpoint " + hex(declared).substr(0, 16) +
                                             ", the model is " + hex(checkpoint).substr(0, 16));

    std::vector<LowbitPackEntry> entries;
    std::set<std::string> names;
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto f = split(lines[static_cast<std::size_t>(i) + 2U]);
        std::uint32_t version = 0;
        LowbitPackEntry e{};
        if ((f.size() != 9U && f.size() != 10U) || f[0] != "tensor" || f[1].empty() || !parse_num(f[3], version) ||
            !parse_num(f[4], e.rows) || !parse_num(f[5], e.columns) || !parse_num(f[6], e.offset) ||
            !parse_num(f[7], e.bytes) || !parse_hex(f[8], e.source))
            return fail(Error::malformed, "bad low-bit pack entry line " + std::to_string(i + 3U));
        e.tensor = std::string(f[1]);
        if (!host && is_strixpack_storage({f[2], version}))
            return fail(Error::unsupported_arithmetic, "Strix pack storage requires the Strix pack loader for " + e.tensor);
        const auto storage = parse_format(f[2], version);
        if (!storage) return fail(Error::unknown_format, "unsupported low-bit pack format for " + e.tensor);
        e.storage = *storage;
        if (f.size() == 10U) {
            sp::Header tensor_header;
            if (const auto why = parse_header(f[9], storage_bits(e.storage), tensor_header); !why.empty())
                return fail(Error::malformed, e.tensor + ": Strix pack header: " + why);
            e.sp_header = tensor_header;
        }
        if (const auto status = validate_entry(e, names); !status) return status;
        if (e.offset % kAlign != 0U || e.offset < padded || e.offset > file_size || e.bytes > file_size - e.offset)
            return fail(Error::malformed, "payload outside file for " + e.tensor);
        entries.push_back(std::move(e));
    }
    std::vector<ByteRange> ranges;
    ranges.reserve(entries.size());
    for (const auto& e : entries) ranges.push_back({e.offset, e.bytes});
    std::ranges::sort(ranges, {}, &ByteRange::offset);
    for (std::size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i - 1U].offset + ranges[i - 1U].bytes > ranges[i].offset)
            return fail(Error::malformed, "overlapping low-bit pack payloads");

    Digest256 identity;
    if (const auto status = pack_identity(*file, header.size(), entries, identity); !status) return status;
    out.path_ = path;
    out.file_ = std::move(file);
    out.entry_index_.clear();
    for (std::size_t i = 0; i < entries.size(); ++i) out.entry_index_.emplace(entries[i].tensor, i);
    out.entries_ = std::move(entries);
    out.header_bytes_ = header.size();
    out.identity_ = identity;
    out.checkpoint_ = declared;
    return kOk;
}

Status LowbitPack::verify_identity() const {
    if (!file_) return fail(Error::stale_source, "calibrated pack was not opened");
    Digest256 current;
    if (const auto status = pack_identity(*file_, header_bytes_, entries_, current); !status)
        return fail(status.code, "calibrated pack changed during upload: " + status.detail);
    if (!(current == identity_)) return fail(Error::stale_source, "calibrated pack digest changed during upload");
    return kOk;
}

Status LowbitPack::find(std::string_view tensor, FormatId storage, const RawTensor& source,
                        std::optional<Slice>& out, std::int32_t* outer_exponent) const {
    const auto rows = source.shape.rank == 2U ? source.shape.dims[0] : 0U;
    const auto columns = source.shape.rank == 2U ? source.shape.dims[1] : 0U;
    const PackMatrixSource matrix{PackMatrixSource::Encoding::bf16, static_cast<std::size_t>(rows),
                                  static_cast<std::size_t>(columns), source.payload, {}};
    return find_impl(tensor, storage, matrix, out, outer_exponent);
}

Status LowbitPack::find(std::string_view tensor, FormatId storage, const PackMatrixSource& source,
                        std::optional<Slice>& out) const {
    return find_impl(tensor, storage, source, out, nullptr);
}

Status LowbitPack::find_impl(std::string_view tensor, FormatId storage, const PackMatrixSource& source,
                             std::optional<Slice>& out, std::int32_t* outer_exponent) const {
    out.reset();
    if (outer_exponent) *outer_exponent = 0;
    const auto it = entry_index_.find(tensor);
    if (it == entry_index_.end()) return kOk;
    const auto& entry = entries_[it->second];
    if (entry.storage != storage) return fail(Error::malformed, "pack storage mismatch for " + entry.tensor);
    if (entry.rows != source.rows || entry.columns != source.columns)
        return fail(Error::malformed, "pack shape mismatch for " + entry.tensor);
    Digest256 digest{};
    if (const auto status = source.digest(digest); !status) return status;
    if (digest.bytes != entry.source.bytes) return fail(Error::stale_source, "pack source digest mismatch for " + entry.tensor);
    out = Slice{file_, {entry.offset, entry.bytes}};
    if (outer_exponent && entry.sp_header) *outer_exponent = entry.sp_header->base;
    return kOk;
}

}  // namespace strix::weight

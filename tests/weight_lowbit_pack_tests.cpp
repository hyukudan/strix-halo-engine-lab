// Low-bit pack sidecar: write/open round trip, find() validation against the checkpoint tensor, strict open(),
// and ResidentSlicePlan consuming a prepacked slice instead of converting BF16.
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/pack_plan.hpp"
#include "strix/weight/resident_slices.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace lb = strix::weight::lowbit;

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
}

struct Matrix {
    std::uint64_t rows{}, columns{};
    std::vector<std::uint8_t> bf16;  // keeps the source memory alive
    std::shared_ptr<const wei::Source> source;
    wei::RawTensor raw;
    std::vector<std::uint8_t> packed;
};

template <class Storage>
Matrix make_matrix(std::uint64_t rows, std::uint64_t columns, std::uint32_t seed) {
    Matrix m;
    m.rows = rows;
    m.columns = columns;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> values(static_cast<std::size_t>(rows * columns));
    for (auto& v : values) v = dist(rng);
    m.bf16.resize(values.size() * 2U);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof bits);
        const auto top = static_cast<std::uint16_t>(bits >> 16U);
        m.bf16[2U * i] = static_cast<std::uint8_t>(top & 0xffU);
        m.bf16[2U * i + 1U] = static_cast<std::uint8_t>(top >> 8U);
    }
    m.source = wei::memory_source(std::as_bytes(std::span(m.bf16)));
    m.raw.wire_dtype = "BF16";
    m.raw.shape.rank = 2;
    m.raw.shape.dims = {rows, columns, 0, 0};
    m.raw.payload = {m.source, {0, m.bf16.size()}};
    m.packed.resize(static_cast<std::size_t>(rows * columns / Storage::elements * Storage::bytes));
    std::size_t at = 0;
    for (std::size_t i = 0; i < values.size(); i += Storage::elements, at += Storage::bytes)
        if (lb::encode_block<Storage>(values.data() + i, m.packed.data() + at).code != lb::EncodeStatus::ok)
            ++failures;
    return m;
}

std::vector<std::uint8_t> read_all(const wei::Slice& slice) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(slice.range.bytes));
    const auto status = wei::read_slice(slice, std::as_writable_bytes(std::span(out)));
    if (!status) {
        ++failures;
        std::cerr << status.detail << '\n';
    }
    return out;
}

wei::Digest256 fingerprint(const wei::RawTensor& raw) {
    wei::Digest256 d{};
    const auto status = wei::digest_slice(raw.payload, d);
    check(static_cast<bool>(status), "fingerprint");
    return d;
}

// The checkpoint identity of the test: SHA-256 of the two tensors' bytes in order (stands for the raw file digest).
wei::Digest256 checkpoint_of(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::vector<std::uint8_t> all(a);
    all.insert(all.end(), b.begin(), b.end());
    return wei::digest_bytes(std::as_bytes(std::span(all)));
}

bool throws_invalid(const auto& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void run(const std::string& dir) {
    auto q4 = make_matrix<wei::Q4G64Storage>(3, 256, 1);
    auto q6 = make_matrix<wei::Q6G64Storage>(2, 128, 2);
    const auto path = dir + "/pack.bin";

    const auto checkpoint = checkpoint_of(q4.bf16, q6.bf16);
    wei::LowbitPackWriter writer(checkpoint);
    writer.add("blk.0.q4", wei::kQ4G64WeightStorage, 3, 256, fingerprint(q4.raw), q4.packed);
    writer.add("blk.0.q6", wei::kQ6G64WeightStorage, 2, 128, fingerprint(q6.raw), q6.packed);
    check(static_cast<bool>(writer.write(path)), "write");

    wei::LowbitPack pack;
    const auto opened = wei::LowbitPack::open(path, checkpoint, pack);
    check(static_cast<bool>(opened), "open");
    if (!opened) return;
    const auto& entries = pack.entries();
    check(entries.size() == 2U, "entry count");
    if (entries.size() == 2U) {
        check(entries[0].tensor == "blk.0.q4" && entries[0].storage == wei::kQ4G64WeightStorage &&
                  entries[0].rows == 3U && entries[0].columns == 256U && entries[0].bytes == q4.packed.size() &&
                  entries[0].offset % 4096U == 0U && entries[0].offset >= 4096U,
              "entry 0 round trip");
        check(entries[1].tensor == "blk.0.q6" && entries[1].storage == wei::kQ6G64WeightStorage &&
                  entries[1].rows == 2U && entries[1].columns == 128U && entries[1].bytes == q6.packed.size() &&
                  entries[1].offset % 4096U == 0U && entries[1].offset > entries[0].offset,
              "entry 1 round trip");
        check(entries[0].source.bytes == fingerprint(q4.raw).bytes, "entry 0 fingerprint");
    }
    check(pack.identity().bytes != wei::Digest256{}.bytes && pack.checkpoint().bytes == checkpoint.bytes, "identity set");
    {
        // Bound to the exact checkpoint bytes: a change in a tensor the pack does not hold (statistics depend on it)
        // and a missing checkpoint identity are refused as stale; a writer without a checkpoint refuses to write.
        auto other_tensor = q6.bf16;
        other_tensor[other_tensor.size() / 2U] ^= 1U;
        wei::LowbitPack stale;
        const auto refused = wei::LowbitPack::open(path, checkpoint_of(q4.bf16, other_tensor), stale);
        check(!refused && refused.code == wei::Error::stale_source && refused.detail.find("checkpoint") != std::string::npos,
              "pack refused for another checkpoint (non-packed tensor changed)");
        check(wei::LowbitPack::open(path, wei::Digest256{}, stale).code == wei::Error::stale_source,
              "pack refused without a checkpoint identity");
        wei::LowbitPackWriter unbound(wei::Digest256{});
        check(!unbound.write(path + ".unbound"), "writer refuses a pack without a checkpoint");
    }
    {
        // Same tensors, shapes and sources, one other code byte: the identity names the content.
        auto recoded = q4.packed;
        recoded.back() = static_cast<std::uint8_t>(recoded.back() ^ 1U);
        const auto other_path = path + ".recoded";
        wei::LowbitPackWriter other(checkpoint);
        other.add("blk.0.q4", wei::kQ4G64WeightStorage, 3, 256, fingerprint(q4.raw), recoded);
        other.add("blk.0.q6", wei::kQ6G64WeightStorage, 2, 128, fingerprint(q6.raw), q6.packed);
        wei::LowbitPack reopened;
        check(static_cast<bool>(other.write(other_path)) && static_cast<bool>(wei::LowbitPack::open(other_path, checkpoint, reopened)) &&
                  reopened.identity().bytes != pack.identity().bytes,
              "identity covers the payload bytes");
        std::remove(other_path.c_str());
    }

    std::optional<wei::Slice> found;
    check(static_cast<bool>(pack.find("blk.0.q4", wei::kQ4G64WeightStorage, q4.raw, found)) && found &&
              read_all(*found) == q4.packed,
          "find q4 bytes");
    check(static_cast<bool>(pack.find("blk.0.q6", wei::kQ6G64WeightStorage, q6.raw, found)) && found &&
              read_all(*found) == q6.packed,
          "find q6 bytes");

    check(static_cast<bool>(pack.find("absent", wei::kQ4G64WeightStorage, q4.raw, found)) && !found, "unknown tensor");

    check(!pack.find("blk.0.q4", wei::kQ6G64WeightStorage, q4.raw, found) && !found, "wrong storage");
    auto reshaped = q4.raw;
    reshaped.shape.dims = {6, 128, 0, 0};
    check(!pack.find("blk.0.q4", wei::kQ4G64WeightStorage, reshaped, found) && !found, "wrong shape");
    auto changed = q4;
    changed.bf16.back() ^= 1U;
    changed.source = wei::memory_source(std::as_bytes(std::span(changed.bf16)));
    changed.raw.payload = {changed.source, {0, changed.bf16.size()}};
    const auto stale = pack.find("blk.0.q4", wei::kQ4G64WeightStorage, changed.raw, found);
    check(!stale && stale.detail.find("blk.0.q4") != std::string::npos, "changed source fingerprint");
    {
        // Same size, same name and shape, same first and last 64 KiB: one byte in the middle of a tensor longer than
        // 128 KiB differs. The complete payload digest refuses it.
        auto big = make_matrix<wei::Q4G64Storage>(256, 512, 7);  // 256 KiB of BF16
        wei::LowbitPackWriter big_writer(checkpoint);
        big_writer.add("blk.1.q4", wei::kQ4G64WeightStorage, 256, 512, fingerprint(big.raw), big.packed);
        const auto big_path = path + ".big";
        wei::LowbitPack big_pack;
        check(static_cast<bool>(big_writer.write(big_path)) &&
                  static_cast<bool>(wei::LowbitPack::open(big_path, checkpoint, big_pack)),
              "big pack round trip");
        auto interior = big;
        interior.bf16[interior.bf16.size() / 2U] ^= 0x40U;
        interior.source = wei::memory_source(std::as_bytes(std::span(interior.bf16)));
        interior.raw.payload = {interior.source, {0, interior.bf16.size()}};
        const auto mid = big_pack.find("blk.1.q4", wei::kQ4G64WeightStorage, interior.raw, found);
        check(!mid && mid.code == wei::Error::stale_source && !found, "interior same-size mutation refused");
        check(static_cast<bool>(big_pack.find("blk.1.q4", wei::kQ4G64WeightStorage, big.raw, found)) && found,
              "unchanged big tensor found");
        std::remove(big_path.c_str());
    }

    // Plan with a prepacked slice.
    std::optional<wei::Slice> q4_slice;
    check(static_cast<bool>(pack.find("blk.0.q4", wei::kQ4G64WeightStorage, q4.raw, q4_slice)) && q4_slice, "slice q4");
    if (q4_slice) {
        const auto plan = wei::ResidentSlicePlan::make({{"w", &q4.raw, wei::kQ4G64WeightStorage, 0, q4_slice}}, 16);
        check(plan.bytes() == q4.packed.size() && plan.ranges().size() == 1U &&
                  plan.ranges()[0].source.range.offset == q4_slice->range.offset &&
                  plan.ranges()[0].source.range.bytes == q4.packed.size(),
              "plan uses prepacked range");
        std::vector<std::uint64_t> arena((plan.bytes() + 7U) / 8U);
        const auto bytes = std::span(reinterpret_cast<std::uint8_t*>(arena.data()), arena.size() * 8U);
        for (const auto& range : plan.ranges())
            check(static_cast<bool>(wei::read_slice(
                      range.source, std::as_writable_bytes(bytes.subspan(range.destination_offset, range.bytes)))),
                  "read plan range");
        const auto slices = wei::ResidentSlices::bind(plan, bytes);
        const auto view = slices.matrix("w");
        check(view.rows == 3U && view.columns == 256U && view.storage == wei::kQ4G64WeightStorage &&
                  std::equal(view.bytes.begin(), view.bytes.end(), q4.packed.begin(), q4.packed.end()),
              "bound matrix holds prepacked bytes");

        wei::Slice wrong = *q4_slice;
        wrong.range.bytes -= 1U;
        check(throws_invalid([&] {
                  (void)wei::ResidentSlicePlan::make({{"w", &q4.raw, wei::kQ4G64WeightStorage, 0, wrong}}, 16);
              }),
              "wrong-size prepacked throws");
        check(throws_invalid([&] { (void)wei::ResidentSlicePlan::make({{"w", &q4.raw, std::nullopt, 0, q4_slice}}, 16); }),
              "prepacked without storage throws");
        check(throws_invalid([&] {
                  (void)wei::ResidentSlicePlan::make({{"w", &q4.raw, wei::kBf16WeightStorage, 0, q4_slice}}, 16);
              }),
              "prepacked with bf16 storage throws");
    }

    // Strict open.
    std::vector<char> file;
    {
        std::ifstream in(path, std::ios::binary);
        file.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const auto write_file = [&](const std::string& name, const std::vector<char>& data) {
        const auto p = dir + "/" + name;
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        return p;
    };
    wei::LowbitPack scratch;
    auto truncated = file;
    truncated.resize(truncated.size() - 1U);
    check(!wei::LowbitPack::open(write_file("trunc.bin", truncated), checkpoint, scratch), "truncated file fails");
    auto bad_tag = file;
    bad_tag[14] = 'X';
    check(!wei::LowbitPack::open(write_file("tag.bin", bad_tag), checkpoint, scratch), "bad tag fails");
    auto v1 = file;
    v1[19] = '1';  // "strix-lowbit-pack/v1": the sampled-source format is refused
    check(!wei::LowbitPack::open(write_file("v1.bin", v1), checkpoint, scratch), "v1 pack refused");
    check(!wei::LowbitPack::open(dir + "/missing.bin", checkpoint, scratch), "missing file fails");
    check(!wei::LowbitPack::open(write_file("empty.bin", {}), checkpoint, scratch), "empty file fails");
}

void slice_plan_test(const std::string& dir) {
    const auto path = std::filesystem::path(dir) / "slice.tsv";
    const std::vector<wei::PackSliceMatrix> matrices{{"a.w1", 4, 256}, {"a.w2", 8, 128}};
    wei::write_pack_slice_plan(path, "sp2", matrices);
    std::ifstream in(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const auto fmt = wei::strixpack_format("sp2");
    const auto expected = "tensor\tformat\tbytes\na.w1\tsp2\t" + std::to_string(wei::strixpack_matrix_bytes(fmt, 4, 256)) +
                          "\na.w2\tsp2\t" + std::to_string(wei::strixpack_matrix_bytes(fmt, 8, 128)) + "\n";
    check(text == expected, "slice plan rows");
    bool refused = false;
    try { wei::write_pack_slice_plan(path, "sp2", matrices); } catch (const std::runtime_error&) { refused = true; }
    check(refused, "slice plan never overwrites");
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "strix-lowbit-pack-test";
    std::filesystem::create_directories(dir);
    try {
        run(dir.string());
        slice_plan_test(dir.string());
    } catch (const std::exception& e) {
        ++failures;
        std::cerr << "exception: " << e.what() << '\n';
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (failures != 0) return 1;
    std::cout << "weight lowbit pack tests passed\n";
    return 0;
}

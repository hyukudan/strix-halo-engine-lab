// Explicit two-rank deciding fixture: real packed output weights, production
// Q8/BF16 head operator and stable project USB4 host transport. No model executor,
// reduction kernel, cache, sampler, drafter or ordinary route is modified.
#include "options.hpp"
#include "output_split.hpp"
#include "scalar_q8.hpp"
#include "strix/dist/communicator_hip.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/model_content_hash.hpp"
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime.h>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
extern char **environ;
#else
#error The native USB4 fixture is Linux-only
#endif
#ifndef STRIX_SPLIT_SOURCE_SHA
#error A source packet revision must be supplied by the explicit build hook
#endif
namespace qs = strix::qwen_split;
namespace dist = strix::dist;
namespace {
constexpr const char *recipe = "qwen-output-row-q8-bf16-prefetch8-wave8-v1";
void hip_check(hipError_t e) {
    if (e != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(e));
    }
}
qs::Digest hash(std::span<const std::byte> bytes) { return strix::sha256_bytes_fast(bytes).bytes; }
qs::Digest hash_text(const std::string &text) {
    return hash(std::as_bytes(std::span(text.data(), text.size())));
}
std::string hex(qs::Digest digest) {
    return strix::model_content_identity_hex(strix::ModelContentIdentity{digest});
}
using qs::Options;
class Deadline {
    std::atomic<bool> done_{false};
    std::thread thread_;

  public:
    explicit Deadline(std::uint64_t ms)
        : thread_([this, ms] {
              const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
              while (!done_.load()) {
                  if (std::chrono::steady_clock::now() >= until) {
                      std::fputs("fixture absolute deadline; no result publication\n", stderr);
                      std::fflush(stderr);
                      std::_Exit(124);
                  }
                  std::this_thread::sleep_for(std::chrono::milliseconds(20));
              }
          }) {}
    ~Deadline() {
        done_.store(true);
        thread_.join();
    }
};
class Stream {
  public:
    hipStream_t value = nullptr;
    ~Stream() {
        if (value) {
            (void)hipStreamDestroy(value);
        }
    }
};
class Device {
  public:
    void *pointer = nullptr;
    std::uint64_t bytes = 0;
    Device() = default;
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device() {
        if (pointer) {
            (void)hipFree(pointer);
        }
    }
    void allocate(std::uint64_t n) {
        qs::require(!pointer && n, "device owner already live/empty");
        hip_check(hipMalloc(&pointer, qs::extent(n)));
        bytes = n;
    }
    void release() {
        if (pointer) {
            hip_check(hipFree(pointer));
            pointer = nullptr;
            bytes = 0;
        }
    }
};
struct Devices {
    Device weights, hidden, dense, local, peer;
};
std::vector<std::byte> read_region(const std::filesystem::path &path, std::uint64_t offset,
                                   std::uint64_t count) {
    qs::require(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) &&
                    count <=
                        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
                "asset region exceeds stream limits");
    qs::require(qs::add(offset, count) <= std::filesystem::file_size(path),
                "asset region truncated");
    std::ifstream input(path, std::ios::binary);
    qs::require(bool(input), "cannot open bound asset");
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> result(qs::extent(count));
    input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count));
    qs::require(input.gcount() == static_cast<std::streamsize>(count), "asset region short read");
    return result;
}
qs::Digest hash_region(const std::filesystem::path &path, std::uint64_t offset,
                       std::uint64_t count) {
    qs::require(qs::add(offset, count) <= std::filesystem::file_size(path) &&
                    offset <=
                        static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
                "tensor hash region invalid");
    std::ifstream input(path, std::ios::binary);
    qs::require(bool(input), "cannot open packed tensor");
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> chunk(1024 * 1024);
    strix::Sha256Fast sha;
    while (count) {
        const auto n = std::min<std::uint64_t>(count, chunk.size());
        input.read(reinterpret_cast<char *>(chunk.data()), static_cast<std::streamsize>(n));
        qs::require(input.gcount() == static_cast<std::streamsize>(n), "tensor hash short read");
        sha.update(std::span(chunk).first(qs::extent(n)));
        count -= n;
    }
    return sha.finalize().bytes;
}
std::uint64_t host_available() {
    std::ifstream input("/proc/meminfo");
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string name, unit;
        std::uint64_t value = 0;
        if (row >> name >> value >> unit; name == "MemAvailable:") {
            qs::require(unit == "kB", "invalid memory observation unit");
            return qs::mul(value, 1024);
        }
    }
    throw std::runtime_error("MemAvailable observation absent");
}
std::string environment(const hipDeviceProp_t &props) {
    int runtime = 0, driver = 0;
    hip_check(hipRuntimeGetVersion(&runtime));
    hip_check(hipDriverGetVersion(&driver));
    std::map<std::string, std::string> values;
    for (char **e = environ; *e; ++e) {
        const std::string item = *e;
        const auto equal = item.find('=');
        if (equal == std::string::npos) {
            continue;
        }
        const auto name = item.substr(0, equal);
        if (name.starts_with("STRIX_") || name.starts_with("HIP_") || name.starts_with("HSA_") ||
            name.starts_with("ROCR_") || name.starts_with("ROCM_") || name.starts_with("AMD_") ||
            name == "LD_PRELOAD" || name == "LD_LIBRARY_PATH") {
            values[name] = item.substr(equal + 1);
        }
    }
    values["@architecture"] = props.gcnArchName;
    values["@hip-runtime"] = std::to_string(runtime);
    values["@hip-driver"] = std::to_string(driver);
    std::string result;
    for (const auto &[name, value] : values) {
        qs::append_framed(result, name, value);
    }
    return result;
}
void views(const Devices &d) {
    const std::array<const Device *, 4> all = {&d.weights, &d.hidden, &d.local, &d.peer};
    for (std::size_t i = 0; i < all.size(); ++i) {
        for (std::size_t j = i + 1; j < all.size(); ++j) {
            qs::disjoint(reinterpret_cast<std::uintptr_t>(all[i]->pointer), all[i]->bytes,
                         reinterpret_cast<std::uintptr_t>(all[j]->pointer), all[j]->bytes);
        }
    }
}
void output_file(const std::string &path, std::span<const float> rows) {
    // Called only AFTER complete bilateral commit, never on a failed/partial row.
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        throw std::runtime_error("result path exists/cannot be created");
    }
    bool ok = true;
    const auto bytes = std::as_bytes(rows);
    std::size_t at = 0;
    while (at < bytes.size()) {
        const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            ok = false;
            break;
        }
        at += static_cast<std::size_t>(n);
    }
    if (ok && ::fsync(fd) != 0) {
        ok = false;
    }
    if (::close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        (void)::unlink(path.c_str());
        throw std::runtime_error("result write failed; own incomplete output removed");
    }
}
int run(const Options &o) {
    qs::require(std::endian::native == std::endian::little,
                "native asset bytes require little endian");
    qs::require(o.number("--world") == 2 && o.get("--transport") == "usb4-auto",
                "fixture requires world2/usb4-auto");
    const auto rank64 = o.number("--rank");
    qs::require(rank64 < 2, "fixture rank must be 0 or 1");
    const auto rank = static_cast<std::uint32_t>(rank64);
    qs::Shape shape;
    shape.inputs = o.number("--inputs");
    shape.capacity = o.number("--capacity");
    shape.position = o.number("--position");
    const auto p = qs::plan(shape, rank);
    const auto deadline = o.number("--deadline-ms");
    qs::require(deadline && deadline <= 430000, "fixture deadline must be 1..430000 ms");
    Deadline watchdog(deadline);
    qs::require(o.get("--source-sha") == STRIX_SPLIT_SOURCE_SHA,
                "fixture source revision differs from the binary binding");
    qs::Binding b;
    b.source = hash_text(STRIX_SPLIT_SOURCE_SHA);
    b.target = hash_text(o.get("--target-identity") + ";" + recipe);
    b.asset = qs::hex_digest(o.get("--model-sha"));
    b.tensor = qs::hex_digest(o.get("--tensor-sha"));
    b.input = qs::hex_digest(o.get("--hidden-sha"));
    b.reference = qs::hex_digest(o.get("--reference-sha"));
    b.epoch = o.number("--epoch");
    b.reserve = o.number("--reserve-bytes");
    const auto leaf = o.get("--scalar-leaf");
    qs::require(leaf == "fma" || leaf == "separate",
                "declare scalar leaf fma or separate before execution");
    b.scalar_leaf = leaf == "fma" ? 1U : 2U;
    // Actual input/profile identity, not an arbitrary label, must be pinned by
    // the capture owner. Reject absent/noncanonical target identity digests.
    (void)qs::hex_digest(o.get("--target-identity"));
    Stream stream;
    Devices d;
    // All host DMA sources/destinations outlive the catch-side cancel and drain,
    // including uncertain HIP errors. Device owners likewise outlive the drain.
    std::vector<std::byte> input_bytes, reference_bytes, packed, shard;
    std::vector<std::uint16_t> hidden;
    std::vector<float> reference, dense, candidate, local, peer;
    std::vector<std::uint32_t> selections;
    std::unique_ptr<dist::Communicator> comm;
    std::unique_ptr<qs::Gate> gate;
    try {
        int device = 0;
        hip_check(hipGetDevice(&device));
        hipDeviceProp_t props{};
        hip_check(hipGetDeviceProperties(&props, device));
        const std::string arch = props.gcnArchName;
        qs::require(arch == "gfx1151" || arch.starts_with("gfx1151:"),
                    "fixture requires actual gfx1151");
        hip_check(hipStreamCreateWithFlags(&stream.value, hipStreamNonBlocking));
        b.environment = hash_text(environment(props));
        b.binary = strix::sha256_file_content(std::filesystem::read_symlink("/proc/self/exe"),
                                              strix::FileHashPolicy::full)
                       .bytes;
        dist::Usb4AutoOptions transport_options;
        transport_options.rank = rank;
        transport_options.timeout =
            std::chrono::milliseconds(std::min<std::uint64_t>(deadline, 120000));
        dist::Usb4Topology topology;
        auto transport = dist::usb4_host_auto(transport_options, &topology);
        std::fprintf(stderr, "fixture transport %s\n", topology.receipt().c_str());
        dist::Group group;
        group.rank = rank;
        group.world = 2;
        group.epoch = b.epoch;
        group.plan = 0x315450554f5751ULL;
        comm = std::make_unique<dist::Communicator>(
            group, std::move(transport), std::max(qs::extent(p.wire_bytes), qs::frame_bytes));
        const auto control = [&](const qs::Frame &frame) {
            const auto tx = qs::encode(frame);
            qs::Wire rx{};
            dist::Tag tag;
            tag.kind = static_cast<std::uint32_t>(dist::TagKind::control);
            tag.layer = static_cast<std::uint32_t>(frame.phase);
            tag.request = frame.sequence;
            comm->exchange_host(tx, rx, tag);
            gate->accept(rx);
        };
        std::uint64_t available = 0;
        bool ready = true;
        try {
            const auto info = strix::gguf::read_file(o.get("--model"));
            b.asset_bytes = info.file_size;
            const auto name = o.get("--tensor");
            const strix::gguf::TensorInfo *tensor = nullptr;
            for (const auto &item : info.tensors) {
                if (item.name == name) {
                    qs::require(!tensor, "duplicate head tensor name");
                    tensor = &item;
                }
            }
            qs::require(tensor && tensor->ggml_type == 8 &&
                            tensor->dimensions ==
                                std::vector<std::uint64_t>{shape.width, shape.outputs} &&
                            tensor->byte_size == p.full_weights,
                        "fixture needs the actual Q8_0 production head with exact dimensions");
            b.tensor_offset = tensor->absolute_offset;
            qs::require(
                strix::sha256_file_content(o.get("--model"), strix::FileHashPolicy::full).bytes ==
                    b.asset,
                "full model asset SHA mismatch");
            qs::require(hash_region(o.get("--model"), b.tensor_offset, p.full_weights) == b.tensor,
                        "actual packed head SHA mismatch");
            qs::require(
                std::filesystem::file_size(o.get("--hidden")) == p.input_bytes &&
                    strix::sha256_file_content(o.get("--hidden"), strix::FileHashPolicy::full)
                            .bytes == b.input,
                "captured BF16 hidden extent/SHA mismatch");
            qs::require(
                std::filesystem::file_size(o.get("--reference")) == p.all_logits_bytes &&
                    strix::sha256_file_content(o.get("--reference"), strix::FileHashPolicy::full)
                            .bytes == b.reference,
                "independent full raw-logit reference extent/SHA mismatch");
            std::size_t free = 0, total = 0;
            hip_check(hipMemGetInfo(&free, &total));
            available = std::min(
                {static_cast<std::uint64_t>(free), host_available(), o.number("--budget-bytes")});
            qs::require(available >= qs::add(p.declared_peak, b.reserve),
                        "local declared storage plus reserve does not fit");
            qs::validate_binding(b);
            qs::require(!std::filesystem::exists(o.get("--out")), "result path already exists");
        } catch (const std::exception &e) {
            ready = false;
            std::fprintf(stderr, "fixture preflight refused: %s\n", e.what());
        }
        if (!ready) {
            // Refusal only: no fabricated ready frontier, execution or retry.
            const auto tx = qs::encode(qs::Frame{p, b, qs::Phase::offer, 0, available, 0, 1, {}});
            qs::Wire rx{};
            comm->exchange_host(tx, rx, {static_cast<std::uint32_t>(dist::TagKind::control), 1, 0});
            throw std::runtime_error("local admission refused; both ranks must stop");
        }
        gate = std::make_unique<qs::Gate>(p, b);
        control(gate->offer(available));
        input_bytes = read_region(o.get("--hidden"), 0, p.input_bytes);
        qs::require(hash(input_bytes) == b.input, "hidden changed after admission");
        hidden.resize(qs::extent(p.input_bytes / 2));
        std::memcpy(hidden.data(), input_bytes.data(), input_bytes.size());
        std::vector<std::byte>().swap(input_bytes);
        for (auto value : hidden) {
            qs::require(std::isfinite(qs::bf16(value)), "nonfinite captured hidden input");
        }
        reference_bytes = read_region(o.get("--reference"), 0, p.all_logits_bytes);
        qs::require(hash(reference_bytes) == b.reference, "reference changed after admission");
        reference.resize(qs::extent(p.all_logits_bytes / 4));
        std::memcpy(reference.data(), reference_bytes.data(), reference_bytes.size());
        std::vector<std::byte>().swap(reference_bytes);
        packed = read_region(o.get("--model"), b.tensor_offset, p.full_weights);
        qs::require(hash(packed) == b.tensor, "packed head changed after admission");
        const auto shard_offset = qs::mul(p.first, p.row_bytes);
        const auto expected_shard =
            hash(std::span(packed).subspan(qs::extent(shard_offset), qs::extent(p.weights)));
        d.weights.allocate(p.full_weights);
        d.hidden.allocate(p.input_bytes);
        d.dense.allocate(p.full_row_bytes);
        hip_check(hipMemcpy(d.weights.pointer, packed.data(), qs::extent(p.full_weights),
                            hipMemcpyHostToDevice));
        hip_check(hipMemcpy(d.hidden.pointer, hidden.data(), qs::extent(p.input_bytes),
                            hipMemcpyHostToDevice));
        dense.resize(qs::extent(shape.outputs));
        const std::array<std::uint64_t, 7> probes = {
            0, 1, p.count - 1, p.count, p.count + 1, shape.outputs - 2, shape.outputs - 1};
        for (std::uint64_t input = 0; input < shape.inputs; ++input) {
            strix::launch_hip_q8_head_rows(static_cast<const std::uint16_t *>(d.hidden.pointer) +
                                               qs::extent(input * shape.width),
                                           d.weights.pointer,
                                           static_cast<std::uint32_t>(shape.width),
                                           static_cast<std::uint32_t>(shape.outputs),
                                           static_cast<float *>(d.dense.pointer), stream.value);
            hip_check(hipStreamSynchronize(stream.value));
            hip_check(hipMemcpy(dense.data(), d.dense.pointer, qs::extent(p.full_row_bytes),
                                hipMemcpyDeviceToHost));
            const auto expected = std::span(reference).subspan(qs::extent(input * shape.outputs),
                                                               qs::extent(shape.outputs));
            qs::require(std::memcmp(dense.data(), expected.data(), qs::extent(p.full_row_bytes)) ==
                            0,
                        "dense native head differs from independent target raw logits");
            selections.push_back(qs::firstmax(dense));
            qs::require(selections.back() == qs::firstmax(expected),
                        "dense/reference firstmax differs");
            for (auto row : probes) {
                const auto value =
                    qs::scalar_q8(std::span(hidden).subspan(qs::extent(input * shape.width),
                                                            qs::extent(shape.width)),
                                  std::span(packed).subspan(qs::extent(row * p.row_bytes),
                                                            qs::extent(p.row_bytes)),
                                  b.scalar_leaf);
                qs::require(std::bit_cast<std::uint32_t>(value) ==
                                std::bit_cast<std::uint32_t>(dense[qs::extent(row)]),
                            "declared scalar leaf/tree differs from dense native head");
            }
        }
        control(gate->reference(hash(std::as_bytes(std::span(reference)))));
        d.weights.release();
        d.dense.release();
        std::vector<std::byte>().swap(packed);
        std::vector<float>().swap(dense);
        shard = read_region(o.get("--model"), qs::add(b.tensor_offset, shard_offset), p.weights);
        qs::require(hash(shard) == expected_shard, "owned packed shard changed after oracle");
        d.weights.allocate(p.weights);
        d.local.allocate(p.wire_bytes);
        d.peer.allocate(p.wire_bytes);
        views(d);
        hip_check(hipMemcpy(d.weights.pointer, shard.data(), qs::extent(p.weights),
                            hipMemcpyHostToDevice));
        hip_check(hipStreamSynchronize(stream.value));
        std::vector<std::byte>().swap(shard);
        candidate.resize(qs::extent(p.all_logits_bytes / 4));
        local.resize(qs::extent(p.count));
        peer.resize(qs::extent(p.count));
        for (std::uint64_t input = 0; input < shape.inputs; ++input) {
            gate->begin_payload(input + 1);
            strix::launch_hip_q8_head_rows(static_cast<const std::uint16_t *>(d.hidden.pointer) +
                                               qs::extent(input * shape.width),
                                           d.weights.pointer,
                                           static_cast<std::uint32_t>(shape.width),
                                           static_cast<std::uint32_t>(p.count),
                                           static_cast<float *>(d.local.pointer), stream.value);
            dist::Tag tag;
            tag.kind = static_cast<std::uint32_t>(dist::TagKind::raw_slots);
            tag.layer = 0;
            tag.request = input + 1;
            comm->exchange(stream.value, d.local.pointer, d.peer.pointer, qs::extent(p.wire_bytes),
                           tag);
            hip_check(hipStreamSynchronize(stream.value));
            hip_check(hipMemcpy(local.data(), d.local.pointer, qs::extent(p.wire_bytes),
                                hipMemcpyDeviceToHost));
            hip_check(hipMemcpy(peer.data(), d.peer.pointer, qs::extent(p.wire_bytes),
                                hipMemcpyDeviceToHost));
            comm->check_exchanges();
            gate->finish_payload(input + 1);
            auto full = std::span(candidate).subspan(qs::extent(input * shape.outputs),
                                                     qs::extent(shape.outputs));
            qs::gather(p, local, peer, full);
            const auto expected = std::span(reference).subspan(qs::extent(input * shape.outputs),
                                                               qs::extent(shape.outputs));
            qs::require(std::memcmp(full.data(), expected.data(), qs::extent(p.full_row_bytes)) ==
                            0,
                        "split full raw-logit row differs from target reference");
            qs::require(qs::firstmax(full) == selections[qs::extent(input)],
                        "split firstmax differs from target reference");
            control(gate->row(hash(std::as_bytes(full))));
        }
        const auto result = hash(std::as_bytes(std::span(candidate)));
        qs::require(result == b.reference,
                    "complete split output digest differs from target reference");
        qs::require(hash(std::as_bytes(std::span(hidden))) == b.input,
                    "replicated immutable hidden state changed");
        control(gate->commit(result));
        qs::require(gate->publishable(), "split result remains provisional");
        comm->check_exchanges();
        output_file(o.get("--out"), candidate);
        std::printf("{\"kind\":\"qwen-output-row-fixture\",\"status\":\"EXACT_OPERATOR\",\"full_"
                    "model_split\":false,\"rank\":%u,\"world\":2,\"source\":\"%s\",\"binary_"
                    "sha256\":\"%s\",\"environment_sha256\":\"%s\",\"reference_sha256\":\"%s\","
                    "\"raw_logits_sha256\":\"%s\",\"inputs\":%llu,\"scalar_probe_rows\":%llu,"
                    "\"local_weight_bytes\":%llu,\"declared_peak_bytes\":%llu,\"canonical_model_"
                    "state\":\"NOT_RUN_OPERATOR_FIXTURE\",\"head_firstmax\":[",
                    rank, STRIX_SPLIT_SOURCE_SHA, hex(b.binary).c_str(), hex(b.environment).c_str(),
                    hex(b.reference).c_str(), hex(result).c_str(),
                    static_cast<unsigned long long>(shape.inputs),
                    static_cast<unsigned long long>(shape.inputs * probes.size()),
                    static_cast<unsigned long long>(p.weights),
                    static_cast<unsigned long long>(p.declared_peak));
        for (std::size_t i = 0; i < selections.size(); ++i) {
            std::printf("%s%u", i ? "," : "", selections[i]);
        }
        const auto traffic = qs::census(p);
        std::printf(
            "],\"model_sha256\":\"%s\",\"packed_tensor_sha256\":\"%s\",\"input_sha256\":\"%s\","
            "\"target_capture_identity\":\"%s\",\"target_recipe_identity_sha256\":\"%s\","
            "\"recipe\":\"%s\",\"scalar_leaf\":%llu,\"epoch\":%llu,\"context_capacity\":%llu,"
            "\"first_position\":%llu,\"first_output\":%llu,\"output_count\":%llu,\"asset_bytes\":%"
            "llu,\"tensor_offset\":%llu,\"admission_available_bytes\":%llu,\"reserve_bytes\":%llu,"
            "\"logical_calls\":%llu,\"payload_bytes_per_rank\":%llu,\"trailer_bytes_per_rank\":%"
            "llu,\"both_final_receipts\":true}\n",
            hex(b.asset).c_str(), hex(b.tensor).c_str(), hex(b.input).c_str(),
            o.get("--target-identity").c_str(), hex(b.target).c_str(), recipe,
            static_cast<unsigned long long>(b.scalar_leaf),
            static_cast<unsigned long long>(b.epoch),
            static_cast<unsigned long long>(shape.capacity),
            static_cast<unsigned long long>(shape.position),
            static_cast<unsigned long long>(p.first), static_cast<unsigned long long>(p.count),
            static_cast<unsigned long long>(b.asset_bytes),
            static_cast<unsigned long long>(b.tensor_offset),
            static_cast<unsigned long long>(available), static_cast<unsigned long long>(b.reserve),
            static_cast<unsigned long long>(traffic.total_calls),
            static_cast<unsigned long long>(traffic.payload_bytes),
            static_cast<unsigned long long>(traffic.trailer_bytes));
        return 0;
    } catch (...) {
        if (gate) {
            gate->fail();
        }
        if (comm) {
            comm->cancel_pending("output split fixture failure; no publication/retry");
        }
        if (stream.value) {
            const auto status = hipStreamSynchronize(stream.value);
            if (status != hipSuccess) {
                std::fprintf(stderr, "failure drain: %s\n", hipGetErrorString(status));
            }
        }
        throw;
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        return run(Options(argc, argv));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "qwen output split REFUSED/FAILED: %s\n", e.what());
        return 3;
    }
}

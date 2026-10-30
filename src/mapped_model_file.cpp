#include "strix/file_snapshot.hpp"
#include "strix/mapped_model_file.hpp"

#include "strix/model_content_hash.hpp"
#include "strix/ple_direct.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix {
namespace {

#if defined(__linux__)

std::array<std::uint64_t,7> file_state(const struct stat& value) {
    return {static_cast<std::uint64_t>(value.st_dev),static_cast<std::uint64_t>(value.st_ino),
        static_cast<std::uint64_t>(value.st_size),static_cast<std::uint64_t>(value.st_mtim.tv_sec),
        static_cast<std::uint64_t>(value.st_mtim.tv_nsec),static_cast<std::uint64_t>(value.st_ctim.tv_sec),
        static_cast<std::uint64_t>(value.st_ctim.tv_nsec)};
}
bool same_file_state(
    const struct stat& left,
    const struct stat& right) noexcept {
    return left.st_dev == right.st_dev &&
           left.st_ino == right.st_ino &&
           left.st_size == right.st_size &&
           left.st_mtim.tv_sec ==
               right.st_mtim.tv_sec &&
           left.st_mtim.tv_nsec ==
               right.st_mtim.tv_nsec &&
           left.st_ctim.tv_sec ==
               right.st_ctim.tv_sec &&
           left.st_ctim.tv_nsec ==
               right.st_ctim.tv_nsec;
}

#endif

}  // namespace

MappedModelFile::MappedModelFile(
    MappedModelFile&& other) noexcept
    : fd_(other.fd_),
      mapping_base_(other.mapping_base_),
      bytes_(other.bytes_),
      allocation_generation_(
          other.allocation_generation_),
      logical_name_(
          std::move(other.logical_name_)),
      content_identity_(
          other.content_identity_), file_sha256_(other.file_sha256_),file_state_(other.file_state_) {
    other.fd_ = -1;
    other.mapping_base_ = 0;
    other.bytes_ = 0;
    other.allocation_generation_ = 0;
    other.content_identity_ = {};
    other.file_sha256_ = {};
    other.file_state_ = {};
}

MappedModelFile&
MappedModelFile::operator=(
    MappedModelFile&& other) noexcept {
    if (this != &other) {
        release();
        fd_ = other.fd_;
        mapping_base_ =
            other.mapping_base_;
        bytes_ = other.bytes_;
        allocation_generation_ =
            other.allocation_generation_;
        logical_name_ =
            std::move(other.logical_name_);
        content_identity_ =
            other.content_identity_;
        file_sha256_=other.file_sha256_;
        file_state_=other.file_state_;

        other.fd_ = -1;
        other.mapping_base_ = 0;
        other.bytes_ = 0;
        other.allocation_generation_ = 0;
        other.content_identity_ = {};
        other.file_sha256_ = {};
        other.file_state_ = {};
    }
    return *this;
}

MappedModelFile::~MappedModelFile() {
    release();
}

MappedModelFile MappedModelFile::open_read_only(
    const std::filesystem::path& path,
    std::string logical_name,
    AllocationGenerationSource& generations) {
#if !defined(__linux__)
    static_cast<void>(path);
    static_cast<void>(logical_name);
    static_cast<void>(generations);
    throw std::runtime_error(
        "mapped model files require native Linux");
#else
    if (logical_name.empty()) {
        throw std::invalid_argument(
            "mapped model logical name must be non-empty");
    }

    const auto native =
        path.string();
    const int fd =
        ::open(
            native.c_str(),
            O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "cannot open model file read-only");
    }

    MappedModelFile mapped;
    mapped.fd_ = fd;
    mapped.logical_name_ =
        std::move(logical_name);

    try {
        strix::LoaderCapture::record(path, fd);
        struct stat before {};
        if (::fstat(fd, &before) != 0 ||
            before.st_size <= 0) {
            throw std::runtime_error(
                "cannot stat non-empty model file");
        }
        mapped.bytes_ =
            static_cast<std::uint64_t>(
                before.st_size);
        if (mapped.bytes_ >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "model file size does not fit host size_t");
        }

        const auto proc_path =
            std::filesystem::path(
                "/proc/self/fd") /
            std::to_string(fd);
        const ModelContentFile content_file{
            mapped.logical_name_,
            proc_path,
        };
        const auto identities=hash_single_model_content_file(content_file);
        if(identities.file_bytes!=mapped.bytes_)
            throw std::runtime_error("model file extent changed during its full-file hash");
        mapped.content_identity_=identities.content_identity;
        mapped.file_sha256_=identities.file_sha256;

        struct stat after_hash {};
        if (::fstat(fd, &after_hash) != 0 ||
            !same_file_state(
                before,
                after_hash)) {
            throw std::runtime_error(
                "model file changed while establishing content identity");
        }

        void* address =
            ::mmap(
                nullptr,
                static_cast<std::size_t>(
                    mapped.bytes_),
                PROT_READ,
                MAP_PRIVATE,
                fd,
                0);
        if (address == MAP_FAILED) {
            throw std::runtime_error(
                "cannot mmap model file read-only");
        }
        mapped.mapping_base_ =
            reinterpret_cast<std::uintptr_t>(
                address);
        mapped.allocation_generation_ =
            generations.next();

        struct stat after_map {};
        if (::fstat(fd, &after_map) != 0 ||
            !same_file_state(
                after_hash,
                after_map)) {
            throw std::runtime_error(
                "model file changed while mapping");
        }
        mapped.file_state_=file_state(after_map);
    } catch (...) {
        mapped.release();
        throw;
    }

    return mapped;
#endif
}

void MappedModelFile::validate_unchanged() const {
#if defined(__linux__)
    struct stat current {};
    if(!valid() || fd_<0 || !file_sha256_.valid() || !content_identity_.valid() ||
        ::fstat(fd_,&current)!=0 || file_state(current)!=file_state_)
        throw std::runtime_error("mapped model changed after its original full-file hash");
#else
    throw std::runtime_error("mapped model file validation requires native Linux");
#endif
}
void MappedModelFile::advise_dontneed(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept {
#if defined(__linux__)
    if (fd_ >= 0 && bytes && byte_offset < bytes_)
        (void)::posix_fadvise(fd_, static_cast<off_t>(byte_offset), static_cast<off_t>(std::min(bytes, bytes_ - byte_offset)),
            POSIX_FADV_DONTNEED);
#else
    (void)byte_offset; (void)bytes;
#endif
}

std::size_t MappedModelFile::prefault_parallel(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) const noexcept {
#if defined(__linux__)
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;
    const auto page = static_cast<std::uint64_t>(page_size);
    constexpr std::uint64_t chunk = std::uint64_t{64} << 20;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> chunks;
    try {
        for (const auto& [offset, length] : ranges) {
            if (!length || offset >= bytes_) continue;
            const auto begin = offset / page * page;
            const auto end = offset + std::min(length, bytes_ - offset);
            // Normal advice for the pass: an earlier MADV_RANDOM on the range would disable readahead.
            (void)::madvise(reinterpret_cast<void*>(mapping_base_ + begin), static_cast<std::size_t>(end - begin), MADV_NORMAL);
            for (auto at = begin; at < end; at += chunk) chunks.emplace_back(at, std::min(chunk, end - at));
        }
    } catch (...) {
        return 0;
    }
    // Workers only issue madvise calls and touch atomics: nothing below throws once the chunk list exists.
    std::atomic<std::size_t> next{0}, done{0};
    auto worker = [&]() noexcept {
        for (std::size_t c; (c = next.fetch_add(1)) < chunks.size();) {
            auto* address = reinterpret_cast<void*>(mapping_base_ + chunks[c].first);
            const auto length = static_cast<std::size_t>(chunks[c].second);
#if defined(MADV_POPULATE_READ)
            if (::madvise(address, length, MADV_POPULATE_READ) == 0) { done.fetch_add(1); continue; }
#endif
            (void)::madvise(address, length, MADV_WILLNEED);
        }
    };
    const auto workers = std::min<std::size_t>(chunks.size(), std::min(16U, std::max(1U, std::thread::hardware_concurrency())));
    std::vector<std::thread> threads;
    try {
        threads.reserve(workers);
        for (std::size_t w = 1; w < workers; ++w) threads.emplace_back(worker);
    } catch (...) {}  // fewer threads: the chunks left are taken by the threads already running and this one
    if (workers) worker();
    for (auto& t : threads) t.join();
    return done.load();
#else
    (void)ranges;
    return 0;
#endif
}

int MappedModelFile::drop_page_cache(std::uint64_t byte_offset, std::uint64_t bytes) const noexcept {
#if defined(__linux__)
    if (fd_ < 0) return EBADF;
    if (!bytes || byte_offset >= bytes_ || bytes > bytes_ - byte_offset) return EINVAL;
    return ::posix_fadvise(fd_, static_cast<off_t>(byte_offset), static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
#else
    (void)byte_offset; (void)bytes;
    return ENOSYS;
#endif
}

int MappedModelFile::open_direct_read_only() const {
#if defined(__linux__)
    try { validate_unchanged(); }
    catch (const std::exception& error) { throw ple_reclaim::DirectError(error.what()); }
    const auto path = "/proc/self/fd/" + std::to_string(fd_);
    const int direct = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (direct < 0) {
        const auto message = "PLE O_DIRECT: reopen unsupported/failed; rolling refused (errno " + std::to_string(errno) + ")";
        if (errno == EINVAL || errno == EOPNOTSUPP) throw ple_reclaim::DirectUnsupported(message);
        throw ple_reclaim::DirectError(message);
    }
    struct stat state{};
    if (::fstat(direct, &state) || file_state(state) != file_state_) {
        ::close(direct);
        throw ple_reclaim::DirectError("PLE O_DIRECT: reopened backing changed; rolling refused");
    }
    return direct;
#else
    throw ple_reclaim::DirectUnsupported("PLE O_DIRECT requires Linux; rolling refused");
#endif
}

void MappedModelFile::read_at(
    std::uint64_t byte_offset,
    std::span<std::byte> destination) const {
#if !defined(__linux__)
    static_cast<void>(byte_offset);
    static_cast<void>(destination);
    throw std::runtime_error(
        "mapped model direct reads require native Linux");
#else
    if (!valid() ||
        fd_ < 0) {
        throw std::logic_error(
            "mapped model file is not initialized");
    }
    if (byte_offset > bytes_ ||
        destination.size() >
            bytes_ - byte_offset) {
        throw std::out_of_range(
            "mapped model read exceeds file range");
    }
    if (byte_offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())) {
        throw std::overflow_error(
            "mapped model read offset does not fit off_t");
    }

    std::size_t completed = 0;
    while (completed < destination.size()) {
        const auto remaining =
            destination.size() -
            completed;
        const auto offset =
            byte_offset +
            static_cast<std::uint64_t>(
                completed);
        if (offset >
            static_cast<std::uint64_t>(
                std::numeric_limits<off_t>::max())) {
            throw std::overflow_error(
                "mapped model read range does not fit off_t");
        }

        const auto result =
            ::pread(
                fd_,
                destination.data() + completed,
                remaining,
                static_cast<off_t>(
                    offset));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                "mapped model pread failed");
        }
        if (result == 0) {
            throw std::runtime_error(
                "mapped model pread reached unexpected EOF");
        }
        completed +=
            static_cast<std::size_t>(
                result);
    }
#endif
}

ModelRuntimeBinding
MappedModelFile::build_runtime_binding(
    const ModelManifest& manifest) const {
    if (!valid() ||
        !content_identity_.valid()) {
        throw std::logic_error(
            "mapped model file is not initialized");
    }
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "mapped model binding requires valid manifest identity");
    }

    ModelRuntimeBinding binding;
    binding.manifest_layout_identity =
        manifest.layout_identity;
    binding.model_content_identity =
        content_identity_;
    binding.tensors.reserve(
        manifest.tensors.size());

    for (std::size_t index = 0;
         index < manifest.tensors.size();
         ++index) {
        const auto& tensor =
            manifest.tensors[index];

        RuntimeTensorBinding runtime;
        runtime.tensor_index =
            static_cast<std::uint32_t>(
                index);
        runtime.bytes =
            tensor.byte_size;
        runtime.owner_node =
            tensor.placement.node;

        if (tensor.placement.kind ==
            PlacementKind::remote) {
            runtime.storage =
                RuntimeTensorStorage::remote;
            binding.tensors.push_back(
                runtime);
            continue;
        }

        if (tensor.absolute_offset >
                bytes_ ||
            tensor.byte_size >
                bytes_ -
                    tensor.absolute_offset) {
            throw std::invalid_argument(
                "manifest tensor range exceeds mapped model file");
        }

        const auto maximum =
            std::numeric_limits<
                std::uintptr_t>::max();
        if (tensor.absolute_offset >
            static_cast<std::uint64_t>(
                maximum - mapping_base_)) {
            throw std::overflow_error(
                "mapped tensor address overflows uintptr_t");
        }

        runtime.storage =
            RuntimeTensorStorage::file_mapped;
        runtime.address =
            mapping_base_ +
            static_cast<std::uintptr_t>(
                tensor.absolute_offset);
        runtime.allocation_generation =
            allocation_generation_;
        binding.tensors.push_back(
            runtime);
    }

    validate_model_runtime_binding(
        manifest,
        binding);
    return binding;
}

void MappedModelFile::release() noexcept {
#if defined(__linux__)
    if (mapping_base_ != 0 &&
        bytes_ != 0) {
        static_cast<void>(
            ::munmap(
                reinterpret_cast<void*>(
                    mapping_base_),
                static_cast<std::size_t>(
                    bytes_)));
    }
    if (fd_ >= 0) {
        static_cast<void>(
            ::close(fd_));
    }
#endif
    fd_ = -1;
    mapping_base_ = 0;
    bytes_ = 0;
    allocation_generation_ = 0;
    logical_name_.clear();
    content_identity_ = {};
    file_sha256_ = {};
    file_state_ = {};
}

}  // namespace strix

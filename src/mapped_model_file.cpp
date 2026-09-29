#include "strix/mapped_model_file.hpp"

#include "strix/model_content_hash.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix {
namespace {

#if defined(__linux__)

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
          other.content_identity_) {
    other.fd_ = -1;
    other.mapping_base_ = 0;
    other.bytes_ = 0;
    other.allocation_generation_ = 0;
    other.content_identity_ = {};
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

        other.fd_ = -1;
        other.mapping_base_ = 0;
        other.bytes_ = 0;
        other.allocation_generation_ = 0;
        other.content_identity_ = {};
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
        struct stat before {};
        if (::fstat(fd, &before) != 0 ||
            before.st_size <= 0) {
            throw std::runtime_error(
                "cannot stat non-empty model file");
        }
        mapped.bytes_ =
            static_cast<std::uint64_t>(
                before.st_size);

        const auto proc_path =
            std::filesystem::path(
                "/proc/self/fd") /
            std::to_string(fd);
        const ModelContentFile content_file{
            mapped.logical_name_,
            proc_path,
        };
        const std::array files{
            content_file,
        };
        mapped.content_identity_ =
            build_model_content_identity(
                files);

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
    } catch (...) {
        mapped.release();
        throw;
    }

    return mapped;
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
}

}  // namespace strix

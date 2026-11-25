#include "strix/weight/source.hpp"
#include "strix/weight/confined_file.hpp"
#include "core/sha256.hpp"
#include "strix/file_snapshot.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <streambuf>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace strix::weight {
namespace {
Digest256 digest(std::span<const Byte> bytes) {
    core::Sha256 hash; hash.update(bytes.data(),bytes.size()); return {hash.finish()};
}
struct FileClose {
    void operator()(std::FILE* file) const noexcept { std::fclose(file); }
};
using FileHandle = std::unique_ptr<std::FILE, FileClose>;

class FileSource final : public Source {
    struct Snapshot { std::array<std::uint64_t,6> words{}; bool operator==(const Snapshot&) const = default; };
    std::FILE* file_; std::uint64_t size_; std::string shard_; Snapshot snapshot_;
    Snapshot snapshot() const {
#ifdef _WIN32
        auto handle=reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file_)));
        FILE_BASIC_INFO basic{}; FILE_STANDARD_INFO standard{};
        if(!GetFileInformationByHandleEx(handle,FileBasicInfo,&basic,sizeof basic) || !GetFileInformationByHandleEx(handle,FileStandardInfo,&standard,sizeof standard))
            throw std::runtime_error("cannot snapshot source handle");
        return {{{std::uint64_t(standard.EndOfFile.QuadPart),std::uint64_t(basic.LastWriteTime.QuadPart),std::uint64_t(basic.ChangeTime.QuadPart),0,0,0}}};
#else
        struct stat st{};if(fstat(fileno(file_),&st))throw std::runtime_error("cannot snapshot source handle");
        return {{{std::uint64_t(st.st_size),std::uint64_t(st.st_mtim.tv_sec),std::uint64_t(st.st_mtim.tv_nsec),std::uint64_t(st.st_ctim.tv_sec),std::uint64_t(st.st_ctim.tv_nsec),std::uint64_t(st.st_ino)}}};
#endif
    }
    mutable std::mutex mutex_; mutable std::optional<Digest256> digest_; mutable std::once_flag digest_once_;
public:
    explicit FileSource(FileHandle file, std::string shard)
        : file_(file.get()), size_(0), shard_(std::move(shard)) {
        snapshot_ = snapshot();
        size_ = snapshot_.words[0];
        (void)file.release();
    }
    explicit FileSource(const std::string& path,std::string shard) : file_(std::fopen(path.c_str(),"rb")), size_(0),
        shard_(shard.empty()?std::filesystem::path(path).filename().generic_string():std::move(shard)) {
        if (!file_) throw std::runtime_error("cannot open source");
        try {
            snapshot_=snapshot();size_=snapshot_.words[0];
#ifndef _WIN32
            LoaderCapture::record(path,fileno(file_));
#endif
        }catch(...){std::fclose(file_);throw; }
    }
    ~FileSource() override { std::fclose(file_); }
    FileKey key() const {
#ifdef _WIN32
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file_))), &info))
            throw std::runtime_error("cannot identify source handle");
        return {info.dwVolumeSerialNumber, (std::uint64_t(info.nFileIndexHigh) << 32U) | info.nFileIndexLow};
#else
        struct stat st{};
        if (fstat(fileno(file_), &st)) throw std::runtime_error("cannot identify source handle");
        return {std::uint64_t(st.st_dev), std::uint64_t(st.st_ino)};
#endif
    }
    std::string read_all(std::size_t cap, std::optional<std::size_t> exact) const {
        if (size_ > cap || (exact && size_ != *exact))
            throw std::runtime_error(shard_ + ": file length outside bounds");
        // This is a fresh handle. The EOF probe should go to the file.
        if (std::setvbuf(file_, nullptr, _IONBF, 0) != 0) throw std::runtime_error("cannot configure source handle");
        std::string data(static_cast<std::size_t>(size_), '\0');
        if (const auto status = read({0, size_}, {reinterpret_cast<Byte*>(data.data()), data.size()}); !status)
            throw std::runtime_error(shard_ + ": " + status.detail);
        if (std::fgetc(file_) != EOF || std::ferror(file_) || snapshot() != snapshot_)
            throw std::runtime_error(shard_ + ": file changed during read");
        return data;
    }
    std::uint64_t size() const noexcept override { return size_; }
    void discard(ByteRange range) const noexcept override {
#ifndef _WIN32
        if(range.offset<=size_ && range.bytes<=size_-range.offset)
            (void)posix_fadvise(fileno(file_),static_cast<off_t>(range.offset),static_cast<off_t>(range.bytes),POSIX_FADV_DONTNEED);
#else
        (void)range;
#endif
    }
    std::string shard_name() const override { return shard_; }
    Status read(ByteRange range,std::span<Byte> out) const override {
        if (range.offset>size_ || range.bytes>size_-range.offset || range.bytes!=out.size())
            return {Error::malformed,"source read outside bounds"};
        if (out.empty()) return {Error::ok,{}};
        std::lock_guard lock(mutex_);
        try {if(!(snapshot()==snapshot_))return {Error::stale_source,"source changed after indexing"};}
        catch(const std::exception& e){return {Error::stale_source,e.what()};}
#ifdef _WIN32
        if (_fseeki64(file_,static_cast<__int64>(range.offset),SEEK_SET))
#else
        if (fseeko(file_,static_cast<off_t>(range.offset),SEEK_SET))
#endif
            return {Error::device_failure,"source seek failed"};
        if (std::fread(out.data(),1,out.size(),file_)!=out.size())
            return {Error::stale_source,"source short read"};
        try {if(!(snapshot()==snapshot_))return {Error::stale_source,"source changed during read"};}
        catch(const std::exception& e){return {Error::stale_source,e.what()};}
        return {Error::ok,{}};
    }
    Digest256 identity() const override {
        {std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed after indexing");}
        std::call_once(digest_once_,[&] {
            core::Sha256 hash;std::array<Byte,65536> bytes;
            for(std::uint64_t at=0;at<size_;) {
                const auto n=std::size_t(std::min<std::uint64_t>(bytes.size(),size_-at));
                auto status=read({at,n},{bytes.data(),n});if(!status)throw std::runtime_error(status.detail);
                hash.update(bytes.data(),n);at+=n;
            }
            const auto result=Digest256{hash.finish()};
            std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed during digest");digest_=result;
        });
        std::lock_guard lock(mutex_);if(!(snapshot()==snapshot_))throw std::runtime_error("source changed after digest");return *digest_;
    }
};
class NamespaceSource final:public Source {
    std::shared_ptr<const Source> source_;std::string shard_;
public:
    NamespaceSource(std::shared_ptr<const Source> source,std::string shard):source_(std::move(source)),shard_(std::move(shard)){}
    Status read(ByteRange range,std::span<Byte> out) const override{return source_->read(range,out);}
    std::uint64_t size() const noexcept override{return source_->size();}
    Digest256 identity() const override{return source_->identity();}
    std::string shard_name() const override{return shard_;}
    void discard(ByteRange range) const noexcept override{source_->discard(range);}
};
class MemorySource final : public Source {
    std::vector<Byte> data_; std::string shard_;
    mutable std::once_flag identity_once_;mutable Digest256 identity_{};
public:
    MemorySource(std::span<const Byte> bytes,std::string shard):data_(bytes.begin(),bytes.end()),shard_(std::move(shard)){}
    std::uint64_t size() const noexcept override {return data_.size();}
    std::string shard_name() const override {return shard_;}
    Digest256 identity() const override {std::call_once(identity_once_,[&]{identity_=digest(data_);});return identity_;}
    Status read(ByteRange r,std::span<Byte> out) const override {
        if(r.offset>data_.size() || r.bytes>data_.size()-r.offset || r.bytes!=out.size())
            return {Error::malformed,"source read outside bounds"};
        if(!out.empty()) std::memcpy(out.data(),data_.data()+r.offset,out.size());
        return {Error::ok,{}};
    }
};
class ViewSource final : public Source {
    std::span<const Byte> bytes_; std::shared_ptr<const void> owner_; std::string shard_;
    mutable std::once_flag identity_once_; mutable Digest256 identity_{};
public:
    ViewSource(std::span<const Byte> bytes,std::shared_ptr<const void> owner,std::string shard):bytes_(bytes),owner_(std::move(owner)),shard_(std::move(shard)){}
    std::uint64_t size() const noexcept override{return bytes_.size();}
    std::string shard_name() const override{return shard_;}
    Digest256 identity() const override{std::call_once(identity_once_,[&]{identity_=digest(bytes_);});return identity_;}
    Status read(ByteRange range,std::span<Byte> out) const override {
        if(!owner_)return {Error::stale_source,"borrowed view has no owner"};
        if(range.offset>bytes_.size() || range.bytes>bytes_.size()-range.offset || range.bytes!=out.size())return {Error::malformed,"view read outside bounds"};
        if(!out.empty())std::memcpy(out.data(),bytes_.data()+range.offset,out.size());return {Error::ok,{}};
    }
};
class SourceBuffer : public std::streambuf {
    std::shared_ptr<const Source> source_; std::uint64_t at_=0;std::array<char,65536> buffer_{};
public:
    explicit SourceBuffer(std::shared_ptr<const Source> s):source_(std::move(s)){setg(buffer_.data(),buffer_.data(),buffer_.data());}
    int_type underflow() override {
        if(gptr()!=egptr())return traits_type::to_int_type(*gptr());
        if(at_==source_->size())return traits_type::eof();
        const auto n=std::size_t(std::min<std::uint64_t>(buffer_.size(),source_->size()-at_));
        if(!source_->read({at_,n},{reinterpret_cast<Byte*>(buffer_.data()),n}))return traits_type::eof();
        at_+=n;setg(buffer_.data(),buffer_.data(),buffer_.data()+n);return traits_type::to_int_type(*gptr());
    }
    pos_type seekoff(off_type offset,std::ios_base::seekdir dir,std::ios_base::openmode which) override {
        if(!(which&std::ios_base::in))return pos_type(off_type(-1));
        const auto current=at_-std::uint64_t(egptr()-gptr());
        const auto base=dir==std::ios_base::beg?0:dir==std::ios_base::cur?current:source_->size();
        if(offset<0 && std::uint64_t(-(offset+1))+1>base)return pos_type(off_type(-1));
        const auto next=offset<0?base-(std::uint64_t(-(offset+1))+1):base+std::uint64_t(offset);
        if(next>source_->size() || (offset>=0 && next<base))return pos_type(off_type(-1));
        at_=next;setg(buffer_.data(),buffer_.data(),buffer_.data());return pos_type(at_);
    }
    pos_type seekpos(pos_type pos,std::ios_base::openmode which) override {return seekoff(off_type(pos),std::ios_base::beg,which);}
};
class SourceStream final:public std::istream {
    SourceBuffer buffer_;
public: explicit SourceStream(std::shared_ptr<const Source> source):std::istream(nullptr),buffer_(std::move(source)){rdbuf(&buffer_);}
};
}

struct ConfinedDirectory::Impl {
#ifdef _WIN32
    HANDLE handle;
    explicit Impl(const std::filesystem::path& path)
        : handle(CreateFileW(path.c_str(), FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES,
                             FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                             FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr)) {
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open directory");
        FILE_ATTRIBUTE_TAG_INFO info{};
        if (!GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &info, sizeof info) ||
            (info.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != FILE_ATTRIBUTE_DIRECTORY) {
            CloseHandle(handle);
            throw std::runtime_error("not a directory");
        }
    }
    ~Impl() { CloseHandle(handle); }
#else
    int handle;
    explicit Impl(const std::filesystem::path& path) : handle(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)) {
        if (handle < 0) throw std::runtime_error("cannot open directory");
    }
    ~Impl() { ::close(handle); }
#endif

    std::shared_ptr<FileSource> open(const std::filesystem::path& name, bool missing_ok) const {
        const auto& basename = name.native();
        using Char = std::filesystem::path::value_type;
        if (basename.empty() || basename.size() > 255 || name == "." || name == ".." ||
            basename.find_first_of(std::filesystem::path("/\\:").native()) != std::filesystem::path::string_type::npos ||
            basename.find(Char{}) != std::filesystem::path::string_type::npos ||
            basename.back() == static_cast<Char>('.') || basename.back() == static_cast<Char>(' '))
            throw std::invalid_argument("confined file: expected a basename");
#ifdef _WIN32
        // Win32 CreateFile has no root-handle argument. NtCreateFile opens the
        // component relative to this directory even if its path was replaced.
        static const auto create = std::bit_cast<decltype(&NtCreateFile)>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile"));
        if (!create) throw std::runtime_error("relative file open unavailable");
        UNICODE_STRING relative{};
        relative.Length = static_cast<USHORT>(basename.size() * sizeof(wchar_t));
        relative.MaximumLength = relative.Length;
        relative.Buffer = const_cast<wchar_t*>(basename.data());
        OBJECT_ATTRIBUTES attrs{};
        InitializeObjectAttributes(&attrs, &relative, OBJ_CASE_INSENSITIVE, handle, nullptr);
        IO_STATUS_BLOCK io{};
        HANDLE child = INVALID_HANDLE_VALUE;
        const NTSTATUS status = create(&child, FILE_GENERIC_READ, &attrs, &io, nullptr, 0,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_OPEN,
                                       FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT | FILE_NON_DIRECTORY_FILE,
                                       nullptr, 0);
        if (status < 0) {
            constexpr unsigned long name_not_found = 0xc0000034UL;
            if (missing_ok && static_cast<unsigned long>(status) == name_not_found) return {};
            throw std::runtime_error("cannot open confined file: " + name.string());
        }
        FILE_ATTRIBUTE_TAG_INFO info{};
        if (GetFileType(child) != FILE_TYPE_DISK ||
            !GetFileInformationByHandleEx(child, FileAttributeTagInfo, &info, sizeof info) ||
            (info.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE))) {
            CloseHandle(child);
            throw std::runtime_error("not a confined regular file: " + name.string());
        }
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(child), _O_RDONLY | _O_BINARY | _O_NOINHERIT);
        if (fd < 0) {
            CloseHandle(child);
            throw std::runtime_error("cannot adopt confined file: " + name.string());
        }
        FileHandle file(_fdopen(fd, "rb"));
        if (!file) _close(fd);
#else
        const int fd = ::openat(handle, basename.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) {
            if (missing_ok && errno == ENOENT) return {};
            throw std::runtime_error("cannot open confined file: " + name.string());
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
            ::close(fd);
            throw std::runtime_error("not a confined regular file: " + name.string());
        }
        FileHandle file(::fdopen(fd, "rb"));
        if (!file) ::close(fd);
#endif
        if (!file) throw std::runtime_error("cannot adopt confined file: " + name.string());
        return std::make_shared<FileSource>(std::move(file), name.string());
    }
};

ConfinedDirectory::ConfinedDirectory(const std::filesystem::path& directory)
    : directory_(std::make_unique<Impl>(directory)) {}
ConfinedDirectory::~ConfinedDirectory() = default;

std::shared_ptr<const Source> ConfinedDirectory::open(const std::filesystem::path& name) const {
    return directory_->open(name, false);
}

std::optional<std::string> ConfinedDirectory::read(const std::filesystem::path& name, std::size_t max_bytes,
                                                   bool missing_ok, std::optional<std::size_t> exact_bytes,
                                                   std::optional<FileKey>* key) const {
    auto file = directory_->open(name, missing_ok);
    if (!file) return std::nullopt;
    if (key) *key = file->key();
    return file->read_all(max_bytes, exact_bytes);
}

Digest256 digest_bytes(std::span<const Byte> bytes) {return digest(bytes);}
namespace {
struct PinnedSource {
    std::mutex mutex;
    std::string path;
    std::shared_ptr<const Source> source;
    std::size_t uses = 0;
};
PinnedSource& pinned_source() { static PinnedSource pin; return pin; }
}  // namespace
std::optional<FileKey> source_file_key(const Source& source) {
    const auto* file = dynamic_cast<const FileSource*>(&source);
    if (!file) return std::nullopt;
    return file->key();
}
SourcePin::SourcePin(std::string path, std::shared_ptr<const Source> source) {
    auto& pin = pinned_source();
    std::lock_guard lock(pin.mutex);
    if (pin.source) throw std::logic_error("a source is already pinned");
    if (!source || path.empty()) throw std::invalid_argument("source pin needs a path and a source");
    pin.path = std::move(path);
    pin.source = std::move(source);
    pin.uses = 0;
}
SourcePin::~SourcePin() {
    auto& pin = pinned_source();
    std::lock_guard lock(pin.mutex);
    pin.source.reset();
    pin.path.clear();
}
std::size_t SourcePin::uses() const {
    auto& pin = pinned_source();
    std::lock_guard lock(pin.mutex);
    return pin.uses;
}
Status open_source(const std::string& path,std::shared_ptr<const Source>& out,std::string shard) {
    {
        auto& pin = pinned_source();
        std::lock_guard lock(pin.mutex);
        if (pin.source && path == pin.path) {  // the held file, under the shard name this path gives
            if (shard.empty()) shard = std::filesystem::path(path).filename().generic_string();
            out = shard == pin.source->shard_name() ? pin.source : namespace_source(pin.source, std::move(shard));
            ++pin.uses;
            return {Error::ok, {}};
        }
    }
    try {auto source=std::make_shared<FileSource>(path,std::move(shard));out=std::move(source);return {Error::ok,{}};}
    catch(const std::exception& e){return {Error::malformed,e.what()};}
}
std::shared_ptr<const Source> memory_source(std::span<const Byte> bytes,std::string shard) {return std::make_shared<MemorySource>(bytes,std::move(shard));}
std::shared_ptr<const Source> view_source(std::span<const Byte> bytes,std::shared_ptr<const void> owner,std::string shard){return std::make_shared<ViewSource>(bytes,std::move(owner),std::move(shard));}
std::unique_ptr<std::istream> source_stream(std::shared_ptr<const Source> source) {return std::make_unique<SourceStream>(std::move(source));}
TensorId tensor_id(const Source& source,std::string_view name,const Digest256& header) {
    core::Sha256 hash;const auto shard=source.shard_name();
    if(shard.empty()){auto d=source.identity();hash.update(d.bytes.data(),d.bytes.size());}
    else hash.update(shard.data(),shard.size());
    const char separator=0;hash.update(&separator,1);hash.update(header.bytes.data(),header.bytes.size());hash.update(name.data(),name.size());
    const auto d=hash.finish();TensorId id=0;for(unsigned i=0;i<8;++i)id|=std::uint64_t(d[i])<<(8*i);return id;
}
std::shared_ptr<const Source> namespace_source(std::shared_ptr<const Source> source,std::string shard){return std::make_shared<NamespaceSource>(std::move(source),std::move(shard));}
Status digest_slice(const Slice& slice,Digest256& out) {
    if(!slice.source || slice.range.offset>slice.source->size() || slice.range.bytes>slice.source->size()-slice.range.offset)return {Error::malformed,"digest slice outside source"};
    core::Sha256 hash;std::array<Byte,65536> bytes;
    for(std::uint64_t at=0;at<slice.range.bytes;) {
        const auto n=std::size_t(std::min<std::uint64_t>(bytes.size(),slice.range.bytes-at));
        auto status=slice.source->read({slice.range.offset+at,n},{bytes.data(),n});if(!status)return status;
        hash.update(bytes.data(),n);at+=n;
    }
    out={hash.finish()};return {Error::ok,{}};
}

Status read_slice(const Slice& slice,std::span<Byte> out) {
    if(!slice.source)return {Error::stale_source,"missing source ownership"};
    return slice.source->read(slice.range,out);
}
}

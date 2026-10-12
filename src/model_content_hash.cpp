#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace strix {
namespace {

constexpr std::array<std::uint32_t, 64> k{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotr(
    std::uint32_t value,
    unsigned shift) noexcept {
    return (value >> shift) |
           (value << (32U - shift));
}

class Sha256 {
public:
    void update(
        const std::uint8_t* data,
        std::size_t size) {
        if (size != 0 && data == nullptr) {
            throw std::invalid_argument(
                "SHA-256 update received null data");
        }
        if (size >
            std::numeric_limits<std::uint64_t>::max() -
                total_bytes_) {
            throw std::overflow_error(
                "SHA-256 input length overflow");
        }
        total_bytes_ +=
            static_cast<std::uint64_t>(size);

        for (std::size_t index = 0;
             index < size;
             ++index) {
            buffer_[buffer_size_++] =
                data[index];
            if (buffer_size_ ==
                buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    void update(
        std::string_view value) {
        update(
            reinterpret_cast<
                const std::uint8_t*>(
                    value.data()),
            value.size());
    }

    void update_u64(
        std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned index = 0;
             index < 8;
             ++index) {
            bytes[index] =
                static_cast<std::uint8_t>(
                    (value >> (index * 8U)) &
                    0xffULL);
        }
        update(
            bytes.data(),
            bytes.size());
    }

    [[nodiscard]] ModelContentIdentity
    finalize() {
        const auto message_bytes =
            total_bytes_;
        if (message_bytes >
            std::numeric_limits<std::uint64_t>::max() /
                8ULL) {
            throw std::overflow_error(
                "message length exceeds SHA-256 64-bit bit counter");
        }
        const auto bit_length =
            message_bytes * 8ULL;

        buffer_[buffer_size_++] =
            0x80U;

        if (buffer_size_ > 56U) {
            while (buffer_size_ <
                   buffer_.size()) {
                buffer_[buffer_size_++] =
                    0;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }

        while (buffer_size_ < 56U) {
            buffer_[buffer_size_++] = 0;
        }

        for (unsigned index = 0;
             index < 8;
             ++index) {
            buffer_[63U - index] =
                static_cast<std::uint8_t>(
                    (bit_length >>
                     (index * 8U)) &
                    0xffULL);
        }
        transform(buffer_.data());
        buffer_size_ = 0;

        ModelContentIdentity identity;
        for (std::size_t word = 0;
             word < state_.size();
             ++word) {
            identity.bytes[word * 4U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 24U);
            identity.bytes[word * 4U + 1U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 16U);
            identity.bytes[word * 4U + 2U] =
                static_cast<std::uint8_t>(
                    state_[word] >> 8U);
            identity.bytes[word * 4U + 3U] =
                static_cast<std::uint8_t>(
                    state_[word]);
        }
        return identity;
    }

private:
    void transform(
        const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64>
            schedule{};

        for (unsigned index = 0;
             index < 16;
             ++index) {
            const auto offset =
                index * 4U;
            schedule[index] =
                (static_cast<std::uint32_t>(
                     block[offset])
                 << 24U) |
                (static_cast<std::uint32_t>(
                     block[offset + 1U])
                 << 16U) |
                (static_cast<std::uint32_t>(
                     block[offset + 2U])
                 << 8U) |
                static_cast<std::uint32_t>(
                    block[offset + 3U]);
        }

        for (unsigned index = 16;
             index < 64;
             ++index) {
            const auto s0 =
                rotr(
                    schedule[index - 15U],
                    7U) ^
                rotr(
                    schedule[index - 15U],
                    18U) ^
                (schedule[index - 15U] >>
                 3U);
            const auto s1 =
                rotr(
                    schedule[index - 2U],
                    17U) ^
                rotr(
                    schedule[index - 2U],
                    19U) ^
                (schedule[index - 2U] >>
                 10U);
            schedule[index] =
                schedule[index - 16U] +
                s0 +
                schedule[index - 7U] +
                s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];

        for (unsigned index = 0;
             index < 64;
             ++index) {
            const auto sigma1 =
                rotr(e, 6U) ^
                rotr(e, 11U) ^
                rotr(e, 25U);
            const auto choose =
                (e & f) ^
                ((~e) & g);
            const auto temp1 =
                h +
                sigma1 +
                choose +
                k[index] +
                schedule[index];
            const auto sigma0 =
                rotr(a, 2U) ^
                rotr(a, 13U) ^
                rotr(a, 22U);
            const auto majority =
                (a & b) ^
                (a & c) ^
                (b & c);
            const auto temp2 =
                sigma0 +
                majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

struct FileDigest {
    ModelContentIdentity digest;
    std::uint64_t bytes = 0;
};

FileDigest hash_file_stable(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto before =
        std::filesystem::file_size(
            path,
            error);
    if (error) {
        throw std::runtime_error(
            "cannot stat model content file before hashing");
    }

    std::ifstream input(
        path,
        std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot open model content file");
    }

    Sha256 hash;
    std::vector<char> buffer(
        1024U * 1024U);
    std::uint64_t bytes = 0;

    while (input) {
        input.read(
            buffer.data(),
            static_cast<std::streamsize>(
                buffer.size()));
        const auto count =
            input.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<
                    const std::uint8_t*>(
                        buffer.data()),
                static_cast<std::size_t>(
                    count));
            bytes +=
                static_cast<std::uint64_t>(
                    count);
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "error while hashing model content file");
    }

    const auto after =
        std::filesystem::file_size(
            path,
            error);
    if (error ||
        before != after ||
        bytes != before) {
        throw std::runtime_error(
            "model content file changed while hashing");
    }

    return FileDigest{
        hash.finalize(),
        bytes,
    };
}

}  // namespace

ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path) {
    return hash_file_stable(path).digest;
}

ModelContentIdentity sha256_bytes(std::span<const std::byte> bytes) {
    Sha256 digest;
    digest.update(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    return digest.finalize();
}

ModelContentIdentity build_model_content_identity(
    std::span<const ModelContentFile> files) {
    if (files.empty()) {
        throw std::invalid_argument(
            "model content identity requires at least one file");
    }

    std::vector<const ModelContentFile*>
        ordered;
    ordered.reserve(files.size());
    for (const auto& file : files) {
        if (file.logical_name.empty()) {
            throw std::invalid_argument(
                "model content logical file name must be non-empty");
        }
        ordered.push_back(&file);
    }

    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto* left,
           const auto* right) {
            return left->logical_name <
                   right->logical_name;
        });

    for (std::size_t index = 1;
         index < ordered.size();
         ++index) {
        if (ordered[index - 1]->logical_name ==
            ordered[index]->logical_name) {
            throw std::invalid_argument(
                "model content logical file names must be unique");
        }
    }

    Sha256 aggregate;
    aggregate.update(
        "STRIX-MODEL-CONTENT-V1");

    for (const auto* file : ordered) {
        const auto digest =
            hash_file_stable(file->path);
        aggregate.update_u64(
            static_cast<std::uint64_t>(
                file->logical_name.size()));
        aggregate.update(
            file->logical_name);
        aggregate.update_u64(
            digest.bytes);
        aggregate.update(
            digest.digest.bytes.data(),
            digest.digest.bytes.size());
    }

    const auto identity =
        aggregate.finalize();
    if (!identity.valid()) {
        throw std::logic_error(
            "model content SHA-256 identity unexpectedly all zero");
    }
    return identity;
}

std::string model_content_identity_hex(
    const ModelContentIdentity& identity) {
    constexpr char digits[] =
        "0123456789abcdef";
    std::string output;
    output.resize(
        identity.bytes.size() * 2U);

    for (std::size_t index = 0;
         index < identity.bytes.size();
         ++index) {
        const auto byte =
            identity.bytes[index];
        output[index * 2U] =
            digits[byte >> 4U];
        output[index * 2U + 1U] =
            digits[byte & 0x0fU];
    }
    return output;
}

}  // namespace strix

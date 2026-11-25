#pragma once

// The SHA-256 of the H3 text-encoder pack is the shared one (core/sha256.hpp).

#include "core/sha256.hpp"

namespace strix::h3::te {
using strix::core::Sha256;
using strix::core::sha256_file_hex;
using strix::core::sha256_hex;
using strix::core::sha256_uses_sha_ni;
}  // namespace strix::h3::te

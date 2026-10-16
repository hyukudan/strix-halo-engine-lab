#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix {

// Physical FP32 matrix addresses only. Neither value denotes a numerical mode.
enum class GdnRecurrentStorage : std::uint32_t { key_value = 0, value_key = 1 };

inline void validate_gdn_recurrent_storage(GdnRecurrentStorage storage) {
    if (storage != GdnRecurrentStorage::key_value && storage != GdnRecurrentStorage::value_key)
        throw std::invalid_argument("unsupported GDN recurrent storage layout");
}

inline std::string_view gdn_recurrent_storage_marker(GdnRecurrentStorage storage) {
    validate_gdn_recurrent_storage(storage);
    return storage == GdnRecurrentStorage::key_value ? "" : ".state-vk-v1";
}

} // namespace strix

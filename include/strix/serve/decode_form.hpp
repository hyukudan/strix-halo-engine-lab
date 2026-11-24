#pragma once

#include <stdexcept>

namespace strix::serve {

// The decode expert form of a request, resolved in ONE place for the command line and the server: the family's default
// unless --packed-decode / --grouped-decode named one; naming both is refused. Returns true for the packed form.
constexpr bool resolve_decode_form(bool packed_flag, bool grouped_flag, bool family_default) {
    if (packed_flag && grouped_flag)
        throw std::invalid_argument("--packed-decode and --grouped-decode are exclusive");
    return packed_flag || (family_default && !grouped_flag);
}

}  // namespace strix::serve

#pragma once

// Binding helpers for the generated model constant headers (include/strix/model_data/<ident>.hpp, see
// strix/model/constants.hpp). Engines keep their own typed names and initialise them from the TOML value:
//
//     namespace d = strix::model_data::nemotron_3_5_bolt_e4b;
//     inline constexpr std::uint32_t hidden = strix::model::narrow<std::uint32_t>(d::dims::hidden);
//
// Both helpers are constant-evaluated in those initialisers. A value outside the target type (or a real that is
// not exactly a float) reaches a non-constexpr function, which makes the initialiser ill-formed: the build fails
// instead of truncating a model value. No SDK types; usable from HIP / CUDA translation units.

#include <cstdint>
#include <limits>

namespace strix::model {

namespace detail {
inline void model_constant_does_not_fit_its_type() {}  // not constexpr on purpose: reaching it is a compile error
} // namespace detail

template<class T> constexpr T narrow(std::int64_t value) {
    static_assert(std::numeric_limits<T>::is_integer, "narrow<T> binds integer model constants");
    if constexpr (std::numeric_limits<T>::is_signed) {
        if (value < std::int64_t(std::numeric_limits<T>::min()) || value > std::int64_t(std::numeric_limits<T>::max()))
            detail::model_constant_does_not_fit_its_type();
    } else {
        if (value < 0 || std::uint64_t(value) > std::uint64_t(std::numeric_limits<T>::max()))
            detail::model_constant_does_not_fit_its_type();
    }
    return static_cast<T>(value);
}

// A model real declared as a binary32 value (for example `norm_eps_f32`): exact, never rounded here.
constexpr float exact_float(double value) {
    const auto rounded = static_cast<float>(value);
    if (static_cast<double>(rounded) != value) detail::model_constant_does_not_fit_its_type();
    return rounded;
}

// A model real written in decimal (for example `eps = 1e-6`) bound to a binary32 constant: rounded to nearest from
// the TOML double. A binding that replaces a float literal static_asserts the old literal next to it, so the double
// rounding is checked at compile time, never assumed.
constexpr float nearest_float(double value) { return static_cast<float>(value); }

} // namespace strix::model

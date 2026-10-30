#pragma once
#include "minimax/dspark/minimax_auto.hpp"
#include "strix/serve/json.hpp"

namespace strix::minimax::serve {
inline dspark::Mode dspark_mode_update(dspark::Mode current, const strix::serve::Json& body) {
    if (body.find("dspark") && body.find("dspark_mode"))
        throw std::invalid_argument("dspark and dspark_mode conflict");
    if (const auto* v = body.find("dspark")) {
        if (!v->is_bool()) throw std::invalid_argument("dspark: boolean required");
        return v->as_bool() ? dspark::Mode::on : dspark::Mode::off;
    }
    if (const auto* v = body.find("dspark_mode")) {
        if (!v->is_string()) throw std::invalid_argument("dspark_mode: off | auto | on");
        const auto& mode = v->as_string();
        if (mode != "off" && mode != "auto" && mode != "on")
            throw std::invalid_argument("dspark_mode: off | auto | on");
        return dspark::parse_mode(mode);
    }
    return current;
}
}  // namespace strix::minimax::serve

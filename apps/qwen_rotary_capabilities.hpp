#pragma once
#include "strix/serve/json.hpp"
namespace strix::decode_cli {
inline void qwen_rotary_capabilities(serve::Json& document) {
    document.set("rotary", serve::parse_json(
        R"({"rope_mode":"native","scope":"server","yarn":{"supported":false,"scope":"CLI-only"}})"));
}
} // namespace strix::decode_cli

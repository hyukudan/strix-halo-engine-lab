#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

namespace strix::serve {

extern const BpeTokenizer::HfIdentity inkling2_mini_tokenizer_identity;

// Text binding of the Inkling 2 Mini checkpoint's chat_template.jinja. Effort is an
// alias, a number in [0, 0.99], or null for the default. Messages support text parts
// and the template's tool protocol. Media requires a separate model input path.
// The checkpoint template emits no BOS or assistant generation prefix.
[[nodiscard]] std::string render_inkling2_chat(const Json& messages, const Json& tools = {},
                                              const Json& reasoning_effort = {});

[[nodiscard]] std::string render_inkling2_chat_request(std::string_view json);

}  // namespace strix::serve

#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

namespace strix::serve {

extern const BpeTokenizer::HfIdentity inkling2_mini_tokenizer_identity;

// Text binding of the Inkling 2 Mini checkpoint's chat_template.jinja. Effort is an
// alias, a number in [0, 0.99], or null for the default. Messages support text parts
// and the template's tool protocol. With images, an image part (image, input_image, image_url) of a user or tool
// message renders the template's frame <|content_image|><|unused_200054|><|end_message|>; the placeholder is expanded
// to the tower rows by the vision path (src/inkling2/inkling2_vision_adapter.hpp). Without images they are refused.
// The checkpoint template emits no BOS or assistant generation prefix.
[[nodiscard]] std::string render_inkling2_chat(const Json& messages, const Json& tools = {},
                                              const Json& reasoning_effort = {}, bool images = false);

inline constexpr std::string_view kInkling2ImagePlaceholder = "<|unused_200054|>";

[[nodiscard]] std::string render_inkling2_chat_request(std::string_view json);
// The same request with images on: image parts (as the shared media layer normalizes them) render the frame.
[[nodiscard]] std::string render_inkling2_chat_media_request(std::string_view json);

}  // namespace strix::serve

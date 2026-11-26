#include "strix/ernie/serve.hpp"

#include "strix/serve/ernie_chat.hpp"
#include "strix/serve/session_runtime.hpp"

#include <utility>

namespace strix::ernie {

void bind_serve(serve::ApiConfig& config, const serve::SessionRuntime& runtime,
                std::shared_ptr<const serve::BpeTokenizer> tokenizer) {
    if (!tokenizer) throw std::invalid_argument("ERNIE: serving requires a tokenizer");
    runtime.configure_api(config);
    serve::bind_ernie_chat(config, std::move(tokenizer));
}

} // namespace strix::ernie

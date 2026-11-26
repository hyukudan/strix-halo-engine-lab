#pragma once

#include <memory>

namespace strix::serve {
struct ApiConfig;
class BpeTokenizer;
class SessionRuntime;
}

namespace strix::ernie {

// Connect an admitted session backend to the shared API and ERNIE token
// protocol. The caller attaches the same tokenizer instance to ApiService.
void bind_serve(serve::ApiConfig& config, const serve::SessionRuntime& runtime,
                std::shared_ptr<const serve::BpeTokenizer> tokenizer);

} // namespace strix::ernie

#pragma once

#include "chat_runner.hpp"
#include "strix/serve/http_server.hpp"

namespace strix::gemma4::engine {

// Portable family handler over shared HTTP/JSON/GenerationQueue primitives.
// Chat jobs use one FIFO worker; feature exports share exclusive backend access.
// Socket handling stays in src/serve.
class Api {
public:
    Api(Backend& backend, std::shared_ptr<const gemma4::serve::Tokenizer> tokenizer,
        DraftFactory drafts = {}, DraftMode mode = DraftMode::automatic, std::string model = "nemotron-bolt-e4b",
        Request defaults = {});
    ~Api();
    void handle(const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& writer);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
ChatRequest parse_chat_request(const strix::serve::Json& json, DraftMode default_mode = DraftMode::automatic,
    const Request& defaults = {});

} // namespace strix::gemma4::engine

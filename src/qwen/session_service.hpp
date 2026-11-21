#pragma once

#include "strix/serve/run.hpp"

namespace strix::decode_cli { struct Options; }
namespace strix::serve { class Json; struct ApiConfig; }

namespace strix::qwen {
using NativeHttp = std::function<int(const std::vector<std::string>&, const serve::ServiceFactory::ServeScoped&)>;
void install_session_service(serve::ServiceFactories&, NativeHttp);
void install_session_service(serve::ServiceFactories&);
[[nodiscard]] serve::Json inspect_http_options(const decode_cli::Options&);
void bind_http_session_support(serve::ApiConfig&, bool enabled);
} // namespace strix::qwen

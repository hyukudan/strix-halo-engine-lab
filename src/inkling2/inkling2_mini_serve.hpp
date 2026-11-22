#pragma once

#include "inkling2_mini_serve_options.hpp"
#include "strix/serve/cohort_session.hpp"
#include "strix/serve/run.hpp"

namespace strix::inkling2 {

using MiniServeBackend = std::function<std::unique_ptr<serve::CohortBackend>(const MiniServeOptions&)>;
struct MiniServeBackends {
    MiniServeBackend hip;
    MiniServeBackend cuda;
};

using MiniServeTokenizer = std::function<std::shared_ptr<const serve::BpeTokenizer>(const std::string&)>;
using MiniChat = std::function<std::string(const serve::Json&, const serve::Json&, const serve::Json&)>;

void install_mini_service(serve::ServiceFactories&, MiniServeBackends);
void install_mini_service(serve::ServiceFactories&, MiniServeBackends, MiniServeTokenizer, MiniChat);
} // namespace strix::inkling2

#pragma once

#include "strix/cli/cli.hpp"
#include "strix/profile/multi_device.hpp"
#include "strix/serve/run.hpp"

#include <string>

namespace strix::serve {

// strix-serve's native handoff (a family without an in-process adapter, or a CUDA backend served by an external
// launcher): the strix serve request for the selected model. It carries every launch input the shell parsed, the
// multi-device query included (--device-count N, --placement FORM), so the backend launcher resolves and admits the
// same placement the shell reported; nothing is rebuilt from defaults.
[[nodiscard]] inline cli::Request native_serve_request(const ServiceSelection& selection,
                                                       const ServeShell::NativeRequest& request) {
    cli::Request native;
    native.verb = cli::Verb::serve;
    native.model = selection.model.name;
    native.asset = request.asset;
    native.args = selection.arguments;
    native.profile = request.tier;
    native.backend = request.backend;
    native.host = request.bind;
    native.port = request.port;
    native.print_effective_config = request.print;
    native.device_count = request.placement.devices;
    if (request.placement.placement)
        native.placement = std::string(profile::multi_device::form_name(*request.placement.placement));
    return native;
}

} // namespace strix::serve

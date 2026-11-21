#include "strix/serve/run.hpp"

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strix::serve {
void ServiceFactories::install(std::string family, ServiceFactory f) {
    if (family.empty() || bool(f.create) == bool(f.create_scoped) || !f.validate || !f.inspect)
        throw std::invalid_argument("service factory requires family, validation, inspection and construction");
    if (!factories_.emplace(std::move(family), std::move(f)).second)
        throw std::invalid_argument("duplicate service factory");
}
const ServiceFactory* ServiceFactories::find(const std::string& family) const {
    const auto it = factories_.find(family);
    return it == factories_.end() ? nullptr : &it->second;
}

void serve_session(const HttpServerConfig& config, SessionService& service, bool owner_on_caller) {
    auto& runtime = service.runtime();
    try {
        HttpServer server(config, [&](const HttpRequest& request, HttpResponseWriter& writer) {
            service.handle(request, writer);
        });
        std::once_flag stopped;
        const auto stop = [&] {
            std::call_once(stopped, [&] {
                runtime.shutdown();
                service.shutdown();
                server.stop();
            });
        };
        std::exception_ptr owner_error, listener_error;
        const auto run_owner = [&] {
            try { runtime.run(); } catch (...) { owner_error = std::current_exception(); }
            stop();
        };
        const auto run_listener = [&] {
            try { server.serve(); } catch (...) { listener_error = std::current_exception(); }
            stop();
        };
        server.listen();
        service.listening(config, server.port());
        if (owner_on_caller) {
            std::jthread watcher([&](std::stop_token done) {
                while (!done.stop_requested()) {
                    if (service.stop_requested()) { stop(); return; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            });
            std::thread listener(run_listener);
            run_owner();
            listener.join();
        } else {
            std::thread owner(run_owner);
            run_listener();
            owner.join();
        }
        if (owner_error) std::rethrow_exception(owner_error);
        if (listener_error) std::rethrow_exception(listener_error);
    } catch (...) {
        service.failed();
        runtime.shutdown();
        service.shutdown();
        throw;
    }
    service.stopped();
}
} // namespace strix::serve

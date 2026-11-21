#pragma once

#include "strix/model_registry.hpp"
#include "strix/support/matrix.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace strix::cli {

enum class Verb { help, version, doctor, support, devices, models, inspect, plan, run, serve, update, rollback };

struct Request {
    Verb verb = Verb::help;
    std::string model, host, port, devices;
    std::uint32_t context = 0;
    bool json = false, dry_run = false;
    std::vector<std::string> args;
    support::Filter support_filter;
};

enum class Platform { linux_host, windows_host, macos_host, other };
enum class Status { ok, warn, fail, skip };
struct Check { std::string name; Status status; std::string detail; };

class DoctorProbes {
public:
    virtual ~DoctorProbes() = default;
    [[nodiscard]] virtual Platform platform() const = 0;
    [[nodiscard]] virtual Check os() const = 0;
    [[nodiscard]] virtual Check cpu() const = 0;
    [[nodiscard]] virtual Check ram() const = 0;
    [[nodiscard]] virtual Check runtime(std::string_view backend) const = 0;
    [[nodiscard]] virtual std::vector<std::string> architectures(std::string_view backend) const = 0;
    [[nodiscard]] virtual Check node(std::string_view name) const = 0;
    [[nodiscard]] virtual Check component(std::string_view name) const = 0;
};

struct Command {
    std::filesystem::path executable;
    std::vector<std::string> args;
};
using ExecutableExists = std::function<bool(const std::filesystem::path&)>;

[[nodiscard]] Request parse(const std::vector<std::string>& args);
[[nodiscard]] std::string doctor(const DoctorProbes&, const support::Matrix&, bool json);
[[nodiscard]] Command resolve(const Request&, const std::vector<model_registry::Entry>&,
                              const support::Matrix&, Platform, const std::filesystem::path& bin,
                              const ExecutableExists&);
[[nodiscard]] std::string command_line(const Command&);
int dispatch(const Request&, const std::filesystem::path& bin, const std::filesystem::path& data,
             const DoctorProbes&, const ExecutableExists&, std::ostream& out);
int main(const std::vector<std::string>& args);

} // namespace strix::cli

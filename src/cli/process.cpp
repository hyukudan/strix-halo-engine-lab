#include "detail.hpp"

#include <cerrno>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace strix::cli {
namespace {
namespace fs = std::filesystem;

#if defined(_WIN32)
// Windows C runtime argument quoting for direct CreateProcessW, including trailing backslashes.
template<class Char>
std::basic_string<Char> quote(std::basic_string_view<Char> arg) {
    std::basic_string<Char> result(1, '"');
    std::size_t slashes = 0;
    for (const auto ch : arg) {
        if (ch == '\\') {
            ++slashes;
            continue;
        }
        result.append(ch == '"' ? 2 * slashes + 1 : slashes, '\\');
        slashes = 0;
        result.push_back(ch);
    }
    result.append(2 * slashes, '\\');
    result.push_back('"');
    return result;
}
#else
std::string quote(std::string_view arg) {
    std::string result = "'";
    for (const auto ch : arg) {
        if (ch == '\'') result += "'\\''";
        else result.push_back(ch);
    }
    return result + "'";
}
#endif
} // namespace

fs::path executable_directory() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    const auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (size && size < path.size()) {
        path.resize(size);
        return fs::path(path).parent_path();
    }
#elif defined(__linux__)
    std::error_code ec;
    const auto target = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return target.parent_path();
#endif
    throw std::runtime_error("cannot locate the directory containing strix");
}

bool executable_exists(const fs::path& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return false;
#if defined(__linux__)
    return access(path.c_str(), X_OK) == 0;
#else
    return true;
#endif
}

std::string command_line(const Command& command) {
#if defined(_WIN32)
    const auto quote_arg = [](std::string_view value) { return quote<char>(value); };
#else
    const auto quote_arg = quote;
#endif
    auto result = quote_arg(command.executable.string());
    for (const auto& arg : command.args) result += " " + quote_arg(arg);
    return result;
}

int execute(const Command& command) {
#if defined(_WIN32)
    auto line = quote<wchar_t>(command.executable.native());
    for (const auto& arg : command.args) line += L" " + quote<wchar_t>(fs::path(arg).wstring());
    STARTUPINFOW startup{};
    startup.cb = static_cast<DWORD>(sizeof(startup));
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(command.executable.c_str(), line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child))
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "cannot launch engine");
    CloseHandle(child.hThread);
    DWORD code = 0;
    const bool success = WaitForSingleObject(child.hProcess, INFINITE) == WAIT_OBJECT_0 &&
        GetExitCodeProcess(child.hProcess, &code);
    const auto error = success ? 0 : GetLastError();
    CloseHandle(child.hProcess);
    if (!success) throw std::system_error(static_cast<int>(error), std::system_category(), "cannot wait for engine");
    return static_cast<int>(code);
#elif defined(__linux__)
    auto executable = command.executable.string();
    auto args = command.args;
    std::vector<char*> argv{executable.data()};
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    execv(executable.c_str(), argv.data());
    throw std::system_error(errno, std::generic_category(), "cannot launch engine");
#else
    (void)command;
    throw std::runtime_error("engine launch is unavailable on this OS");
#endif
}
} // namespace strix::cli

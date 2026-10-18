#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace strix::serve {

std::string HttpRequest::header(std::string_view name) const {
    for (const auto& [key, value] : headers) if (key == name) return value;
    return {};
}

std::string_view http_status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

namespace {
std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}
std::string trim(std::string_view text) {
    std::size_t a = 0, b = text.size();
    while (a < b && (text[a] == ' ' || text[a] == '\t')) ++a;
    while (b > a && (text[b - 1U] == ' ' || text[b - 1U] == '\t' || text[b - 1U] == '\r')) --b;
    return std::string(text.substr(a, b - a));
}

class Connection final : public HttpResponseWriter {
public:
    Connection(int fd, bool cors) : fd_(fd), cors_(cors) {}
    bool send_all(std::string_view data) {
        while (!data.empty()) {
            const auto n = ::send(fd_, data.data(), data.size(), MSG_NOSIGNAL);
            if (n < 0) { if (errno == EINTR) continue; gone_ = true; return false; }
            data.remove_prefix(static_cast<std::size_t>(n));
        }
        return true;
    }
    std::string head(int status, std::string_view content_type, const std::vector<std::pair<std::string, std::string>>& headers) const {
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + std::string(http_status_text(status)) + "\r\n";
        if (!content_type.empty()) out += "Content-Type: " + std::string(content_type) + "\r\n";
        if (cors_) out += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        for (const auto& [name, value] : headers) out += name + ": " + value + "\r\n";
        out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
        return out;
    }
    void send(int status, std::string_view content_type, std::string_view body,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) override {
        if (responded_) return;
        responded_ = true;
        auto out = head(status, content_type, headers);
        out += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        out += body;
        send_all(out);
    }
    bool begin_stream(int status, std::string_view content_type, const std::vector<std::pair<std::string, std::string>>& headers = {}) override {
        if (responded_) return false;
        responded_ = streaming_ = true;
        auto out = head(status, content_type, headers);
        out += "Transfer-Encoding: chunked\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n\r\n";
        return send_all(out);
    }
    bool write(std::string_view data) override {
        if (!streaming_ || gone_) return false;
        if (data.empty()) return !peer_closed();
        char size[32];
        const auto n = std::snprintf(size, sizeof(size), "%zx\r\n", data.size());
        std::string chunk(size, static_cast<std::size_t>(n));
        chunk.append(data);
        chunk += "\r\n";
        return send_all(chunk) && !peer_closed();
    }
    void end_stream() override {
        if (!streaming_) return;
        streaming_ = false;
        send_all("0\r\n\r\n");
    }
    bool client_gone() const override { return gone_ || const_cast<Connection*>(this)->peer_closed(); }
    // A readable socket with no data pending (orderly shutdown) or an error means the client left.
    bool peer_closed() {
        if (gone_) return true;
        pollfd p{fd_, POLLIN, 0};
        if (::poll(&p, 1, 0) > 0) {
            if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) { gone_ = true; return true; }
            char probe;
            const auto n = ::recv(fd_, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
            if (n == 0) { gone_ = true; return true; }
        }
        return false;
    }
    void reset() { responded_ = streaming_ = false; }
    bool responded() const { return responded_; }
    bool keep_alive = true;
private:
    int fd_;
    bool cors_;
    bool responded_ = false, streaming_ = false, gone_ = false;
};
}  // namespace

struct HttpServer::Impl {
    HttpServerConfig config;
    HttpHandler handler;
    int listen_fd = -1;
    std::uint16_t bound_port = 0;
    std::atomic<bool> stopping{false};
    std::atomic<unsigned> connections{0};
    std::mutex mutex;
    std::condition_variable idle;

    void handle(int fd, std::string peer) {
        timeval timeout{config.idle_timeout_seconds, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        Connection connection(fd, config.cors);
        std::string buffer;
        char chunk[16384];
        while (!stopping.load()) {
            // Header.
            std::size_t end = std::string::npos;
            while ((end = buffer.find("\r\n\r\n")) == std::string::npos) {
                if (buffer.size() > config.max_header_bytes) {
                    connection.keep_alive = false;
                    connection.send(431, "application/json", R"({"error":{"message":"request header too large","type":"invalid_request_error"}})");
                    ::close(fd); return;
                }
                const auto n = ::recv(fd, chunk, sizeof(chunk), 0);
                if (n <= 0) { ::close(fd); return; }
                buffer.append(chunk, static_cast<std::size_t>(n));
            }
            HttpRequest request;
            request.peer = peer;
            const std::string_view head(buffer.data(), end);
            const auto line_end = head.find("\r\n");
            const auto first = head.substr(0, line_end);
            const auto s1 = first.find(' '), s2 = first.rfind(' ');
            if (s1 == std::string_view::npos || s2 == s1) { ::close(fd); return; }
            request.method = std::string(first.substr(0, s1));
            request.target = std::string(first.substr(s1 + 1U, s2 - s1 - 1U));
            request.version = std::string(first.substr(s2 + 1U));
            const auto q = request.target.find('?');
            request.path = request.target.substr(0, q);
            if (q != std::string::npos) request.query = request.target.substr(q + 1U);
            for (std::size_t at = line_end == std::string_view::npos ? head.size() : line_end + 2U; at < head.size();) {
                auto next = head.find("\r\n", at);
                if (next == std::string_view::npos) next = head.size();
                const auto line = head.substr(at, next - at);
                const auto colon = line.find(':');
                if (colon != std::string_view::npos) request.headers.emplace_back(lower(line.substr(0, colon)), trim(line.substr(colon + 1U)));
                at = next + 2U;
            }
            buffer.erase(0, end + 4U);
            if (lower(request.header("transfer-encoding")).find("chunked") != std::string::npos) {
                connection.keep_alive = false;
                connection.send(501, "application/json", R"({"error":{"message":"chunked request bodies are not supported","type":"invalid_request_error"}})");
                ::close(fd); return;
            }
            const auto length_text = request.header("content-length");
            std::uint64_t length = 0;
            if (!length_text.empty()) {
                try { length = std::stoull(length_text); } catch (...) { ::close(fd); return; }
            }
            if (length > config.max_body_bytes) {
                connection.keep_alive = false;
                connection.send(413, "application/json", R"({"error":{"message":"request body too large","type":"invalid_request_error"}})");
                ::close(fd); return;
            }
            while (buffer.size() < length) {
                const auto n = ::recv(fd, chunk, sizeof(chunk), 0);
                if (n <= 0) { ::close(fd); return; }
                buffer.append(chunk, static_cast<std::size_t>(n));
            }
            request.body = buffer.substr(0, static_cast<std::size_t>(length));
            buffer.erase(0, static_cast<std::size_t>(length));
            const auto connection_header = lower(request.header("connection"));
            connection.keep_alive = request.version == "HTTP/1.1" ? connection_header != "close" : connection_header == "keep-alive";
            connection.reset();
            if (request.method == "OPTIONS" && config.cors) connection.send(204, "", "");
            else {
                try { handler(request, connection); }
                catch (const std::exception& error) {
                    if (!connection.responded()) {
                        std::string body = R"({"error":{"message":")";
                        for (const char c : std::string(error.what())) { if (c == '"' || c == '\\') body.push_back('\\'); body.push_back(c >= 0x20 ? c : ' '); }
                        body += R"(","type":"server_error"}})";
                        connection.send(500, "application/json", body);
                    }
                }
                if (!connection.responded()) connection.send(500, "application/json", R"({"error":{"message":"no response","type":"server_error"}})");
            }
            if (!connection.keep_alive || connection.client_gone()) break;
        }
        ::close(fd);
    }
};

HttpServer::HttpServer(HttpServerConfig config, HttpHandler handler) : impl_(std::make_shared<Impl>()) {
    impl_->config = std::move(config);
    impl_->handler = std::move(handler);
}
HttpServer::~HttpServer() {
    stop();
    std::unique_lock lock(impl_->mutex);
    impl_->idle.wait_for(lock, std::chrono::seconds(5), [&] { return impl_->connections.load() == 0U; });
}

void HttpServer::listen() {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICHOST;
    addrinfo* found = nullptr;
    const auto service = std::to_string(impl_->config.port);
    if (const int rc = ::getaddrinfo(impl_->config.bind_address.c_str(), service.c_str(), &hints, &found); rc != 0)
        throw std::runtime_error("HTTP bind address " + impl_->config.bind_address + ": " + ::gai_strerror(rc));
    const int fd = ::socket(found->ai_family, found->ai_socktype, found->ai_protocol);
    if (fd < 0) { ::freeaddrinfo(found); throw std::runtime_error("HTTP socket failed"); }
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, found->ai_addr, found->ai_addrlen) != 0 || ::listen(fd, 128) != 0) {
        const auto error = std::string(std::strerror(errno));
        ::freeaddrinfo(found); ::close(fd);
        throw std::runtime_error("HTTP listen on " + impl_->config.bind_address + ":" + service + " failed: " + error);
    }
    ::freeaddrinfo(found);
    sockaddr_storage bound{};
    socklen_t size = sizeof(bound);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &size);
    impl_->bound_port = ntohs(bound.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&bound)->sin6_port :
                                                            reinterpret_cast<sockaddr_in*>(&bound)->sin_port);
    impl_->listen_fd = fd;
}

void HttpServer::serve() {
    auto& impl = *impl_;
    if (impl.listen_fd < 0) listen();
    while (!impl.stopping.load()) {
        pollfd p{impl.listen_fd, POLLIN, 0};
        if (::poll(&p, 1, 200) <= 0) continue;
        sockaddr_storage peer{};
        socklen_t size = sizeof(peer);
        const int fd = ::accept(impl.listen_fd, reinterpret_cast<sockaddr*>(&peer), &size);
        if (fd < 0) continue;
        char host[NI_MAXHOST] = "";
        ::getnameinfo(reinterpret_cast<sockaddr*>(&peer), size, host, sizeof(host), nullptr, 0, NI_NUMERICHOST);
        if (impl.connections.load() >= impl.config.max_connections) {
            const char* busy = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            (void)::send(fd, busy, std::strlen(busy), MSG_NOSIGNAL);
            ::close(fd);
            continue;
        }
        ++impl.connections;
        std::thread([shared = impl_, fd, peer_name = std::string(host)] {
            shared->handle(fd, peer_name);
            {
                const std::lock_guard lock(shared->mutex);
                --shared->connections;
            }
            shared->idle.notify_all();
        }).detach();
    }
}

void HttpServer::stop() {
    impl_->stopping.store(true);
    if (impl_->listen_fd >= 0) { ::shutdown(impl_->listen_fd, SHUT_RDWR); ::close(impl_->listen_fd); impl_->listen_fd = -1; }
}
std::uint16_t HttpServer::port() const noexcept { return impl_->bound_port; }
unsigned HttpServer::active_connections() const noexcept { return impl_->connections.load(); }

}  // namespace strix::serve

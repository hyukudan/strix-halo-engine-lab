#include "strix/ds41_expert_server.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

namespace strix::ds41::xrpc {

namespace {

using Clock = std::chrono::steady_clock;
std::uint64_t ns_since(Clock::time_point t) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count());
}

// rank 0 -> rank 1 of a relay server: a fixed block, then the request payload (op_experts)
struct RelayControl {
    std::uint32_t magic = 0x31585352U;  // "RSX1"
    std::uint32_t op = 0;
    std::uint64_t bytes = 0;
};
static_assert(sizeof(RelayControl) == 16U);
enum RelayOp : std::uint32_t { op_experts = 1, op_quit = 2 };

class TcpPeer final : public PeerLink {
public:
    explicit TcpPeer(std::unique_ptr<Connection> c) : c_(std::move(c)) {}
    void send(std::span<const std::uint8_t> bytes) override { c_->send(MsgType::ack, 0, bytes); }
    void receive(std::span<std::uint8_t> bytes) override {
        const auto h = c_->receive_into(buf_);
        if (h.type != MsgType::ack || buf_.size() != bytes.size()) throw WireError("ds41 expert server: peer message of an unexpected size");
        if (!bytes.empty()) std::memcpy(bytes.data(), buf_.data(), bytes.size());
    }
    [[nodiscard]] std::string name() const override { return "tcp " + c_->peer(); }

private:
    std::unique_ptr<Connection> c_;
    std::vector<std::uint8_t> buf_;
};

// The items a rank of a relay computes: the low half's are first (plan order), rank 1 takes the rest.
std::size_t low_items(const ResponsePlan& plan) {
    std::size_t n = 0;
    while (n < plan.items.size() && plan.items[n].half == 0U) ++n;
    return n;
}

void fill_zero(float* out, std::size_t n) { std::fill(out, out + n, 0.0F); }

}  // namespace

std::unique_ptr<PeerLink> tcp_peer_listen(const std::string& host_port, std::chrono::milliseconds wait) {
    const auto [host, port] = split_host_port(host_port);
    Listener l(host, port);
    auto c = l.accept(wait, std::chrono::milliseconds(0));
    if (!c) throw WireError("ds41 expert server: no peer connected to " + host_port);
    return std::make_unique<TcpPeer>(std::move(c));
}
std::unique_ptr<PeerLink> tcp_peer_connect(const std::string& host_port, std::chrono::milliseconds wait) {
    return std::make_unique<TcpPeer>(connect(host_port, wait, std::chrono::milliseconds(0)));
}

std::uint64_t parse_layer_mask(const std::string& s, std::uint32_t layers) {
    if (layers > 64U) throw std::invalid_argument("at most 64 layers");
    if (s.empty() || s == "all") return layers == 64U ? ~0ULL : ((1ULL << layers) - 1U);
    std::uint64_t mask = 0;
    std::stringstream ss(s);
    for (std::string part; std::getline(ss, part, ',');) {
        const auto dash = part.find('-');
        const auto a = static_cast<std::uint32_t>(std::stoul(part.substr(0, dash)));
        const auto b = dash == std::string::npos ? a : static_cast<std::uint32_t>(std::stoul(part.substr(dash + 1)));
        if (a > b || b >= layers) throw std::invalid_argument("layer list out of range: " + s);
        for (auto l = a; l <= b; ++l) mask |= 1ULL << l;
    }
    return mask;
}

ServerStats serve(const ServerOptions& o, ExpertBackend& backend, PeerLink* peer) {
    const auto [host, port] = split_host_port(o.listen);
    Listener listener(host, port);
    if (o.on_listen) o.on_listen(listener.port());
    const auto& g = o.geometry;
    Residency endpoint = backend.residency();
    if (peer != nullptr) {
        const auto own = backend.residency();
        if (own.expert_first != 0U || own.expert_end != g.split || o.peer_residency.expert_first != g.split ||
            o.peer_residency.expert_end != g.experts)
            throw std::invalid_argument("ds41 expert server: a relay needs rank 0 = experts [0, split) and rank 1 = [split, experts)");
        endpoint = Residency{0U, g.experts, own.layer_mask & o.peer_residency.layer_mask};
    }
    ServerStats stats;
    std::vector<std::uint8_t> payload;
    for (;;) {
        if (o.stop != nullptr && o.stop->load()) break;
        auto conn = listener.accept(o.accept_wait.count() > 0 ? o.accept_wait : std::chrono::milliseconds(1000), o.io_timeout);
        if (!conn) {
            if (o.accept_wait.count() > 0 && o.once) throw WireError("ds41 expert server: no client connected");
            continue;
        }
        if (o.verbose) std::printf("session from %s\n", conn->peer().c_str());
        try {
            const auto h0 = conn->receive_into(payload, 1U << 20U);
            if (h0.type != MsgType::hello) throw WireError("ds41 expert server: expected HELLO");
            const auto hello = decode_hello(payload);
            HelloAck ack;
            ack.model_identity = o.model_identity;
            ack.geometry = g;
            ack.residency = endpoint;
            ack.world = peer != nullptr ? 2U : 1U;
            ack.max_rows = o.max_rows;
            ack.engine = backend.engine() + (peer != nullptr ? "+relay(" + peer->name() + ")" : "");
            ack.numeric_identity = backend.numeric_identity();
            if (!same_token(hello.token, o.token)) { ack.status = 1; ack.message = "session token refused"; }
            else if (!(hello.geometry == g)) { ack.status = 2; ack.message = "geometry differs"; }
            else if (hello.model_identity != 0U && o.model_identity != 0U && hello.model_identity != o.model_identity) {
                ack.status = 3;
                ack.message = "model identity differs";
            }
            conn->send(MsgType::hello_ack, h0.request, encode(ack));
            bool open = ack.status == 0U;
            if (!open) std::printf("session from %s refused: %s\n", conn->peer().c_str(), ack.message.c_str());
            if (open && o.busy_poll) conn->set_busy_poll(true);
            while (open) {
                const auto h = conn->receive_into(payload);
                const auto t0 = Clock::now();
                switch (h.type) {
                case MsgType::experts: {
                    const auto req = decode_experts(payload, g);
                    if (req.rows > o.max_rows) throw WireError("ds41 expert server: more rows than max_rows");
                    const auto plan = plan_response(req, g, endpoint);
                    const std::size_t n = plan.items.size(), n0 = peer != nullptr ? low_items(plan) : n;
                    const bool null = (req.flags & kNullCompute) != 0U;
                    if (peer != nullptr && n > n0) {
                        RelayControl c;
                        c.op = op_experts;
                        c.bytes = payload.size();
                        peer->send({reinterpret_cast<const std::uint8_t*>(&c), sizeof(c)});
                        peer->send(payload);
                    }
                    float* out = backend.answer_buffer(std::max<std::size_t>(n, 1U) * g.hidden);
                    ResultTiming t;
                    const auto tc = Clock::now();
                    if (n0 != 0U) {
                        if (null) fill_zero(out, n0 * g.hidden);
                        else backend.compute(req, plan, 0, n0, out, t.compute_ns);
                    }
                    if (t.compute_ns == 0U) t.compute_ns = ns_since(tc);
                    t.server_ns = ns_since(t0);
                    const auto head = encode_result_head(req.layer, static_cast<std::uint32_t>(n), g.hidden, t);
                    const std::uint64_t vec = g.vector_bytes();
                    // stream: header + this rank's items now, the peer's items when they arrive
                    const std::span<const std::uint8_t> first[2] = {head, {reinterpret_cast<const std::uint8_t*>(out), n0 * vec}};
                    conn->send_begin(MsgType::result, h.request, head.size() + n * vec, first);
                    if (n > n0) {
                        const auto tr = Clock::now();
                        auto* rest = reinterpret_cast<std::uint8_t*>(out + n0 * g.hidden);
                        peer->receive({rest, (n - n0) * vec});
                        stats.relay_s += static_cast<double>(ns_since(tr)) * 1e-9;
                        conn->send_raw({rest, (n - n0) * vec});
                    }
                    ++stats.requests;
                    stats.items += n;
                    stats.bytes_in += kHeaderBytes + payload.size();
                    stats.bytes_out += kHeaderBytes + head.size() + n * vec;
                    stats.compute_s += static_cast<double>(t.compute_ns) * 1e-9;
                    stats.server_s += static_cast<double>(ns_since(t0)) * 1e-9;
                    break;
                }
                case MsgType::ping: {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "{\"status\":\"ok\",\"protocol\":\"%s\",\"requests\":%llu,\"world\":%u}", kProtocol,
                                  static_cast<unsigned long long>(stats.requests), peer != nullptr ? 2U : 1U);
                    const std::string s(buf);
                    conn->send(MsgType::pong, h.request, {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()});
                    break;
                }
                case MsgType::close:
                    conn->send(MsgType::ack, h.request, {});
                    open = false;
                    break;
                default: throw WireError(std::string("ds41 expert server: unexpected ") + to_string(h.type));
                }
            }
        } catch (const std::exception& e) {
            std::printf("session error: %s\n", e.what());
            try {
                const std::string s = e.what();
                conn->send(MsgType::error, 0, {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()});
            } catch (...) {
            }
        }
        std::fflush(stdout);
        if (o.once) break;
    }
    if (peer != nullptr) {
        RelayControl c;
        c.op = op_quit;
        peer->send({reinterpret_cast<const std::uint8_t*>(&c), sizeof(c)});
    }
    return stats;
}

void follow(const Geometry& g, const Residency& endpoint, ExpertBackend& backend, PeerLink& peer) {
    std::vector<std::uint8_t> payload;
    for (;;) {
        RelayControl c;
        peer.receive({reinterpret_cast<std::uint8_t*>(&c), sizeof(c)});
        if (c.magic != RelayControl{}.magic) throw WireError("ds41 expert server rank 1: bad relay control block");
        if (c.op == op_quit) return;
        if (c.op != op_experts || c.bytes > kMaxMessageBytes) throw WireError("ds41 expert server rank 1: unknown relay operation");
        payload.resize(static_cast<std::size_t>(c.bytes));
        peer.receive(payload);
        const auto req = decode_experts(payload, g);
        const auto plan = plan_response(req, g, endpoint);
        const std::size_t n = plan.items.size(), n0 = low_items(plan);
        float* out = backend.answer_buffer(std::max<std::size_t>(n - n0, 1U) * g.hidden);
        std::uint64_t ns = 0;
        if ((req.flags & kNullCompute) != 0U) fill_zero(out, (n - n0) * g.hidden);
        else backend.compute(req, plan, n0, n - n0, out, ns);
        peer.send({reinterpret_cast<const std::uint8_t*>(out), (n - n0) * g.vector_bytes()});
    }
}

// ---- CPU reference backend ----

namespace {

class HostBackend final : public ExpertBackend {
public:
    HostBackend(const std::string& path, std::uint64_t mask, std::uint32_t first, std::uint32_t end, Geometry* geometry_out)
        : path_(path), in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("ds41 expert server: cannot open " + path);
        const auto file = strix::gguf::read_file(path);
        if (const auto* m = file.find_metadata("deepseek41.swiglu_limit"); m != nullptr && m->floating_value)
            limit_ = static_cast<float>(*m->floating_value);
        std::uint32_t layers = 0;
        for (const auto& t : file.tensors) {
            if (!t.name.starts_with("blk.") || !t.name.ends_with("_exps.weight")) continue;
            const auto dot = t.name.find('.', 4);
            const auto layer = static_cast<std::uint32_t>(std::stoul(t.name.substr(4, dot - 4)));
            layers = std::max(layers, layer + 1U);
            if (layer >= 64U || ((mask >> layer) & 1U) == 0U) continue;
            auto& L = tensors_[layer];
            if (t.name.ends_with("ffn_gate_exps.weight")) { L.gate = t.absolute_offset; geo_.hidden = static_cast<std::uint32_t>(t.dimensions[0]);
                geo_.inter = static_cast<std::uint32_t>(t.dimensions[1]); geo_.experts = static_cast<std::uint32_t>(t.dimensions[2]);
                if (t.ggml_type != 16U) throw std::runtime_error("ds41 expert server: gate experts are not IQ2_XXS"); }
            else if (t.name.ends_with("ffn_up_exps.weight")) L.up = t.absolute_offset;
            else if (t.name.ends_with("ffn_down_exps.weight")) { L.down = t.absolute_offset;
                if (t.ggml_type != 10U) throw std::runtime_error("ds41 expert server: down experts are not Q2_K"); }
        }
        (void)layers;
        geo_ = read_geometry(path);
        std::uint64_t have = 0;
        for (const auto& [l, L] : tensors_)
            if (L.gate && L.up && L.down) have |= 1ULL << l;
        if (have == 0U) throw std::runtime_error("ds41 expert server: no routed-expert layer of the list is in " + path);
        res_ = Residency{first, end == 0U ? geo_.experts : end, have};
        if (geometry_out) *geometry_out = geo_;
    }
    [[nodiscard]] Residency residency() const override { return res_; }
    [[nodiscard]] std::string engine() const override { return "host-reference"; }
    [[nodiscard]] std::string numeric_identity() const override { return "ds41-exact-v1"; }

    void compute(const ExpertsRequest& req, const ResponsePlan& plan, std::size_t first, std::size_t count, float* out,
                 std::uint64_t& compute_ns) override {
        const auto t0 = Clock::now();
        const auto H = geo_.hidden, I = geo_.inter, K = req.top_k;
        // the slots these items need, computed in parallel (one expert row per task)
        std::vector<std::int32_t> index(std::size_t{req.rows} * K, -1);
        std::vector<std::uint32_t> needed;
        const auto need = [&](std::uint32_t slot) {
            if (index[slot] < 0) {
                index[slot] = static_cast<std::int32_t>(needed.size());
                needed.push_back(slot);
            }
        };
        for (std::size_t i = 0; i < count; ++i) {
            const auto& it = plan.items[first + i];
            if (it.kind == PlanItem::Kind::raw) need(it.slot);
            else
                for (std::uint32_t j = 0; j < it.count; ++j) need(plan.fold_slots[plan.fold_begin[first + i] + j]);
        }
        std::vector<std::vector<float>> raw(needed.size());
        const auto& L = tensors_.at(req.layer);
        const auto work = [&](std::size_t n) {
            std::ifstream in(path_, std::ios::binary);
            const auto slot = needed[n];
            const auto r = slot / K, e = req.ids[slot];
            A8Activation x;
            x.codes.assign(req.codes.begin() + std::size_t{r} * H, req.codes.begin() + std::size_t{r + 1U} * H);
            x.deltas.assign(req.deltas.begin() + std::size_t{r} * geo_.groups(), req.deltas.begin() + std::size_t{r + 1U} * geo_.groups());
            x.sums.assign(geo_.groups(), 0);
            for (std::uint32_t gi = 0; gi < geo_.groups(); ++gi)
                for (std::uint32_t i = 0; i < 32U; ++i) x.sums[gi] += x.codes[gi * 32U + i];
            const std::uint64_t grow = H / super_block_elements * iq2_xxs_block_bytes, drow = I / super_block_elements * q2_k_block_bytes;
            std::vector<std::uint8_t> gate(grow * I), up(grow * I), down(drow * H);
            read(in, L.gate + e * grow * I, gate);
            read(in, L.up + e * grow * I, up);
            read(in, L.down + e * drow * H, down);
            std::vector<float> act(I);
            for (std::uint32_t o = 0; o < I; ++o) {
                const float gv = product(WeightFormat::iq2_xxs, {gate.data() + o * grow, grow}, x, H);
                const float uv = product(WeightFormat::iq2_xxs, {up.data() + o * grow, grow}, x, H);
                act[o] = swiglu_clamped(gv, uv, limit_);
            }
            const auto a = quantize_a8(act);
            auto& v = raw[n];
            v.resize(H);
            for (std::uint32_t o = 0; o < H; ++o) v[o] = product(WeightFormat::q2_k, {down.data() + o * drow, drow}, a, I);
        };
        {
            const std::size_t threads = std::min<std::size_t>(needed.size(), std::max(1U, std::thread::hardware_concurrency()));
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> pool;
            std::exception_ptr failure;
            std::mutex failure_mu;
            for (std::size_t t = 0; t < threads; ++t)
                pool.emplace_back([&] {
                    try {
                        for (std::size_t n; (n = next.fetch_add(1)) < needed.size();) work(n);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(failure_mu);
                        failure = std::current_exception();
                    }
                });
            for (auto& t : pool) t.join();
            if (failure) std::rethrow_exception(failure);
        }
        const auto raw_of = [&](std::uint32_t slot) -> const float* { return raw[static_cast<std::size_t>(index[slot])].data(); };
        for (std::size_t i = 0; i < count; ++i) {
            const auto& it = plan.items[first + i];
            float* dst = out + i * H;
            if (it.kind == PlanItem::Kind::raw) {
                const float* v = raw_of(it.slot);
                std::copy(v, v + H, dst);
            } else {
                fold_item(plan, first + i, req.weights, H, raw_of, dst);
            }
        }
        compute_ns = ns_since(t0);
    }

private:
    struct LayerTensors {
        std::uint64_t gate = 0, up = 0, down = 0;
    };
    void read(std::ifstream& in, std::uint64_t offset, std::vector<std::uint8_t>& dst) const {
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size()));
        if (!in) throw std::runtime_error("ds41 expert server: short read of " + path_);
    }
    std::string path_;
    std::ifstream in_;
    std::map<std::uint32_t, LayerTensors> tensors_;
    Geometry geo_;
    Residency res_;
    float limit_ = 10.0F;
};

}  // namespace

Geometry read_geometry(const std::string& gguf) {
    const auto file = strix::gguf::read_file(gguf);
    Geometry g;
    bool found = false;
    std::uint32_t layers = 0;
    for (const auto& t : file.tensors) {
        if (!t.name.starts_with("blk.") || !t.name.ends_with("_exps.weight")) continue;
        const auto dot = t.name.find('.', 4);
        layers = std::max(layers, static_cast<std::uint32_t>(std::stoul(t.name.substr(4, dot - 4))) + 1U);
        if (t.name.ends_with("ffn_gate_exps.weight") && t.dimensions.size() == 3U) {
            g.hidden = static_cast<std::uint32_t>(t.dimensions[0]);
            g.inter = static_cast<std::uint32_t>(t.dimensions[1]);
            g.experts = static_cast<std::uint32_t>(t.dimensions[2]);
            found = true;
        }
    }
    if (!found) throw std::runtime_error("ds41 expert server: no routed-expert tensor in " + gguf);
    if (const auto* m = file.find_metadata("deepseek41.num_hidden_layers"); m != nullptr && m->unsigned_value)
        layers = static_cast<std::uint32_t>(*m->unsigned_value);
    if (const auto* m = file.find_metadata("deepseek41.num_experts_per_tok"); m != nullptr && m->unsigned_value)
        g.top_k = static_cast<std::uint32_t>(*m->unsigned_value);
    g.layers = layers;
    g.split = g.experts / 2U;
    return g;
}

std::unique_ptr<ExpertBackend> make_host_backend(const std::string& gguf, std::uint64_t layer_mask, std::uint32_t expert_first,
                                                 std::uint32_t expert_end, Geometry* geometry_out) {
    return std::make_unique<HostBackend>(gguf, layer_mask, expert_first, expert_end, geometry_out);
}

}  // namespace strix::ds41::xrpc

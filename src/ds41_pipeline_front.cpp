#include "strix/ds41_pipeline_front.hpp"

#include <chrono>
#include <cstdio>

namespace strix::ds41::pipe {

StageClient::StageClient(const ClientOptions& o, const Hello& hello) : vocab_(hello.vocab) {
    conn_ = connect(o.connect, o.connect_wait, o.io_timeout);
    Hello h = hello;
    h.token = o.token;
    const auto m = call(MsgType::hello, encode(h), MsgType::hello_ack);
    ack_ = decode_hello_ack(m.payload);
    if (ack_.status != 0U) throw WireError("pipeline TAIL refused the session: " + ack_.message);
}

Message StageClient::call(MsgType type, std::span<const std::uint8_t> payload, MsgType answer, std::uint32_t flags) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t id = ++request_;
    conn_->send(type, id, payload, flags);
    auto m = conn_->receive();
    wire_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (m.header.type == MsgType::error) throw WireError("pipeline TAIL: " + decode_text(m.payload));
    if (m.header.type != answer || m.header.request != id)
        throw WireError(std::string("pipeline TAIL answered ") + to_string(m.header.type) + " to " + to_string(type));
    return m;
}

Result StageClient::step(const Boundary& boundary, bool want_logits) {
    StepMsg s;
    s.flags = want_logits ? kStepWantLogits : 0U;
    s.boundary = boundary;
    const auto m = call(MsgType::step, encode(s), MsgType::result);
    auto r = decode_result(m.payload, vocab_, 64U);
    if (r.tokens.size() != boundary.rows || r.position != boundary.position) throw WireError("pipeline TAIL: result of another step");
    return r;
}

void StageClient::commit(std::uint32_t kept) { (void)call(MsgType::commit, encode_u32(kept), MsgType::ack); }
void StageClient::reset() { (void)call(MsgType::reset, {}, MsgType::ack); }
StateMsg StageClient::export_state() { return decode_state(call(MsgType::export_state, {}, MsgType::state).payload); }
void StageClient::import_state(const StateMsg& state) { (void)call(MsgType::import_state, encode(state), MsgType::ack); }
std::string StageClient::ping() { return decode_text(call(MsgType::ping, {}, MsgType::pong).payload); }
void StageClient::close() { (void)call(MsgType::close, {}, MsgType::ack); }

std::string state_manifest(const deepseek41::AttentionState& s, std::uint32_t first_layer, std::uint32_t end_layer, std::uint64_t model_identity,
                           const std::string& engine) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"schema\":\"strix/ds41-state-manifest-v1\",\"format\":\"strix/ds41-attention-state-v1\",\"layers\":[%u,%u],"
                  "\"position\":%u,\"next_token\":%u,\"tokens\":%zu,\"sampler\":{\"kind\":\"greedy\"},\"speculative\":null,"
                  "\"model_identity\":\"%016llx\",\"digest\":\"%016llx\",\"engine\":\"%s\"}",
                  first_layer, end_layer, s.position, s.next_token, s.tokens.size(), static_cast<unsigned long long>(model_identity),
                  static_cast<unsigned long long>(s.digest()), engine.c_str());
    return buf;
}

}  // namespace strix::ds41::pipe

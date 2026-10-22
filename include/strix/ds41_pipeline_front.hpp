#pragma once

#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

// The FRONT side of a deepseek41 pipeline (strix/ds41-pipeline-v1): a backend that runs the embedding and target layers
// [0, K) of a step and hands over the boundary, and the client that drives a TAIL stage over TCP. Portable: the device
// backends (HIP DeviceForward, a CUDA backend) implement FrontBackend; the driver and the client do not depend on them.
namespace strix::ds41::pipe {

class FrontBackend {
public:
    virtual ~FrontBackend() = default;
    // Runs rows tokens[0..R) at positions position() .. + R - 1 through layers [0, K) and returns the boundary.
    virtual Boundary step(std::span<const std::uint32_t> tokens) = 0;
    // Keep rows [0, kept) of the last step (verify windows); the next step starts at the first dropped row.
    virtual void commit(std::uint32_t kept) = 0;
    virtual void reset() = 0;
    // The attention state of layers [0, K) after the positions so far (layers vector of size K).
    [[nodiscard]] virtual deepseek41::AttentionState export_state() const = 0;
    // A state with at least K layers (a merged pipeline state or a whole forward's): layers [0, K) are used.
    virtual void import_state(const deepseek41::AttentionState& state) = 0;
    [[nodiscard]] virtual std::uint32_t position() const = 0;
    [[nodiscard]] virtual std::uint32_t cut() const = 0;            // K
    [[nodiscard]] virtual std::uint32_t max_rows() const = 0;       // rows per step (8 for the decode window)
    [[nodiscard]] virtual std::string backend() const = 0;          // "hip", "host", "cuda"
    [[nodiscard]] virtual std::string numeric_identity() const = 0; // "ds41-exact-v1" when bitwise the HIP engine's
};

struct ClientOptions {
    std::string connect;                  // HOST:PORT of the TAIL (rank 0)
    std::array<std::uint8_t, 32> token{};  // session secret
    std::chrono::milliseconds connect_wait{120000}, io_timeout{600000};
};

// One session with a TAIL. Every call is one request and its answer; an ERROR answer throws WireError with its text.
class StageClient {
public:
    StageClient(const ClientOptions& options, const Hello& hello);
    [[nodiscard]] const HelloAck& ack() const noexcept { return ack_; }
    [[nodiscard]] Result step(const Boundary& boundary, bool want_logits = false);
    void commit(std::uint32_t kept);
    void reset();
    [[nodiscard]] StateMsg export_state();
    void import_state(const StateMsg& state);
    [[nodiscard]] std::string ping();
    void close();
    [[nodiscard]] std::uint64_t bytes_sent() const noexcept { return conn_->bytes_sent(); }
    [[nodiscard]] std::uint64_t bytes_received() const noexcept { return conn_->bytes_received(); }
    [[nodiscard]] double wire_seconds() const noexcept { return wire_seconds_; }  // inside send + receive of the requests

private:
    Message call(MsgType type, std::span<const std::uint8_t> payload, MsgType answer, std::uint32_t flags = 0);
    std::unique_ptr<Connection> conn_;
    HelloAck ack_;
    std::uint64_t request_ = 0;
    std::uint32_t vocab_ = 0;
    double wire_seconds_ = 0.0;
};

// The JSON manifest of a state message (strix/ds41-state-manifest-v1): layer range, position, next token, the token
// count, the sampler (greedy) and the model identity; the state's digest ties it to the bytes.
[[nodiscard]] std::string state_manifest(const deepseek41::AttentionState& state, std::uint32_t first_layer, std::uint32_t end_layer,
                                         std::uint64_t model_identity, const std::string& engine);

}  // namespace strix::ds41::pipe

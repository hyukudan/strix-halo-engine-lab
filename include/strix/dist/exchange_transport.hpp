#pragma once

#include "strix/dist/transport.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::dist {

// A Transport over a two-sided exchange (both ranks move one equal-size message each way, e.g. a communicator's tagged
// host path between engine calls). Lets a pair engine whose only link is such an exchange run a protocol written
// against Transport (serve::ReplicatedCohort / follow_cohort) without opening a second channel.
//
// send(m) is exchange(m, discarded) and receive(m) is exchange(zeros, m): a one-sided message costs one exchange of
// the same size, so both ranks must make matching send / receive calls of equal sizes in the same order, which is the
// Transport contract already (sizes are known from shared data). Not thread-safe; one owner thread per rank.
class ExchangeTransport final : public Transport {
public:
    using Exchange = std::function<void(std::span<const std::byte> out, std::span<std::byte> in)>;
    using Abort = std::function<void(const std::string& why)>;

    explicit ExchangeTransport(Exchange exchange, std::string name = "exchange", Abort abort = {})
        : exchange_(std::move(exchange)), name_(std::move(name)), abort_(std::move(abort)) {
        if (!exchange_) throw std::invalid_argument("exchange transport: no exchange");
    }

    void send(std::span<const std::byte> message) override {
        scratch_.resize(message.size());
        exchange(message, scratch_);
    }
    void receive(std::span<std::byte> message) override {
        zeros_.assign(message.size(), std::byte{0});
        exchange(zeros_, message);
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        if (aborted_) throw std::runtime_error("exchange transport: aborted (" + why_ + ")");
        if (out.size() != in.size()) throw std::invalid_argument("exchange transport: message sizes differ");
        exchange_(out, in);
    }
    [[nodiscard]] std::string name() const override { return name_; }
    void abort(const std::string& why) override {
        if (aborted_) return;
        aborted_ = true;
        why_ = why;
        if (abort_) abort_(why);
    }
    [[nodiscard]] bool aborted() const noexcept { return aborted_; }

private:
    Exchange exchange_;
    std::string name_;
    Abort abort_;
    std::vector<std::byte> scratch_, zeros_;
    bool aborted_ = false;
    std::string why_;
};

}  // namespace strix::dist

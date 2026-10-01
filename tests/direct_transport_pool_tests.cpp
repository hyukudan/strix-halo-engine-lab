#include "strix/direct_transport_pool.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    strix::transport::DirectFrameGeometry geometry;
    geometry.frame_bytes = 4096;
    geometry.ring_frames = 4095;
    geometry.dedicated_terminal_frame = true;

    const auto one =
        strix::transport::plan_direct_transport_pool(
            geometry);

    const auto expected_pool =
        4096ULL * 4095ULL;
    expect(one.tx_pool_bytes == expected_pool &&
               one.rx_pool_bytes == expected_pool &&
               one.total_gpu_bytes ==
                   2ULL * expected_pool,
           "one physical link owns separate full-ring TX and RX pools");
    expect(one.pool_alignment == 4096,
           "transport pools use frame alignment");
    expect(one.maximum_single_message_payload_bytes ==
               4096ULL * 4094ULL,
           "maximum message leaves one dedicated terminal frame");

    const std::array links{
        geometry,
        geometry,
    };
    const auto dual =
        strix::transport::plan_multi_link_transport_pools(
            links);
    expect(dual.links.size() == 2 &&
               dual.total_gpu_bytes ==
                   2ULL * one.total_gpu_bytes,
           "dual-link plan allocates independent pools per physical link");

    bool empty_rejected = false;
    try {
        static_cast<void>(
            strix::transport::plan_multi_link_transport_pools(
                std::span<
                    const strix::transport::DirectFrameGeometry>{}));
    } catch (const std::invalid_argument&) {
        empty_rejected = true;
    }
    expect(empty_rejected,
           "multi-link pool plan rejects zero physical links");

    if (failures == 0) {
        std::cout << "direct transport pool tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

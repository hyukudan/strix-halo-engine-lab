#include "core/backend.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/staged_ready.hpp"
#include "strix/pair/producer_publish.hpp"
#include "strix/session/backend.hpp"
#include "xport/channel.hpp"
#include "xport/channel_communicator.hpp"
#include "xport/communicator_backend.hpp"

#include <chrono>
#include <cstdint>

static_assert(strix::xport::wait_budget_ticks(std::chrono::milliseconds(5), 10) == 100);
static_assert(sizeof(strix::dist::Trailer) == 128);
static_assert(sizeof(strix::pair::ProducerPublish) == sizeof(strix::xport::ProducerTarget));

int main() { return 0; }

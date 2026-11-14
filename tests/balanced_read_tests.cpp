// Host test of the balanced read (src/core/balanced_read.hpp, src/xport/read_balance.hpp): the split arithmetic, and
// two ranks with readers of different speed that end with every block's bytes, each block published once.
#include "core/balanced_read.hpp"
#include "xport/read_balance.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace strix;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void split_arithmetic() {
    const std::vector<std::uint32_t> units(10, 10U);
    // owner: 100 open, blocks 2.. free (80); helper idle and twice as fast: x = 100 * 200 / 300 = 66 -> 6 blocks
    require(core::balanced_tail({0, 0, 10, 200}, {100, 80, 2, 100}, units, 0) == 4, "tail of 6 blocks");
    // the helper still has 30 of its own: x = (100 * 200 - 30 * 100) / 300 = 56 -> 5 blocks
    require(core::balanced_tail({30, 0, 10, 200}, {100, 80, 2, 100}, units, 0) == 5, "helper's own rows count");
    // never more than the free blocks
    require(core::balanced_tail({0, 0, 10, 5000}, {100, 20, 8, 100}, units, 0) == 8, "free blocks only");
    // the owner is the sooner one, a rate is unknown, the gain is below the bar: nothing
    require(core::balanced_tail({100, 80, 2, 100}, {0, 0, 10, 200}, units, 0) == 10, "owner sooner");
    require(core::balanced_tail({0, 0, 10, 0}, {100, 80, 2, 100}, units, 0) == 10, "helper rate unknown");
    require(core::balanced_tail({0, 0, 10, 200}, {100, 80, 2, 0}, units, 0) == 10, "owner rate unknown");
    require(core::balanced_tail({0, 0, 10, 200}, {100, 80, 2, 100}, units, 700000) == 10, "gain below the bar");  // 60 / 100 s
    require(core::balanced_tail({0, 0, 10, 200}, {100, 80, 2, 100}, units, 600000) == 4, "gain at the bar");
    // a partial last block
    const std::vector<std::uint32_t> uneven{10, 10, 10, 4};
    require(core::balanced_tail({0, 0, 4, 100}, {34, 24, 1, 100}, uneven, 0) == 2, "uneven blocks");  // x = 17 -> 4 + 10
    // both sides compute the same split and at most one direction
    const core::ReadProgress a{12, 0, 10, 900}, b{77, 60, 3, 250};
    require(core::balanced_tail(a, b, units, 0) < 10 && core::balanced_tail(b, a, units, 0) == 10, "one direction");
}

std::uint8_t byte_of(unsigned rank, std::size_t group, std::size_t block) {
    return static_cast<std::uint8_t>(1U + rank * 100U + group * 10U + block);
}

struct Link {  // a symmetric exchange between two threads
    std::mutex mutex;
    std::condition_variable changed;
    std::array<std::vector<std::vector<std::uint64_t>>, 2> sent;
    std::vector<std::uint64_t> exchange(unsigned rank, std::vector<std::uint64_t> message) {
        std::unique_lock<std::mutex> lock(mutex);
        const std::size_t index = sent[rank].size();
        sent[rank].push_back(std::move(message));
        changed.notify_all();
        changed.wait(lock, [&] { return sent[1U - rank].size() > index; });
        return sent[1U - rank][index];
    }
};

struct Rank {
    unsigned id = 0;
    std::chrono::milliseconds read_time{0};
    std::size_t groups = 2, blocks = 8;
    std::optional<core::BalancedReadQueue> queue;
    std::vector<std::uint8_t> data, help;
    std::vector<std::atomic<int>> published;
    std::size_t given = 0, taken = 0;
    bool fail = false;
    Rank(unsigned rank, int ms, bool failing = false) : id(rank), read_time(ms), data(16, 0), published(16), fail(failing) {
        queue.emplace(std::vector<std::vector<std::uint32_t>>(groups, std::vector<std::uint32_t>(blocks, 10U)));
    }
    void reader() {
        queue->run(
            [&](std::size_t g, std::size_t b) {
                std::this_thread::sleep_for(read_time);
                if (fail && g == 0 && b == 2) throw std::runtime_error("read failed");
                data[g * blocks + b] = byte_of(id, g, b);
            },
            [&](std::size_t g, std::size_t b) { ++published[g * blocks + b]; },
            [&](std::size_t g, std::size_t b) { data[g * blocks + b] = 0; },
            [&](std::size_t g, std::size_t first) {
                help.clear();
                for (std::size_t b = first; b < blocks; ++b) {
                    std::this_thread::sleep_for(read_time);
                    help.push_back(byte_of(1U - id, g, b));
                }
            });
    }
    void consumer(Link& link) {
        for (std::size_t g = 0; g < groups; ++g) {
            if (g == 0) std::this_thread::sleep_for(std::chrono::milliseconds(60));
            const auto mine = queue->freeze(g);
            const auto share = xport::read_balance_sync(mine, queue->units(g), queue->units(g), 0,
                [&](const xport::ReadBalanceMessage& send, xport::ReadBalanceMessage& got) {
                    const auto back = link.exchange(id, std::vector<std::uint64_t>(send.begin(), send.end()));
                    std::copy(back.begin(), back.end(), got.begin());
                });
            queue->resolve(g, share.give_from);
            const bool take = share.take_from < blocks, give = share.give_from < blocks;
            require(!(take && give), "both directions");
            if (take) queue->help(g, share.take_from);
            if (take || give) {  // the shipment: the same place of the exchange order on both ranks
                if (take) queue->wait_help();
                std::vector<std::uint64_t> rows;
                if (take)
                    for (const auto v : help) rows.push_back(v);
                const auto got = link.exchange(id, rows);
                if (give) {
                    require(got.size() == blocks - share.give_from, "shipped block count");
                    for (std::size_t b = share.give_from; b < blocks; ++b) {
                        data[g * blocks + b] = static_cast<std::uint8_t>(got[b - share.give_from]);
                        ++published[g * blocks + b];
                    }
                    given += got.size();
                } else {
                    taken += rows.size();
                }
            }
        }
        queue->close();
    }
};

void two_ranks(bool failing) {
    Link link;
    Rank fast(0, 1), slow(1, 30, failing);
    bool reader_failed = false;
    std::thread r0([&] { fast.reader(); });
    std::thread r1([&] {
        try {
            slow.reader();
        } catch (const std::runtime_error&) {
            reader_failed = true;
        }
    });
    std::thread c1([&] { slow.consumer(link); });
    fast.consumer(link);
    c1.join();
    r0.join();
    r1.join();
    require(reader_failed == failing, "read error reaches the caller of run()");
    for (unsigned rank = 0; rank < 2U; ++rank) {
        const Rank& r = rank == 0 ? fast : slow;
        for (std::size_t g = 0; g < r.groups; ++g)
            for (std::size_t b = 0; b < r.blocks; ++b) {
                require(r.published[g * r.blocks + b] == 1, "every block published once");
                const std::uint8_t v = r.data[g * r.blocks + b];
                require(v == byte_of(rank, g, b) || (failing && rank == 1U && v == 0), "block bytes");
            }
    }
    require(fast.given == 0 && slow.taken == 0 && fast.taken == slow.given, "one direction, same count on both ranks");
    if (!failing) require(slow.given != 0, "the slow rank gave blocks away");
    std::printf("two ranks%s: the slow rank gave %zu of 16 blocks\n", failing ? " (failing reader)" : "", slow.given);
}
}  // namespace

int main() {
    try {
        split_arithmetic();
        two_ranks(false);
        two_ranks(true);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "balanced read tests: %s\n", e.what());
        return 1;
    }
    std::puts("balanced read tests: ok");
    return 0;
}

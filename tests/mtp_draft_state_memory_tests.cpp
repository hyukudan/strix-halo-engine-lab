#include "strix/mtp_draft_state_memory.hpp"

#include "strix/model_contract.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeAllocator final
    : public strix::RuntimeMemoryAllocator {
public:
    std::uintptr_t next_device =
        static_cast<std::uintptr_t>(
            0x40000000ULL);
    std::vector<std::uint64_t> freed;

    std::uintptr_t allocate_device(
        std::uint64_t bytes,
        std::uint64_t alignment) override {
        const auto remainder =
            static_cast<std::uint64_t>(
                next_device) %
            alignment;
        if (remainder != 0) {
            next_device +=
                static_cast<std::uintptr_t>(
                    alignment - remainder);
        }
        const auto result =
            next_device;
        next_device +=
            static_cast<std::uintptr_t>(
                bytes + alignment);
        return result;
    }

    std::uintptr_t allocate_host(
        std::uint64_t,
        std::uint64_t) override {
        throw std::logic_error(
            "MTP draft state must not allocate host memory");
    }

    void free_device(
        std::uintptr_t,
        std::uint64_t bytes) noexcept override {
        freed.push_back(bytes);
    }

    void free_host(
        std::uintptr_t,
        std::uint64_t) noexcept override {}
};

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::
            qwen38_flash_next();
    const auto contract =
        strix::build_mtp_draft_state_contract(
            model);

    const auto plan =
        strix::plan_mtp_draft_state_memory(
            0x1234ULL,
            contract,
            2,
            2,
            256);

    expect(
        plan.state_width == 10240 &&
            plan.rows == 2 &&
            plan.total_bytes ==
                10240ULL * 2ULL * 2ULL &&
            plan.plan_identity != 0,
        "MTP draft-state memory plan accounts exact wide rows");

    FakeAllocator allocator;
    strix::AllocationGenerationSource
        generations;
    {
        auto owner =
            strix::MtpDraftStateMemoryOwner::
                allocate(
                    plan,
                    allocator,
                    generations);
        expect(
            owner.valid() &&
                owner.binding().
                    proposal_identity ==
                    0x1234ULL &&
                owner.binding().
                    alignment == 256 &&
                owner.binding().
                    device_base % 256ULL == 0 &&
                owner.binding().
                    allocation_generation == 1,
            "MTP draft-state owner binds aligned device storage to proposal identity");
        strix::validate_mtp_draft_state_memory_binding(
            plan,
            owner.binding());

        auto wrong =
            owner.binding();
        ++wrong.proposal_identity;
        bool rejected = false;
        try {
            strix::validate_mtp_draft_state_memory_binding(
                plan,
                wrong);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(
            rejected,
            "foreign proposal identity cannot reuse MTP draft-state storage");
    }

    expect(
        allocator.freed ==
            std::vector<std::uint64_t>({
                plan.total_bytes,
            }),
        "MTP draft-state owner releases exact device allocation");

    bool zero_rows_rejected = false;
    try {
        static_cast<void>(
            strix::plan_mtp_draft_state_memory(
                1,
                contract,
                0,
                2,
                256));
    } catch (const std::invalid_argument&) {
        zero_rows_rejected = true;
    }
    expect(
        zero_rows_rejected,
        "MTP draft-state plan rejects zero rows");

    if (failures == 0) {
        std::cout
            << "MTP draft-state memory tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

#pragma once
#include "strix/hip_mtp_resources.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/serial_session.hpp"
namespace strix {
class HipMtpTargetHidden;
// Minted by an actual completed target execution, never by pointer admission.
// Copies retain a snapshot slot and its actual allocation owner.
class HipMtpTargetHiddenTicket {
public:
    HipMtpTargetHiddenTicket()=default;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool same_producer(const HipMtpTargetHiddenTicket&) const noexcept;
    [[nodiscard]] MtpDeviceView view() const;
    [[nodiscard]] std::uint64_t source_position() const;
    [[nodiscard]] std::uint32_t input_token() const;
    [[nodiscard]] std::uint32_t predicted_token() const;
    [[nodiscard]] const SerialSessionContext& committed_session() const;
    [[nodiscard]] ModelContentIdentity target_content_identity() const;
private:
    friend class HipMtpTargetHidden;
    struct Lease;
    std::shared_ptr<const Lease> lease_;
};
// Serialized eager target producer. Owns its stream and bounded capture slots;
// direct dispatch uses that stream through the existing native launch scope.
// This producer supplies actual target rows; it does not qualify MTP arithmetic.
class HipMtpTargetHidden final : public std::enable_shared_from_this<HipMtpTargetHidden> {
public:
    [[nodiscard]] static std::shared_ptr<HipMtpTargetHidden> create(
        const ModelContract&,const CompiledSerialExecution&,
        std::shared_ptr<const RuntimeExecutionMemoryOwner>,
        std::shared_ptr<const HipModelResidency>,
        std::shared_ptr<HipRuntimeAllocator>,std::uint32_t retained_rows,
        std::uint64_t retained_byte_budget);
    ~HipMtpTargetHidden();
    [[nodiscard]] std::uint32_t retained_rows() const noexcept;
    HipMtpTargetHidden(const HipMtpTargetHidden&)=delete;
    HipMtpTargetHidden& operator=(const HipMtpTargetHidden&)=delete;
    [[nodiscard]] HipMtpTargetHiddenTicket execute_serial(
        SerialSessionContext&,const ExecutionRuntimeResources&,
        DirectLaunchDispatcher&,std::uint32_t input_token);
private:
    HipMtpTargetHidden();
    friend class HipMtpTargetHiddenTicket;
    friend struct HipMtpTargetHiddenTicket::Lease;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix

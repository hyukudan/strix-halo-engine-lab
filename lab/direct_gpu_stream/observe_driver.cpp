#include "observe_driver.hpp"
#include <algorithm>
#include <cerrno>
#include <sstream>
#include <stdexcept>

namespace strix::transport {
namespace {
std::string quote(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef"; std::string output = "\"";
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte == '"' || byte == '\\') { output += '\\'; output += character; }
        else if (byte < 32) { output += "\\u00"; output += digits[byte >> 4U]; output += digits[byte & 15U]; }
        else output += character;
    }
    return output + '"';
}
const char* status_name(DirectNativeQueryStatus status) {
    switch (status) {
    case DirectNativeQueryStatus::unavailable: return "unavailable";
    case DirectNativeQueryStatus::unsupported: return "unsupported";
    case DirectNativeQueryStatus::inaccessible: return "inaccessible";
    default: return "incompatible"; // This observer never emits verified.
    }
}
#if defined(__linux__)
bool nonzero(const __u8 (&value)[16]) { return std::any_of(std::begin(value), std::end(value), [](__u8 byte) { return byte != 0; }); }
std::string uuid(const __u8 (&value)[16]) {
    constexpr char digits[] = "0123456789abcdef"; std::string output;
    for (const auto byte : value) { output += digits[byte >> 4U]; output += digits[byte & 15U]; } return output;
}
void validate_query(const sgu_query& value) {
    if (value.fence.abi_major != SGU_ABI_MAJOR || value.fence.abi_minor != SGU_ABI_MINOR || value.fence.abi_cookie != SGU_ABI_COOKIE ||
        !nonzero(value.fence.device_instance) || !nonzero(value.fence.local_uuid) || !nonzero(value.fence.peer_uuid) ||
        std::equal(std::begin(value.fence.local_uuid), std::end(value.fence.local_uuid), std::begin(value.fence.peer_uuid)) ||
        value.frame_bytes != SGU_FRAME_BYTES || value.ring_frames < 32 || value.ring_frames > SGU_MAX_FRAMES ||
        !value.maximum_sg_entries || value.maximum_sg_entries > SGU_MAX_SG_ENTRIES ||
        value.maximum_window_bytes != static_cast<std::uint64_t>(value.ring_frames) * SGU_FRAME_BYTES ||
        !value.maximum_imported_bytes || value.maximum_imported_bytes > SGU_MAX_IMPORTED_BYTES || value.maximum_event_batch != SGU_EVENT_BATCH || value.reserved ||
        value.state > SGU_STATE_DISCONNECTED || value.state == SGU_STATE_IMPORTED || value.state == SGU_STATE_ARMED ||
        value.fence.connection_epoch || value.fence.pool_binding_identity || value.fence.import_generation) {
        throw std::runtime_error("observed QUERY ABI/limits/reserved/state/fresh-open identity is incompatible");
    }
}
bool valid_build(const std::optional<std::string>& value) {
    return value && !value->empty() && value->size() <= 128 && value->size() % 2 == 0 && std::all_of(value->begin(), value->end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
#endif
}
DirectGpuStreamDriverObservation observe_direct_gpu_stream_driver(const DirectStreamDeviceIdentity& route, const DirectGpuStreamDriverObservationPolicy& policy, LinuxDirectGpuStreamCalls calls) {
    DirectGpuStreamDriverObservation observation; observation.route = route;
#if defined(__linux__)
    if (!calls.fd_owner || !calls.ioctl_call || !calls.kernel_release || !calls.driver_build_identity) throw std::invalid_argument("driver observation requires complete trusted QUERY/FD/build boundary");
    DirectStreamFd fd;
    const auto identity = [&] {
        const auto current = calls.fd_owner->inspect(fd);
        if (current.device != route || current.open_generation != fd.open_generation) throw std::runtime_error("observed driver FD/route identity changed");
    };
    try {
        if (!policy.configured_gpu_direct || policy.expected_kernel_release.empty() || policy.expected_kernel_release.size() > 256 || !valid_build(policy.expected_driver_build_id))
            throw std::runtime_error("observation requires explicit configured-gpu-direct ownership and exact expected kernel/module before open");
        const auto kernel = calls.kernel_release(); const auto build = calls.driver_build_identity();
        if (!valid_build(build)) throw std::runtime_error("loaded module GNU build ID is unavailable before open");
        observation.kernel_release = kernel; observation.driver_build_id = *build;
        if (kernel != policy.expected_kernel_release || *build != policy.expected_driver_build_id) {
            observation.status = DirectNativeQueryStatus::incompatible; throw std::runtime_error("actual kernel/module differs from expected configured route before open");
        }
        fd = calls.fd_owner->open(route); observation.open_generation = fd.open_generation;
        if (fd.descriptor < 0 || !fd.open_generation) throw std::runtime_error("driver observation open returned no owned instance");
        identity();
        sgu_query query{}; query.fence.abi_major = SGU_ABI_MAJOR; query.fence.abi_minor = SGU_ABI_MINOR;
        const auto result = calls.ioctl_call(fd, SGU_IOC_QUERY, &query); identity();
        if (result.result || result.error) {
            observation.status = result.error == ENOTTY || result.error == EOPNOTSUPP ? DirectNativeQueryStatus::unsupported : DirectNativeQueryStatus::inaccessible;
            throw std::runtime_error("actual experimental driver QUERY is unsupported or inaccessible");
        }
        validate_query(query);
        const auto after_kernel = calls.kernel_release(); const auto after_build = calls.driver_build_identity(); identity();
        if (kernel.empty() || kernel.size() > 256 || kernel != after_kernel || build != after_build) { observation.status = DirectNativeQueryStatus::incompatible; throw std::runtime_error("observed driver/kernel build identity changed during QUERY"); }
        observation.query = query; observation.query_observed = true; observation.kernel_release = kernel;
    } catch (const std::exception& error) {
        observation.error = error.what();
        if (observation.status == DirectNativeQueryStatus::unavailable && !observation.query_observed && fd.descriptor >= 0) observation.status = DirectNativeQueryStatus::incompatible;
    }
    if (fd.descriptor >= 0) {
        try { identity(); calls.fd_owner->close(fd); observation.fd_close_requested = true; }
        catch (...) { observation.fd_retained = true; if (observation.error.empty()) observation.error = "driver observation FD cleanup identity is unknown or replaced"; }
    }
#else
    static_cast<void>(policy); static_cast<void>(calls); observation.status = DirectNativeQueryStatus::unsupported; observation.error = "experimental driver observation requires Linux";
#endif
    return observation;
}
std::string direct_gpu_stream_driver_observation_json(const DirectGpuStreamDriverObservation& value) {
    const auto boolean = [](bool flag) { return flag ? "true" : "false"; };
    std::ostringstream out;
    out << "{\"schema\":\"strix-direct-gpu-driver-observation-v1\",\"qualified\":false,\"production_qualified\":false,\"native_ready\":false,\"executed_payload\":false,\"status\":"
        << quote(value.query_observed && value.error.empty() ? "observed" : status_name(value.status)) << ",\"native_query_status\":" << quote(status_name(value.status))
        << ",\"query_observed\":" << boolean(value.query_observed) << ",\"canonical_path\":" << quote(value.route.canonical_path)
        << ",\"filesystem_device\":" << value.route.filesystem_device << ",\"inode\":" << value.route.inode << ",\"character_device_number\":" << value.route.character_device_number
        << ",\"open_generation\":" << value.open_generation << ",\"kernel_identity\":" << quote(value.kernel_release)
        << ",\"driver_build_id\":" << (value.driver_build_id.empty() ? "null" : quote(value.driver_build_id))
        << ",\"driver_identity\":" << (value.driver_build_id.empty() ? "null" : quote("thunderbolt_stream:gnu-build-id:" + value.driver_build_id))
        << ",\"fd_close_requested\":" << boolean(value.fd_close_requested) << ",\"fd_retained\":" << boolean(value.fd_retained) << ",\"error\":" << quote(value.error) << ",\"query\":";
#if defined(__linux__)
    if (value.query_observed) {
        const auto& query = value.query; const auto& fence = query.fence;
        out << "{\"abi_major\":" << fence.abi_major << ",\"abi_minor\":" << fence.abi_minor << ",\"abi_cookie\":" << fence.abi_cookie
            << ",\"abi_identity\":\"sgu-1.0-5354475055303031\",\"device_instance\":" << quote(uuid(fence.device_instance))
            << ",\"local_endpoint_identity\":" << quote(uuid(fence.local_uuid)) << ",\"peer_endpoint_identity\":" << quote(uuid(fence.peer_uuid))
            << ",\"route\":" << fence.route << ",\"in_hopid\":" << fence.in_hopid << ",\"out_hopid\":" << fence.out_hopid
            << ",\"connection_epoch\":" << fence.connection_epoch << ",\"pool_binding_identity\":" << fence.pool_binding_identity << ",\"import_generation\":" << fence.import_generation
            << ",\"raw_state\":" << query.state << ",\"frame_bytes\":" << query.frame_bytes << ",\"ring_frames\":" << query.ring_frames
            << ",\"maximum_window_bytes\":" << query.maximum_window_bytes << ",\"maximum_imported_bytes\":" << query.maximum_imported_bytes
            << ",\"maximum_sg_entries\":" << query.maximum_sg_entries << ",\"maximum_event_batch\":" << query.maximum_event_batch << '}';
    } else out << "null";
#else
    out << "null";
#endif
    return out.str() + "}\n";
}
} // namespace strix::transport

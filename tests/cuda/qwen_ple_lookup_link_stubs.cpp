// Link stubs for the CUDA PLE tests: src/ple_lookup.cpp is compiled as is (the engine's own hashing and
// FileMappedPleLookupBackend are the references), but its prefetch and planned-I/O backends reference
// engine modules the tests do not build. These entry points are never called by the tests; each throws.
#include "strix/execution_tensor_access.hpp"
#include "strix/session_state_memory.hpp"
#include "strix/sparse_lookup.hpp"
#include "strix/sparse_lookup_executor.hpp"

#include <stdexcept>

namespace strix {

const ExecutionTensorAccess* find_execution_tensor_access(const ExecutionTensorAccessPlan&, std::uint32_t, std::uint32_t) noexcept {
    return nullptr;
}
SessionStateAddress resolve_session_state(const SerialSessionStateLayout&, const SessionStateMemoryBinding&, SessionStateComponent) {
    throw std::logic_error("qwen PLE test stub: resolve_session_state");
}
SparseLookupPlan build_sparse_lookup_plan(const SparseLookupRequest&) {
    throw std::logic_error("qwen PLE test stub: build_sparse_lookup_plan");
}
SparseLookupIoExecution execute_sparse_lookup_io_plan(const SparseLookupPlan&, SparseLookupIoSource&) {
    throw std::logic_error("qwen PLE test stub: execute_sparse_lookup_io_plan");
}
SparseLookupIoExecution execute_sparse_lookup_io_plan(const SparseLookupPlan&, SparseLookupIoSource&, std::uint32_t) {
    throw std::logic_error("qwen PLE test stub: execute_sparse_lookup_io_plan");
}

}  // namespace strix

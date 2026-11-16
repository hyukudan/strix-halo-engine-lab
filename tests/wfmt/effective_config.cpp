#include <strix/weight/family.hpp>
#include <strix/weight/runtime.hpp>
#include <strix/weight/source.hpp>
#include "glm_rank_defaults.hpp"
#include <cstring>
namespace {
struct Memory final : strix::weight::Source {
    std::array<strix::weight::Byte,16> data{};
    strix::weight::Status read(strix::weight::ByteRange r,std::span<strix::weight::Byte> out) const override {
        if (r.offset>16 || r.bytes>16-r.offset || out.size()!=r.bytes) return {strix::weight::Error::malformed,"range"};
        std::memcpy(out.data(),data.data()+r.offset,out.size()); return {strix::weight::Error::ok,{}};
    }
    std::uint64_t size() const noexcept override { return 16; }
    strix::weight::Digest256 identity() const override { return strix::weight::digest_bytes(data); }
};
}
bool wfmt_effective_config_test() {
    using namespace strix::weight;
    namespace glm=strix::glm::rank_defaults;
    const auto env=[](const char*) -> const char* { return nullptr; };
    auto cli=glm::resolve(glm::scan({"--decode","1","--mtp-spec","0"}),env);
    auto serve=glm::resolve_serve({},false,env);
    if (glm::print(cli)!=glm::print(serve)) return false;
    if (glm::print(cli).find("weight-plans=pending")==std::string::npos) return false;
    FamilyPlanInputs input; input.registry=&builtin_registry(); input.hardware.backend=Backend::cpu;
    input.hardware.free_bytes=input.hardware.cache_budget_bytes=1024;
    input.workload={1,1,4,0,0,{},1,false,false};
    WeightMapEntry t{}; t.binding.tensor=t.weight.tensor=3; t.binding.role=Role::dense_up;
    t.binding.logical_shape=t.weight.shape={{1,4,0,0},2}; t.binding.expert_axis=UINT32_MAX;
    t.weight.storage={"plain.f32",1}; t.binding.admitted_formats={t.weight.storage};
    t.weight.payload={std::make_shared<Memory>(),{0,16}};
    auto descriptor=find_format(*input.registry,t.weight.storage); if (!descriptor || !descriptor->geometry(t.weight.parameters,t.weight.geometry)) return false;
    input.weights={t}; input.arithmetic={{"cpu-q8a8-tree32",1}}; input.incumbents={1};
    // Before a source registration exists, both real config entry points report the same decline.
    auto cs=attach_weight_plans(cli,input); auto ss=attach_weight_plans(serve,input);
    if (cs || ss || cs.code!=ss.code || glm::print(cli)!=glm::print(serve)) return false;
    Candidate c{}; c.kernel=1; c.storage=t.weight.storage; c.arithmetic=input.arithmetic[0];
    c.image={"image.raw",1}; c.backend=Backend::cpu; c.strategy=Strategy::native; c.exactness=Exactness::legacy_bitwise;
    c.cost.packed_bytes_per_weight={4,1}; c.cost.image_bytes_per_weight={4,1}; input.candidates={c};
    if (!attach_weight_plans(cli,input) || !attach_weight_plans(serve,input) || glm::print(cli)!=glm::print(serve)) return false;
    if (cli.weight_plans->plans[0].selected.kernel!=1 || glm::print(cli).find("arithmetic=cpu-q8a8-tree32@1")==std::string::npos) return false;
    // A vendor cannot acquire another vendor's capabilities via the adapter boundary.
    RuntimeAdapter api; api.callbacks=cpu_execution_context().callbacks;
    api.query_hardware=[](std::uint32_t,Hardware& h) -> Status { h.backend=Backend::hip; return {Error::ok,{}}; };
    ExecutionContext context; Hardware h{};
    return bind_runtime(Backend::cuda,api,nullptr,0,context,h).code==Error::unsupported_backend;
}

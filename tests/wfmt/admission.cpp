#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace strix::weight;
namespace {void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}}
bool wfmt_admission_test() try {
    std::array<Byte,128> bytes{};auto source=memory_source(bytes,"admission");TensorIndex index;index.container=Container::gguf;index.owner=source;
    RawTensor t{};t.id=tensor_id(*source,"x.weight");t.name="x.weight";t.wire_type=1;t.shape={{{2,32,0,0}},2};t.payload={source,{0,128}};index.tensors.push_back(t);
    TensorBinding binding{};binding.tensor=t.id;binding.role=Role::dense_up;binding.logical_shape=t.shape;binding.expert_axis=UINT32_MAX;binding.admitted_formats={{"plain.f16",1}};
    WeightMap map;check(bool(admit(builtin_registry(),{&index,{&binding,1},{}},map)),"explicit plain admission");check(map.size()==1 && map[0].weight.storage==Id{"plain.f16",1},"source storage retained");
    const auto original=describe_admission(map);binding.admitted_formats={{"plain.f32",1}};check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"format allowlist enforced");check(describe_admission(map)==original,"failed admission atomic");
    binding.admitted_formats.clear();binding.admitted_categories=1;check(bool(admit(builtin_registry(),{&index,{&binding,1},{}},map)),"explicit plain category opt-in");binding.admitted_categories=0;
    binding.admitted_formats={{"plain.f16",1}};binding.logical_shape.dims[0]=3;check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"shape mismatch rejected");binding.logical_shape=t.shape;
    binding.expert_axis=2;check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"expert axis rejected");binding.expert_axis=UINT32_MAX;
    index.tensors.push_back(t);check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"duplicate shard/tensor rejected");index.tensors.pop_back();
    // Reader-recognised Q4_1 is not silently executable.
    index.tensors[0].wire_type=3;index.tensors[0].payload.range.bytes=40;binding.admitted_formats={{"gguf.q4_1",1}};
    auto rejected=admit(builtin_registry(),{&index,{&binding,1},{}},map);check(!rejected && rejected.code==Error::unsupported_arithmetic,"parse-only never executable");
    index.tensors[0]=t;binding.admitted_formats={{"plain.f16",1}};
    WeightMap family;check(bool(bind_qwen(index,family)),"Qwen data schema");check(bool(bind_ds41(index,family)),"DS41 data schema");check(bool(bind_minimax(index,family)),"MiniMax data schema");check(bool(bind_glm(index,family)),"GLM data schema");
    check(bool(bind_h3(index,family)),"H3 data schema");check(bool(bind_vision(index,family)),"vision data schema");
    index.tensors[0].wire_type=0;index.tensors[0].shape={{{2,16,0,0}},2};index.tensors[0].payload.range.bytes=128;check(bool(bind_gemma4(index,family)),"Gemma data schema");
    check(family[0].binding.admitted_categories==0,"legacy binder uses exact allowlist");
    // Ownership remains attached to maps after both index and caller source disappear.
    index={};source.reset();std::array<Byte,128> observed{};check(bool(read_slice(family[0].weight.payload,observed)),"admitted map retains source");
    return true;
}catch(const std::exception& e){std::cerr<<"admission: "<<e.what()<<'\n';return false;}

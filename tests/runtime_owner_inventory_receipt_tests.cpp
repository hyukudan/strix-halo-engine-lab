#include "../apps/runtime_owner_inventory_receipt.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
int failures=0;
void expect(bool condition,const char* message) {
    if(!condition) {std::cerr << "FAIL: " << message << '\n';++failures;}
}
strix::UmaOwnerView view(std::uint64_t id,strix::UmaBackingIdentity backing,std::string category,
    std::uint64_t allocation,std::uint64_t offset,std::uint64_t bytes,std::shared_ptr<const void> pin) {
    strix::UmaOwnerView result;result.view_identity=id;result.backing=backing;
    result.category=std::move(category);result.allocation_bytes=allocation;
    result.view_offset=offset;result.view_bytes=bytes;result.allocation_pin=std::move(pin);
    return result;
}
}
int main() {
    auto owner=std::make_shared<int>(1);std::weak_ptr<int> live=owner;
    auto model=view(1,{1,1,1},"weights",128,0,128,owner);
    model.kind=strix::UmaBackingKind::file_mapping;
    auto alias=model;alias.view_identity=2;alias.category="registered-weights";
    std::vector<strix::UmaOwnerView> pools{
        view(3,{2,1,7},"runtime.scratch",64,8,48,owner),
        view(4,{2,2,8},"runtime.state.device",128,16,96,owner),
        view(5,{2,3,9},"runtime.state.host",16,0,8,owner)};
    pools[0].address_space=pools[1].address_space=strix::UmaAddressSpace::device;
    pools[0].address=0x1000;pools[1].address=0x2000;
    pools[2].address_space=strix::UmaAddressSpace::host;pools[2].address=0x3000;
    auto runtime=strix::build_uma_owner_inventory(pools);
    auto all=pools;all.push_back(model);all.push_back(alias);
    auto combined=strix::build_uma_owner_inventory(all);
    auto text=strix::decode_receipt::runtime_owner_inventory_receipt(combined,runtime,true);
    expect(text.find("\"covered_known_backing_count\":4")!=std::string::npos,
        "mapping aliases are counted once, distinct allocator pools remain distinct");
    expect(text.find("\"covered_unique_backing_reserved_bytes\":336")!=std::string::npos &&
        text.find("\"covered_known_file_mapping_bytes\":128")!=std::string::npos &&
        text.find("\"covered_known_allocated_storage_bytes\":208")!=std::string::npos,
        "covered reservation uses real raw extents rather than sum of logical or observed RSS");
    expect(text.find("\"view_offset\":8,\"view_bytes\":48,\"outside_view_reservation_bytes\":16")!=std::string::npos &&
        text.find("\"issuer\":2,\"allocation\":1,\"generation\":7")!=std::string::npos,
        "pool receipt preserves issued lease, generation, offset and reservation outside the requested view");
    expect(text.find("\"D1\":\"unaccounted\"")!=std::string::npos &&
        text.find("\"complete_resource_census\":false")!=std::string::npos &&
        text.find("\"physical_fit\":null")!=std::string::npos,
        "covered subset does not claim opaque owners or physical capacity");
    expect(text.find("4096")==std::string::npos && text.find("12288")==std::string::npos,
        "host/device addresses are not repurposed as public backing identities");
    combined.observation={99,1024,512,64,64,96};
    expect(strix::decode_receipt::runtime_owner_inventory_receipt(combined,runtime,true)==text,
        "RSS and GTT observations do not change declared backing accounting");
    auto unknown=view(6,{},"opaque",32,0,32,owner);all.push_back(unknown);
    auto incomplete=strix::build_uma_owner_inventory(all);
    text=strix::decode_receipt::runtime_owner_inventory_receipt(incomplete,runtime,false);
    expect(text.find("\"covered_unique_backing_reserved_bytes\":null")!=std::string::npos &&
        text.find("\"covered_unknown_backing_views\":1")!=std::string::npos &&
        text.find("\"D1\":\"absent\"")!=std::string::npos,
        "unknown storage remains unknown rather than a fabricated lease or exhaustive sum");
    auto escaped=runtime;escaped.pinned_views[0].category="quoted\"\n";
    expect(strix::decode_receipt::runtime_owner_inventory_receipt(combined,escaped,true).find("quoted\\\"\\u000a")!=std::string::npos,
        "category text is encoded with the production JSON string helper");
    auto bad=runtime;bad.pinned_views[0].view_bytes=65;
    bool rejected=false;
    try {(void)strix::decode_receipt::runtime_owner_inventory_receipt(combined,bad,true);}
    catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected,"out-of-allocation runtime views cannot produce a plausible padding receipt");
    owner.reset();model={};alias={};pools.clear();all.clear();unknown={};
    expect(!live.expired(),"retained inventory snapshots keep their actual source control blocks alive");
    return failures?1:0;
}
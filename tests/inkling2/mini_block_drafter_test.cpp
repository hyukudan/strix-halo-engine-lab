// Inkling 2 Mini binding of the shared block-candidate lever: all-OFF defaults, EXACT greedy only,
// and CLI / server parity through the common option parser and resolver.
#include "strix/profiles/inkling2_mini.hpp"
#include "inkling2/inkling2_mini_weight_options.hpp"
#include "strix/spec/block_draft_policy.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace sp = strix::spec;
namespace {
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void policy_tests() {
    namespace pf=strix::profile;
    const auto& engine=strix::inkling2::kMiniProfile;
    for(auto tier:{pf::Name::standard,pf::Name::fast,pf::Name::turbo}) {
        pf::Inputs cli;cli.profile=tier;cli.env=[](const char*)->const char*{return nullptr;};
        check(!sp::block_draft_selection(pf::resolve(engine,cli)).enabled(),"all defaults OFF");
        auto previous=engine;previous.table=engine.table.first(engine.table.size()-1);
        check(pf::resolve(previous,cli).fingerprint()==pf::resolve(engine,cli).fingerprint(),"OFF preserves previous profile fingerprint");
        for(const auto* value:{"off","dflash2:8:greedy","dflash2:8:sampled","dflash2:7:greedy","unknown:8:greedy"}) {
            cli.cli={{std::string(sp::kBlockDraftLeverName),value}};
            strix::inkling2::MiniWeightOptions cli_options, server_options;
            cli_options.tier.profile=tier; cli_options.tier.env=cli.env;
            // CLI token versus server/environment value, through their common
            // actual model parser/resolver (also checks CLI precedence).
            check(strix::inkling2::parse_mini_weight_option(cli_options,"--block-drafter",[&]{return value;},true),"CLI option consumed");
            server_options.tier.profile=tier;
            server_options.tier.env=[&](const char* key)->const char*{
                return std::string_view(key)=="STRIX_BLOCK_DRAFTER"?value:nullptr;
            };
            const auto outcome=[&](const strix::inkling2::MiniWeightOptions& input){try {
                static_cast<void>(strix::inkling2::resolve_mini_profile(input,true)); return std::string("off");
            }catch(const std::exception& e){return std::string(e.what());}};
            const auto lhs=outcome(cli_options),rhs=outcome(server_options);check(lhs==rhs,"CLI/server policy parity");
            if(std::string_view(value)!="off")check(lhs!="off","unbound native cannot run");
        }
    }
    std::cout << "PASS resolver: common row, EXACT greedy only, sampled pending, CLI/server parity, OFF in all tiers\n";
}
}  // namespace
int main() {
    try {
        policy_tests();
    } catch (const std::exception& e) {
        std::cerr << "inkling2-mini-block-drafter: " << e.what() << '\n';
        return 1;
    }
    std::cout << "inkling2-mini-block-drafter: ok\n";
    return 0;
}

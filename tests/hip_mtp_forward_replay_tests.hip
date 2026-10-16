// SOURCE fixture for binding independent full payloads. No fake native backend
// or synthesized qualification. Real candidate replay uses the public function
// with actual target tickets/resources and independently supplied full goldens.
#include "strix/hip_mtp_forward_replay.hpp"
#include <iostream>
#include <stdexcept>
namespace {
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
void fixture(){
    strix::HipMtpForwardGolden g;g.independent_source_content.bytes[0]=1U;g.target_content.bytes[0]=2U;g.sidecar_content.bytes[0]=3U;
    g.independent_reference_identity=4U;g.structural_plan_identity=5U;g.semantic_identity=6U;g.numerical_profile_identity=7U;
    strix::HipMtpForwardGoldenRow row;row.source_input_token=8U;row.next_input_token=9U;row.selected_token=1U;
    row.target_hidden_bf16={0x3f80U,0x4000U};row.next_hidden_bf16={0x4040U,0x4080U};row.full_logits={-0.0F,1.0F,+0.0F};
    g.rows.push_back(row);const auto original=strix::hip_mtp_forward_golden_payload_identity(g);
    require(original.valid(),"golden payload digest missing");
    for(unsigned fault=0;fault<13U;++fault){auto changed=g;
        if(fault==0U)changed.independent_source_content.bytes[0]^=1U;
        if(fault==1U)changed.target_content.bytes[0]^=1U;
        if(fault==2U)changed.sidecar_content.bytes[0]^=1U;
        if(fault==3U)++changed.semantic_identity;
        if(fault==4U)++changed.numerical_profile_identity;
        if(fault==5U)++changed.rows[0].source_position;
        if(fault==6U)++changed.rows[0].source_input_token;
        if(fault==7U)++changed.rows[0].next_input_token;
        if(fault==8U)++changed.rows[0].selected_token;
        if(fault==9U)changed.rows[0].target_hidden_bf16[1]^=1U;
        if(fault==10U)changed.rows[0].next_hidden_bf16[1]^=1U;
        if(fault==11U)changed.rows[0].full_logits[2]=-0.0F;
        if(fault==12U)changed.rows.push_back(row);
        require(strix::hip_mtp_forward_golden_payload_identity(changed)!=original,
            "golden metadata, signed-zero bits or complete payload mutation retained its digest");
    }
    bool rejected=false;try{g.rows.resize(4097U);(void)strix::hip_mtp_forward_golden_payload_identity(g);}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unbounded golden framing admitted");
}
}
int main(){try{fixture();std::cout<<"MTP replay payload-binding fixture passed; native replay not exercised\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

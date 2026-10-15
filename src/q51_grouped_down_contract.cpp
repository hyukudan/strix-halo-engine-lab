#include "strix/q51_grouped_down_contract.hpp"
#include <array>
#include <limits>
#include <stdexcept>
namespace strix::q51_grouped_down {
RequiredBytes required_bytes(Shape s) {
    if(s.hidden==0 || s.hidden>2560 || s.intermediate==0 || s.intermediate>640 ||
       s.intermediate%32U!=0 || s.experts==0 || s.experts>512 || s.top_k==0 ||
       s.top_k>10 || s.top_k>s.experts)
        throw std::invalid_argument("Q5_1 grouped-down unsupported H/I/E/K geometry");
    return {static_cast<std::uint64_t>(s.top_k)*2U*s.intermediate*4U,
        static_cast<std::uint64_t>(s.experts)*s.hidden*(s.intermediate/32U)*24U,
        static_cast<std::uint64_t>(s.top_k)*4U,static_cast<std::uint64_t>(s.top_k)*4U,
        4U,static_cast<std::uint64_t>(s.hidden)*4U};
}
void validate(const Request& r,const Bindings& b) {
    if(!r.enabled || r.approved_numerical_identity!=numerical_identity)
        throw std::invalid_argument("Q5_1 grouped-down requires explicit numerical identity");
    const auto n=required_bytes(r.shape);
    const std::array views{b.intermediates,b.weights,b.routes,b.route_weights,b.status,b.accumulation};
    const std::array sizes{n.intermediates,n.weights,n.routes,n.route_weights,n.status,n.accumulation};
    std::array<std::uintptr_t,6> ends{};
    for(unsigned i=0;i<6;++i) {
        if(views[i].address==0 || views[i].address%16U!=0 || views[i].bytes!=sizes[i] ||
           views[i].bytes>std::numeric_limits<std::uintptr_t>::max()-views[i].address)
            throw std::invalid_argument("Q5_1 grouped-down invalid exact/aligned address view");
        ends[i]=views[i].address+static_cast<std::uintptr_t>(views[i].bytes);
    }
    for(unsigned i=4;i<6;++i) for(unsigned j=0;j<i;++j)
        if(views[i].address<ends[j] && views[j].address<ends[i])
            throw std::invalid_argument("Q5_1 grouped-down writable owner alias");
}
} // namespace strix::q51_grouped_down

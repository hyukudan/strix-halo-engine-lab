#pragma once
#include <utility>
namespace strix::decode_cli {
template<class Drain,class Retire,class Allocate>
void complete_mtp_prompt_phase(Drain&& drain,Retire&& retire,Allocate&& allocate) {
    std::forward<Drain>(drain)();
    std::forward<Retire>(retire)();
    std::forward<Allocate>(allocate)();
}
} // namespace strix::decode_cli

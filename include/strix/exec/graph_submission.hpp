#pragma once

namespace strix::exec {

// The scalar ABI and the multi-row ABI publish different descriptors. They
// share the submission protocol: record without executing, publish on every
// use, then launch exactly once. A decline is legal only before launch.
template <class Record, class Publish, class Launch, class Retire>
bool submit_graph(bool needs_capture, Record record, Publish publish, Launch launch, Retire retire) {
    try {
        if ((!needs_capture || record()) && publish()) {
            launch();
            return true;
        }
    } catch (...) {
        retire();
        throw;
    }
    retire();
    return false;
}

}  // namespace strix::exec

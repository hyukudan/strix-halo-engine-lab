#pragma once
#include "strix/serve/engine.hpp"

namespace strix::decode_cli {
// Complete the ticket before propagating a fatal engine failure. The release
// callback removes it from the batch adapter's failure cleanup ownership.
template<class Engine, class Release>
void finish_http_session_job(serve::GenerationQueue& queue, serve::TakenJob& taken,
    Engine& engine, Release release) {
    serve::GenerationStats stats;
    try { stats = engine.generate(taken.job); }
    catch (const serve::GenerationRequestError& e) {
        queue.finish(taken, e.stats, e.what()[0] ? e.what() : "engine failure");
        release();
        if (engine.failed()) throw;
        return;
    } catch (const std::exception& e) {
        // Not a request-local refusal: complete this ticket, then keep the fail-stop path.
        stats.finish = "execution_failure";
        queue.finish(taken, std::move(stats), e.what()[0] ? e.what() : "engine failure");
        release();
        throw;
    } catch (...) {
        queue.finish(taken, {}, "engine failure");
        release();
        throw;
    }
    queue.finish(taken, std::move(stats));
    release();
}
} // namespace strix::decode_cli

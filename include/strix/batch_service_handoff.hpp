#pragma once

#include "strix/known_prefill_cancelled.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <map>
#include <string>
#include <stdexcept>
#include <thread>

namespace strix {

// STRIX_BATCH_SERVICE_LOOKAHEAD: event generations are block-local, including after each park.
class BatchServiceLookahead final {
public:
    explicit BatchServiceLookahead(unsigned limit):limit_(limit) {}
    [[nodiscard]] std::uint64_t advance() {++submitted_;return submitted_>limit_?submitted_-limit_:0U;}
    [[nodiscard]] std::uint64_t submitted() const noexcept {return submitted_;}
    void reset() noexcept {submitted_=0;}
private:
    unsigned limit_;
    std::uint64_t submitted_=0;
};

struct BatchServiceStats {
    std::uint64_t service_rounds=0,service_points=0,cancel_points_cancelled=0;
    std::uint64_t nested_admits=0,nested_rows=0,nest_rows_limit=0,nest_skips=0;
    double nested_ms=0.0;
    std::map<std::string,std::uint64_t> nest_skip_reasons;
    double service_ms_suspended=0.0,max_bundle_gpu_ms=0.0,service_overshoot_ms_max=0.0;
    void add(const BatchServiceStats& s) {
        service_rounds+=s.service_rounds;service_points+=s.service_points;cancel_points_cancelled+=s.cancel_points_cancelled;
        nested_admits+=s.nested_admits;nested_rows+=s.nested_rows;nested_ms+=s.nested_ms;nest_skips+=s.nest_skips;
        nest_rows_limit=std::max(nest_rows_limit,s.nest_rows_limit);
        for(const auto& [reason,count]:s.nest_skip_reasons) nest_skip_reasons[reason]+=count;
        service_ms_suspended+=s.service_ms_suspended;
        if(s.max_bundle_gpu_ms>max_bundle_gpu_ms) max_bundle_gpu_ms=s.max_bundle_gpu_ms;
        if(s.service_overshoot_ms_max>service_overshoot_ms_max) service_overshoot_ms_max=s.service_overshoot_ms_max;
    }
};

// One persistent stack owner; the engine services only while that stack is parked.
class BatchServiceHandoff final {
public:
    enum class State { idle, running, parked, servicing, resuming, completed, cancelled, failed, stopping };
    struct Stopped {};
    BatchServiceHandoff():worker_([this]{loop();}) {}
    ~BatchServiceHandoff(){shutdown();}
    BatchServiceHandoff(const BatchServiceHandoff&)=delete;
    BatchServiceHandoff& operator=(const BatchServiceHandoff&)=delete;
    void run(std::function<void()> job,const std::function<void()>& service) {
        std::unique_lock lock(mutex_);
        if(stopping_ || state_!=State::idle) throw std::logic_error("prefill worker is not idle");
        ++generation_;job_=std::move(job);error_={};state_=State::running;cv_.notify_all();
        std::exception_ptr service_error;
        for(;;) {
            cv_.wait(lock,[&]{return state_==State::parked || terminal();});
            if(terminal()) break;
            state_=State::servicing;lock.unlock();
            try {service();} catch(...) {service_error=std::current_exception();}
            lock.lock();if(!terminal()) state_=State::resuming;cv_.notify_all();
        }
        const auto error=error_;state_=State::idle;cv_.notify_all();
        if(error) std::rethrow_exception(error);
        if(service_error) std::rethrow_exception(service_error);
    }
    // Worker only. Stale notifications cannot transfer the current generation's lease.
    void park() {
        std::unique_lock lock(mutex_);const auto generation=generation_;
        if(stopping_) throw Stopped{};
        state_=State::parked;cv_.notify_all();
        cv_.wait(lock,[&]{return stopping_ || (generation_==generation && state_==State::resuming);});
        if(stopping_) throw Stopped{};
        state_=State::running;
    }
    void shutdown() noexcept {
        {std::lock_guard lock(mutex_);stopping_=true;cv_.notify_all();}
        if(worker_.joinable()) worker_.join();
    }
    void notify() noexcept {cv_.notify_all();}
    [[nodiscard]] State state() const {std::lock_guard lock(mutex_);return state_;}
private:
    bool terminal() const {return state_==State::completed || state_==State::cancelled || state_==State::failed;}
    void loop() {
        std::unique_lock lock(mutex_);
        for(;;) {
            cv_.wait(lock,[&]{return stopping_ || state_==State::running;});
            if(stopping_) {state_=State::stopping;return;}
            auto job=std::move(job_);lock.unlock();State result=State::completed;std::exception_ptr error;
            try {job();} catch(const Stopped&) {result=State::cancelled;error=std::current_exception();}
            catch(const KnownPrefillCancelled&) {result=State::cancelled;error=std::current_exception();}
            catch(...) {result=State::failed;error=std::current_exception();}
            lock.lock();error_=error;state_=result;cv_.notify_all();
            cv_.wait(lock,[&]{return stopping_ || state_==State::idle || state_==State::running;});
            if(stopping_) return;
        }
    }
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_=State::idle;
    bool stopping_=false;
    std::uint64_t generation_=0;
    std::function<void()> job_;
    std::exception_ptr error_;
    std::thread worker_;
};

} // namespace strix

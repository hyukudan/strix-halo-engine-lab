#include "strix/mtp_step_logits_digest.hpp"
#include "strix/qwen_target_state_digest.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace strix;
namespace {
void expect(bool value) {if(!value)throw std::runtime_error("MTP step digest test failed");}
ModelContentIdentity row(unsigned value) {
    const std::array<float,4> logits{static_cast<float>(value),-0.0F,1.0F,-1.0F};
    return sha256_bytes(std::as_bytes(std::span(logits)));
}
}
int main() {
    const std::array<std::uint32_t,2> history{11,12};
    const auto target=[&](unsigned variant) {
        QwenTargetStateDigestFrame frame(2,1,1);
        frame.component(variant==1?"index.L3":"kv.L3",variant==2?8:16,row(0));
        frame.component("position_history",8,variant==3?row(1):row(0));
        return frame.finish(history,variant==4?14:13);
    };
    const auto common=target(0);
    expect(common==target(0));
    expect(model_content_identity_hex(common)=="fc5a5e991eeb5a1f9d9bb6cb8456bd405c71afec4634c87a566869728a05d2f8");
    for(unsigned v=1;v<5;++v)expect(common!=target(v));
    bool refused=false;try {QwenTargetStateDigestFrame missing(2,1,1);(void)missing.finish(history,13);}
    catch(const std::invalid_argument&){refused=true;}expect(refused);
    // Same committed inputs with different physical schedules and rejected tails.
    for(unsigned rows:{2U,4U,8U})for(unsigned keep=1;keep<=rows;++keep) {
        std::vector<ModelContentIdentity> captured;
        for(unsigned r=0;r<rows;++r)captured.push_back(row(r));
        const auto retained=captured;
        for(auto& hash:captured)hash=row(99); // capture slot reused after snapshots
        MtpStepLogitsDigest window(31),serial(31);
        for(unsigned r=0;r<keep;++r)window.commit(31+r,retained[r]);
        for(unsigned r=0;r<keep;++r)serial.commit(31+r,row(r));
        expect(window.complete(31+keep) && window.count()==keep);
        expect(window.chain()==serial.chain()); // rejected suffix never contributes
        MtpStepLogitsDigest changed(31);
        for(unsigned r=0;r<keep;++r)changed.commit(31+r,row(r+1));
        expect(changed.chain()!=window.chain());
    }
    MtpStepLogitsDigest known(5);known.commit(5,row(0));
    expect(model_content_identity_hex(known.chain())=="7124cdcc665a0550d1a103dc2b491bb7e399d877e319eafdc143effe87c5073d");
    const std::array<float,4> positive_zero{0.0F,0.0F,1.0F,-1.0F};
    expect(sha256_bytes(std::as_bytes(std::span(positive_zero)))!=row(0)); // all FP32 bytes, including sign zero
    MtpStepLogitsDigest empty(5);expect(!empty.complete(5));
    MtpStepLogitsDigest gap(5);gap.commit(5,row(0));gap.commit(7,row(1));expect(!gap.complete(8));
    MtpStepLogitsDigest duplicate(5);duplicate.commit(5,row(0));duplicate.commit(5,row(0));expect(!duplicate.complete(6));
    MtpStepLogitsDigest missing(5);missing.commit(5,std::nullopt);missing.commit(6,row(1));expect(!missing.complete(7));
    MtpStepLogitsDigest order(5);order.commit(6,row(1));order.commit(5,row(0));expect(!order.complete(6));
    MtpStepLogitsDigest truncated(5);truncated.commit(5,row(0));expect(!truncated.complete(7));
    MtpStepLogitsDigest shifted(6);shifted.commit(6,row(0));
    expect(shifted.chain()!=truncated.chain()); // position is part of the hash
    MtpStepLogitsDigest overflow(UINT64_MAX);overflow.commit(UINT64_MAX,row(0));expect(!overflow.complete(0));
    // Independent request ledgers, including one emitted prefill prediction only.
    MtpStepLogitsDigest request(100),other(100);request.commit(100,row(0));other.commit(100,row(0));
    expect(request.complete(101) && request.chain()==other.chain());
    request.commit(101,row(1));expect(other.complete(101) && request.chain()!=other.chain());
#ifdef _WIN32
    _putenv_s("STRIX_MTP_STEP_LOGITS_DIGEST","1");
#else
    setenv("STRIX_MTP_STEP_LOGITS_DIGEST","1",1);
#endif
    const std::array<float,4> actual{0,-0.0F,1,-1};
    // Exercise the production serial-ticket capture policy with each independent
    // switch combination, particularly step-only (legacy serial switch OFF).
    for(unsigned mask=0;mask<8;++mask) {
        const MtpSerialLogitsDigestPolicy policy{bool(mask&1U),bool(mask&2U),bool(mask&4U)};
        const auto captured=policy.capture(actual);
        expect(policy.enabled()==bool(mask));
        expect(captured.has_value()==bool(mask));
        if(captured)expect(*captured==sha256_bytes(std::as_bytes(std::span(actual))));
    }
    // The same capture seam serves native and table-backed YaRN serial tickets.
    // These distinct synthetic row streams test publication, not rotary math.
    for(const auto begin:{32767ULL,262143ULL}) {
        constexpr std::size_t vocab=248320;
        const auto produced_row=[&](unsigned r) {
            std::vector<float> values(vocab,-0.0F);
            values.front()=static_cast<float>(begin);values[1]=static_cast<float>(r);
            values.back()=-1.0F-static_cast<float>(r);
            return values;
        };
        MtpStepLogitsDigest mixed(begin),reference(begin);
        const MtpSerialLogitsDigestPolicy step_only{false,false,true};
        std::vector<std::optional<ModelContentIdentity>> retained;
        for(unsigned r=0;r<128;++r) {
            auto produced=produced_row(r);
            const auto expected=sha256_bytes(std::as_bytes(std::span(produced)));
            reference.commit(begin+r,expected);
            // Middle fallback and generation-budget final fallback, with all
            // other rows standing for immutable accepted window captures.
            retained.push_back(r==59 || r==127 ? step_only.capture(produced) : std::optional{expected});
            std::fill(produced.begin(),produced.end(),99); // scratch reused before publication
            mixed.commit(begin+r,retained.back());
            expect(retained.back() && *retained.back()==expected);
            expect(mixed.complete(begin+r+1)); // includes early-stop prefixes
        }
        expect(mixed.count()==128 && mixed.complete(begin+128));
        expect(mixed.chain()==reference.chain());
        // Plain serial observer/terminal-prefill adoption uses the same grammar
        // for native and YaRN rows and must match the ticket ledger chain.
        SerialStepLogitsDiagnostic plain(begin,vocab);
        for(unsigned r=0;r<128;++r) {
            auto produced=produced_row(r);
            plain.row(begin+r,produced);
            std::fill(produced.begin(),produced.end(),99);
        }
        expect(plain.complete(begin+128) && plain.chain()==mixed.chain());
        plain.finish(begin+128);
    }
    for(const auto outputs:{1U,2U,8U}) {
        SerialStepLogitsDiagnostic serial(31,4);MtpStepLogitsDigest mtp(31);
        serial.row(30,actual); // earlier prompt rows do not enter generation coverage
        for(unsigned r=0;r<outputs;++r) {serial.row(31+r,actual);mtp.commit(31+r,sha256_bytes(std::as_bytes(std::span(actual))));}
        expect(serial.complete(31+outputs) && serial.chain()==mtp.chain());
        serial.finish(31+outputs);
    }
    SerialStepLogitsDiagnostic absent(31,4);absent.row(31,std::span<const float>{});expect(!absent.complete(32));
    std::puts("PASS: ordered committed coverage, retained hashes, partial R2/R4/R8 prefixes, missing/duplicate/gap and request isolation");
}

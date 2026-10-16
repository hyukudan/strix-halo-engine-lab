#include "strix/hip_distributed_greedy.hpp"
#include "strix/model_content_hash.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
namespace {
void require_generation(bool condition,const char* message) {
    if(!condition) throw std::invalid_argument(message);
}
}
struct HipDistributedGreedyRequest::Impl final : SerialGreedyStepBackend {
    DistributedGreedyRequestIdentity request;
    std::vector<std::uint32_t> prompt;
    std::shared_ptr<HipDistributedTokenParticipant> participant;
    std::shared_ptr<DistributedControlByteStream> channel;
    std::unique_ptr<FramedDistributedGreedyWire> wire;
    std::unique_ptr<HipDistributedKnownSequence> sequence;
    std::optional<DistributedGreedyMessage> last;
    std::uint32_t node = 0;
    bool used = false, failed = false;

    Impl(HipDistributedSequenceSetup setup,DistributedGreedyRequestIdentity identity,
        std::span<const std::uint32_t> tokens,std::shared_ptr<DistributedControlByteStream> control)
        :request(std::move(identity)),participant(setup.participant),channel(std::move(control)),node(setup.local_node) {
        validate_distributed_greedy_request_identity(request);
        require_generation(participant && channel && node<=1U && setup.streaming &&
            setup.max_target_inputs==request.max_target_inputs &&
            channel.get()!=setup.transport.step_control.get() && channel.get()!=setup.transport.token_control.get(),
            "greedy request requires an actual streaming owner and distinct kind6 channel");
        require_generation(tokens.size()==request.prompt_tokens && tokens.size()<=SIZE_MAX/4U,
            "agreed prompt count/byte extent differs from immutable request");
        std::vector<std::byte> canonical(tokens.size()*4U);
        for(std::size_t i=0;i<tokens.size();++i) {
            require_generation(tokens[i]<request.vocab_size,"agreed prompt token is outside vocabulary");
            for(unsigned byte=0;byte<4U;++byte)
                canonical[i*4U+byte]=static_cast<std::byte>((tokens[i]>>(byte*8U))&255U);
        }
        require_generation(sha256_bytes(canonical)==request.prompt_sha256,
            "actual retained prompt does not match agreed u32le SHA256");
        prompt.assign(tokens.begin(),tokens.end());
        const auto& b=participant->current_binding();
        require_generation(participant->phase()==DistributedTokenPhase::preparing &&
            participant->local_node()==node && b.serial_execution_identity==request.serial_execution_identity &&
            b.kernel_plan_identity==request.kernel_plan_identity && b.placement_plan_identity==request.placement_plan_identity &&
            b.model_content_identity==request.model_content_identity && b.request_id==request.request_id &&
            b.connection_epoch==request.connection_epoch && b.generation==request.initial_generation &&
            participant->binding_identity()==request.initial_binding_identity &&
            b.previous_history_identity==request.initial_history_identity && !b.position && !b.committed_tokens &&
            b.input_token==prompt.front() && b.boundary_dtype_id==request.boundary_dtype_id &&
            setup.model.vocab_size==request.vocab_size && setup.serial.vocab_size==request.vocab_size &&
            request.context_capacity<=setup.serial.max_position_embeddings &&
            b.peers[0].context_capacity==request.context_capacity && b.peers[1].context_capacity==request.context_capacity,
            "greedy request disagrees with actual initial binding/model/stage frontier");
        wire=std::make_unique<FramedDistributedGreedyWire>(*channel,request,node);
        sequence=std::make_unique<HipDistributedKnownSequence>(std::move(setup));
    }
    void fail() noexcept {failed=true;participant->communication_failed();}
    DistributedTokenBinding next_binding(std::uint32_t token) const {
        if(!sequence->completed_inputs()) return participant->current_binding();
        const auto frontier=participant->frontier();
        require_generation(frontier.has_value() && participant->current_binding().generation<UINT64_MAX,
            "greedy next input has no actual distributed committed frontier");
        return frontier->derive_next_binding(token,participant->current_binding().generation+1U);
    }
    void validate_command_binding(const DistributedGreedyMessage& command,const DistributedTokenBinding& b) const {
        require_generation(command.binding_identity==distributed_token_binding_identity(b) &&
            command.generation==b.generation && command.position==b.position && command.committed_tokens==b.committed_tokens &&
            command.history_identity==b.previous_history_identity && command.input_token==b.input_token,
            "greedy command differs from actual next local participant binding");
        if(command.position<prompt.size())
            require_generation(command.input_token==prompt[static_cast<std::size_t>(command.position)],
                "known-prefix command differs from actual agreed immutable prompt");
    }
    std::vector<float> execute(std::uint32_t,std::uint64_t) override {
        throw std::invalid_argument("distributed greedy scalar channel does not transport a full vocabulary");
    }
    std::optional<std::uint32_t> target(std::uint32_t token,std::uint64_t position,bool state_only) {
        require_generation(!failed && node==0U,"only a live coordinator can submit a target");
        try {
            const auto b=next_binding(token);
            auto command=wire->make_message(DistributedGreedyMessageKind::input_command);
            command.generation=b.generation;command.binding_identity=distributed_token_binding_identity(b);
            command.position=position;command.committed_tokens=b.committed_tokens;command.history_identity=b.previous_history_identity;
            command.input_token=token;command.output_intent=state_only?DistributedGreedyOutputIntent::state_only:
                DistributedGreedyOutputIntent::finite_firstmax;
            if(last) {command.previous_binding_identity=last->binding_identity;command.previous_decision_identity=last->decision_identity;}
            validate_command_binding(command,b);
            wire->send(command);
            const auto completed=sequence->execute_input(token);
            require_generation(!completed.terminal_peer && completed.input_token==token,
                "coordinator sequence returned a foreign terminal completion");
            const auto reply=wire->receive();
            validate_distributed_greedy_target_commit(reply,participant->current_binding(),completed.commit);
            last=reply;
            return reply.selected_present?std::optional<std::uint32_t>(reply.selected_token):std::nullopt;
        } catch(...) {fail();throw;}
    }
    std::uint32_t execute_greedy(std::uint32_t token,std::uint64_t position,std::uint64_t vocab) override {
        require_generation(vocab==request.vocab_size,"greedy cursor vocabulary changed");
        const auto selected=target(token,position,false);
        require_generation(selected.has_value(),"greedy target returned no finite selection");return *selected;
    }
    void execute_state(std::uint32_t token,std::uint64_t position) override {(void)target(token,position,true);}
    DistributedGreedyMessage make_finish(const SerialGreedyRequestResult& result,bool cancelled) const {
        auto m=wire->make_message(DistributedGreedyMessageKind::finish);
        if(last) {
            m.generation=last->generation;m.binding_identity=last->binding_identity;m.decision_identity=last->decision_identity;
            m.position=last->position;m.committed_tokens=last->committed_tokens;m.history_identity=last->history_identity;
        } else {
            m.generation=request.initial_generation;m.binding_identity=request.initial_binding_identity;
            m.history_identity=request.initial_history_identity;
        }
        m.emitted_tokens=result.generated_tokens.size();
        // The emitted last token is pending only when not already consumed by
        // a later target whose completion was cancelled before presentation.
        m.pending_present=last && last->selected_present && !result.generated_tokens.empty() &&
            result.generated_tokens.size()==(m.position>=request.prompt_tokens?m.position-request.prompt_tokens+1U:0U);
        if(m.pending_present) m.pending_token=result.generated_tokens.back();
        m.finish_reason=cancelled?DistributedGreedyFinishReason::cancelled:distributed_greedy_finish_reason(result.stop_reason);
        return m;
    }
};
HipDistributedGreedyRequest::HipDistributedGreedyRequest(HipDistributedSequenceSetup setup,
    DistributedGreedyRequestIdentity request,std::span<const std::uint32_t> prompt,
    std::shared_ptr<DistributedControlByteStream> control)
    :impl_(std::make_unique<Impl>(std::move(setup),std::move(request),prompt,std::move(control))) {}
HipDistributedGreedyRequest::~HipDistributedGreedyRequest()=default;
HipDistributedGreedyResult HipDistributedGreedyRequest::execute_coordinator(const std::function<bool()>& cancelled) {
    auto& s=*impl_;require_generation(!s.used && !s.failed && s.node==0U,"greedy coordinator is single-use and role-bound");
    s.used=true;
    try {
        SerialGreedyTokenCursor cursor({s.prompt,s.request.vocab_size,s.request.context_capacity,s.request.max_new_tokens,s.request.eos_token_id});
        while(!cursor.done()) {
            if(cancelled && cancelled()) {cursor.cancel();break;}
            const auto input=cursor.prepare_target(s);
            if(!input) continue;
            std::optional<std::uint32_t> selected;
            try {
                if(input->state_only) s.execute_state(input->token,input->position);
                else selected=s.execute_greedy(input->token,input->position,input->vocab_size);
                if(cancelled && cancelled()) cursor.cancel();
                cursor.accept_target(s,*input,selected);
            } catch(...) {cursor.fail_target();throw;}
        }
        HipDistributedGreedyResult result;result.cancelled=cursor.cancelled();result.generation=cursor.take_result();
        s.wire->send(s.make_finish(result.generation,result.cancelled));
        require_generation(s.wire->receive().kind==DistributedGreedyMessageKind::finished,
            "greedy peer did not confirm the exact finish frontier");
        result.finish_confirmed=true;return result;
    } catch(...) {s.fail();throw;}
}
HipDistributedGreedyTerminalResult HipDistributedGreedyRequest::serve_terminal() {
    auto& s=*impl_;require_generation(!s.used && !s.failed && s.node==1U,"greedy terminal is single-use and role-bound");
    s.used=true;
    try {
        for(;;) {
            const auto command=s.wire->receive();
            if(command.kind==DistributedGreedyMessageKind::finish) {
                auto ack=command;const auto header=s.wire->make_message(DistributedGreedyMessageKind::finished);
                ack.kind=header.kind;ack.sequence=header.sequence;ack.sender_node=header.sender_node;ack.receiver_node=header.receiver_node;
                s.wire->send(ack);
                return {s.sequence->completed_inputs(),command.emitted_tokens,command.pending_present,
                    command.pending_token,command.finish_reason,true};
            }
            require_generation(command.kind==DistributedGreedyMessageKind::input_command,"terminal expected an input or finish");
            const auto b=s.next_binding(command.input_token);s.validate_command_binding(command,b);
            const auto completed=s.sequence->execute_input(command.input_token);
            require_generation(completed.terminal_peer && completed.input_token==command.input_token,
                "terminal sequence returned a foreign coordinator completion");
            const auto logits=s.sequence->last_terminal_logits();
            require_generation(logits.size()==s.request.vocab_size,"terminal completed row is missing or truncated");
            // Recheck the actual retained full row. Strict > preserves the
            // lowest-index finite tie, including +0 versus -0.
            std::uint32_t winner=0;
            for(std::size_t i=0;i<logits.size();++i) {
                require_generation(std::isfinite(logits[i]),"terminal actual complete row is nonfinite");
                if(logits[i]>logits[winner]) winner=static_cast<std::uint32_t>(i);
            }
            require_generation(winner==completed.terminal_firstmax &&
                std::bit_cast<std::uint32_t>(logits[winner])==completed.terminal_firstmax_bits,
                "terminal selected payload differs from actual completed full-row firstmax");
            auto reply=s.wire->make_message(DistributedGreedyMessageKind::target_result);
            reply.generation=b.generation;reply.binding_identity=completed.commit.binding_identity;
            reply.decision_identity=completed.commit.decision_identity;reply.position=completed.commit.next_position;
            reply.committed_tokens=completed.commit.committed_tokens;reply.history_identity=completed.commit.history_identity;
            reply.output_intent=command.output_intent;
            if(command.output_intent==DistributedGreedyOutputIntent::finite_firstmax) {
                reply.selected_present=true;reply.selected_token=winner;reply.selected_logit_bits=completed.terminal_firstmax_bits;
            }
            validate_distributed_greedy_target_commit(reply,s.participant->current_binding(),completed.commit);
            s.wire->send(reply);s.last=reply;
        }
    } catch(...) {s.fail();throw;}
}
} // namespace strix

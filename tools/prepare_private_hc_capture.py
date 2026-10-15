"""Generate one private copied host app. Does not build, run HIP or edit source.

The coordinator separately verifies archive/build/link inputs, compiles this app
object with the original command, and proves all device modules unchanged.
"""
import argparse
import hashlib
import json
import pathlib
import re

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    def pairs(rows):
        out={}
        for key,value in rows:
            if key in out:raise ValueError('duplicate JSON key')
            out[key]=value
        return out
    return json.loads(path.read_text(encoding='utf8'),object_pairs_hook=pairs,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))

def once(source,old,new):
    if source.count(old)!=1:raise ValueError('host source seam changed: '+old[:80])
    return source.replace(old,new)

def vectors(reference,golden):
    prompt=reference['prompt']['token_ids']
    if len(prompt)!=43 or any(type(t)is not int or not 0<=t<248320 for t in prompt):raise ValueError('wrong canonical43 prompt')
    if golden.get('stage')!='finished' or golden.get('all_generated_tokens_equal')is not True:raise ValueError('incomplete golden')
    samples=golden['samples'];ids=samples[0]['generated_tokens']
    if len(ids)!=512 or any(type(t)is not int or not 0<=t<248320 for t in ids)or any(s['generated_tokens']!=ids for s in samples):raise ValueError('invalid immutable512 vector')
    return prompt,prompt+ids[:19],ids[:20]

def render(source,output,reference,golden,commit,archive_sha,model_sha):
    prompt,inputs,expected=vectors(reference,golden)
    if any(not re.fullmatch(r'[0-9a-f]{64}',x)for x in [archive_sha,model_sha])or not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('invalid source/model digest')
    if not output.is_absolute()or any(ord(c)<32 for c in str(output)):raise ValueError('capture output requires absolute safe path')
    array=lambda values:'{'+','.join(map(str,values))+'}'
    literal=lambda value:json.dumps(str(value),ensure_ascii=True)
    config='strix::private_hc_capture::Config{std::filesystem::path('+literal(output)+'),std::filesystem::path(options.model_path),'+array(inputs)+','+array(expected)+','+literal(commit)+','+literal(archive_sha)+','+literal(model_sha)+'}'
    guard='''    if(options.generate_tokens!=20||options.context_tokens!=8192||
        options.execution.mode!=strix::DecodeExecutionMode::eager||options.prefix_cache_roundtrip||
        options.execution.integer_head!=strix::DecodeIntegerHead::none||!options.reference_path.empty()||
        options.coalesce_gap_bytes!=4096||
        options.tokens!=std::vector<std::uint32_t>PROMPT||
        options.residency!=strix::HipModelResidencyMode::registered_file_mapping||
        !options.parallel_hyper_mix||!options.staged_hyper_projections||!options.fused_staged_hyper_mix||
        !options.typed_hyper_projection||!options.prefetch_typed_hyper_rows||!options.prefetch_q8_hyper_output||
        !options.wave_moe_router_topk||options.ranked_moe_router_topk||!options.lane_chain_hyper_output||
        !options.prefetch_q8_qsa_input||!options.prefetch_q8_shared_expert||!options.deferred_moe_status||
        !options.parallel_moe_router||!options.virtual_q8_lm_head||!options.virtual_q8_gdn_input||
        !options.prefetch_q8_gdn_input||!options.virtual_q8_gdn_output||!options.q4k_a8g32_gate_up||
        !options.grouped_exact_q51_down||options.planned_ple_io)
        throw std::invalid_argument("private HC capture requires qualified all5 eager registered43/20");
    strix::DirectLaunchDispatcher native_capture_delegate(launch_plan);
    strix::private_hc_capture::Dispatcher dispatcher(native_capture_delegate,CONFIG,
        original_work_device,experiment_owner,model);'''.replace('PROMPT',array(prompt)).replace('CONFIG',config)
    source='#include "private_hc_capture_dispatcher.hpp"\n'+source
    source=once(source,'    strix::DirectLaunchDispatcher dispatcher(launch_plan);',guard)
    needle='''        finalize_raw(result.generated_tokens, std::string(stop_reason_name(result.stop_reason)));

        double elapsed_total'''
    replacement=needle.replace('\n\n        double elapsed_total','\n        dispatcher.finish(result.generated_tokens,result.steps_executed,result.final_position);\n\n        double elapsed_total')
    return once(source,needle,replacement)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ['source','output','reference','golden']:p.add_argument('--'+name,required=True,type=pathlib.Path)
    for name in ['source-sha','reference-sha','golden-sha','source-commit','archive-sha','model-sha']:p.add_argument('--'+name,required=True)
    a=p.parse_args();source=a.source.resolve(strict=True);reference=a.reference.resolve(strict=True);golden=a.golden.resolve(strict=True)
    for path,digest in [(source,a.source_sha),(reference,a.reference_sha),(golden,a.golden_sha)]:
        if sha(path)!=digest:raise ValueError('pinned input changed: '+path.name)
    out=a.output.absolute()
    if out.exists():raise ValueError('private output must be new')
    header=pathlib.Path(__file__).resolve().parents[1]/'tests/private_hc_capture_dispatcher.hpp'
    text=render(source.read_text(encoding='utf8'),out/'capture',load(reference),load(golden),a.source_commit,a.archive_sha,a.model_sha)
    # All validation precedes mutation. Never touch the original app/build tree.
    out.mkdir();(out/header.name).write_bytes(header.read_bytes());copied=out/'private-hc-capture-main.cpp';copied.write_bytes(text.encode())
    receipt={'purpose':'private165-HC-host-app-source-generation','original_source':str(source),'original_source_sha256':sha(source),'private_source_sha256':sha(copied),'private_header_sha256':sha(out/header.name),'generator_sha256':sha(pathlib.Path(__file__)),'reference_sha256':sha(reference),'golden_sha256':sha(golden),'source_commit_expected_parent_archive_pin':a.source_commit,'archive_sha256_expected_parent_pin':a.archive_sha,'model_sha256_expected_parent_pin':a.model_sha,'capture_directory':str(out/'capture'),'required_host_compile_include_paths':[str(source.parent),str(out)],'scope':'Source generation only. Reuse exact original app compile/link commands, replace only app source/object/output and add the two recorded include paths. No compile/device/module proof or GPU result. Coordinator must bind actual archive/build/binary/libs/model and new Opus review before execution.'}
    (out/'source-generation.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

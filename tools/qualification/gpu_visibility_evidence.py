#!/usr/bin/env python3
"""Check paired, operator-supplied GPU commissioning reports. Never grant readiness.

This verifies consistency/coverage of reports, not their authenticity, the
compiler, physical cache behaviour or kernel hardware qualification. No network,
reset, configuration write, model activation or CI invocation is performed.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys

U64_MAX=(1<<64)-1
MAX_REPORT=4*1024*1024

class EvidenceError(ValueError):
    pass

def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise EvidenceError(f'duplicate JSON field: {key}')
        result[key]=value
    return result

def load_report(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_REPORT:
            raise EvidenceError('report must be a bounded regular file')
        with os.fdopen(fd,'rb',closefd=False) as stream:
            data=stream.read(MAX_REPORT+1)
        if len(data)>MAX_REPORT: raise EvidenceError('report grew beyond limit')
    finally:
        os.close(fd)
    return json.loads(data.decode('utf-8'),object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(EvidenceError('non-finite JSON number')))

def integer(obj,name,low=0,high=U64_MAX):
    value=obj.get(name)
    if type(value) is not int or not low<=value<=high:
        raise EvidenceError(f'invalid integer: {name}')
    return value

def flag(obj,name,expected=True):
    if obj.get(name) is not expected: raise EvidenceError(f'invalid flag: {name}')

def text(obj,name,pattern=None):
    value=obj.get(name)
    if not isinstance(value,str) or not value or len(value)>512 or not value.isascii() or any(ord(c)<32 for c in value):
        raise EvidenceError(f'invalid text: {name}')
    if pattern and not re.fullmatch(pattern,value): raise EvidenceError(f'invalid identity: {name}')
    return value

def _result(value,epoch,sequence,bytes_,nonces):
    if not isinstance(value,dict): raise EvidenceError('missing GPU result')
    if integer(value,'epoch',1)!=epoch or integer(value,'sequence',1)!=sequence:
        raise EvidenceError('GPU result belongs to a different generation')
    nonce=integer(value,'nonce',1)
    if nonce in nonces: raise EvidenceError('replayed GPU result nonce')
    nonces.add(nonce)
    if integer(value,'checked_bytes',1)!=bytes_ or integer(value,'mismatches')!=0 or integer(value,'first_mismatch')!=U64_MAX:
        raise EvidenceError('GPU result mismatch or incomplete byte coverage')
    return (integer(value,'observed_digest'),integer(value,'consumed_digest'))

def validate_report(report):
    if not isinstance(report,dict) or report.get('schema')!='strix-gpu-visibility-v1' or report.get('test_mode')!='native_gpu':
        raise EvidenceError('not a native GPU visibility report')
    flag(report,'production_qualified',False)
    for name in ('campaign','run','kernel','driver_build'):
        text(report,name)
    if report.get('architecture')!='gfx1151': raise EvidenceError('unexpected GPU architecture')
    for name in ('local','peer','gpu_uuid'):
        if text(report,name,r'[0-9a-f]{32}')=='0'*32: raise EvidenceError('zero endpoint/device')
    if report['local']==report['peer']: raise EvidenceError('looped endpoint identity')
    integer(report,'runtime_version',1,2**31-1);integer(report,'gpu_driver_version',1,2**31-1)
    epoch=integer(report,'epoch',1);integer(report,'seed')
    bytes_=integer(report,'bytes',1,4095*4096)
    count=integer(report,'iterations',1,128)
    concurrency=integer(report,'concurrency',1,2)
    if concurrency==2 and (count<2 or count%2): raise EvidenceError('invalid duplex schedule')
    if integer(report,'operations_per_direction',1)!=1:
        raise EvidenceError('this runner does not implement multi-slot engine scheduling')
    if integer(report,'completed')!=count: raise EvidenceError('incomplete run')
    elapsed=integer(report,'elapsed_ns',1)
    integer(report,'producer_delay_ms',0,1000);integer(report,'consumer_delay_ms',0,1000)
    if report.get('cleanup')!='released' or report.get('error')!='': raise EvidenceError('failed or retained run')
    transfers=report.get('transfers')
    if not isinstance(transfers,list) or len(transfers)!=count: raise EvidenceError('missing transfers')
    nonces=set();receives=0
    for index,transfer in enumerate(transfers,1):
        if not isinstance(transfer,dict) or integer(transfer,'sequence',1)!=index:
            raise EvidenceError('non-contiguous/replayed sequence')
        if type(transfer.get('receiver')) is not bool: raise EvidenceError('missing direction')
        if concurrency==2 and index>1:
            if transfer['receiver']==transfers[index-2]['receiver']: raise EvidenceError('duplex roles do not alternate')
        if concurrency==1 and transfer['receiver']!=transfers[0]['receiver']:
            raise EvidenceError('sequential role unexpectedly changes')
        flag(transfer,'completed');flag(transfer,'controller_complete');flag(transfer,'consumer_credit')
        if transfer['receiver']:
            flag(transfer,'raw_gpu_verified');flag(transfer,'copied_gpu_verified')
            flag(transfer,'negative_control_detected');flag(transfer,'receive_reposted')
            flag(transfer,'raw_previously_read',receives>0)
            raw=_result(transfer.get('raw'),epoch,index,bytes_,nonces)
            copied=_result(transfer.get('copied'),epoch,index,bytes_,nonces)
            if raw!=copied: raise EvidenceError('raw/copy consumer diagnostics disagree')
            receives+=1
        else:
            flag(transfer,'producer_completed');flag(transfer,'transport_ack')
    return {'receives':receives,'previous_raw_reads':max(0,receives-1),
        'commissioning_payload_gbps':bytes_*count*8/elapsed}

def pair_reports(left,right):
    a=validate_report(left);b=validate_report(right)
    for name in ('campaign','run','epoch','seed','bytes','iterations','concurrency','operations_per_direction'):
        if left[name]!=right[name]: raise EvidenceError(f'peer metadata differs: {name}')
    if left['local']!=right['peer'] or right['local']!=left['peer'] or left['gpu_uuid']==right['gpu_uuid']:
        raise EvidenceError('reports are not the expected two different GPU endpoints')
    for x,y in zip(left['transfers'],right['transfers']):
        if x['receiver']==y['receiver']: raise EvidenceError('peer transfer directions agree instead of opposing')
    return {'run':left['run'],'epoch':left['epoch'],'paired_transfers':left['iterations'],
        'reports_structurally_consistent':True,'production_qualified':False,
        'gpu_receives':a['receives']+b['receives'],
        'reused_raw_buffers_read':a['previous_raw_reads']+b['previous_raw_reads'],
        'duplex_pairs':left['concurrency']==2,
        'producer_delay_exercised':left['producer_delay_ms']>0 or right['producer_delay_ms']>0,
        'consumer_delay_exercised':left['consumer_delay_ms']>0 or right['consumer_delay_ms']>0,
        'commissioning_payload_gbps':[a['commissioning_payload_gbps'],b['commissioning_payload_gbps']],
        'warning':'Unattested operator reports; structural consistency is not hardware or production qualification.'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',nargs=2,action='append',required=True,metavar=('HOST_A','HOST_B'))
    args=p.parse_args(argv)
    try:
        pairs=[pair_reports(load_report(a),load_report(b)) for a,b in args.pair]
        seen=set()
        for item in pairs:
            key=(item['run'],item['epoch'])
            if key in seen: raise EvidenceError('duplicate run/epoch pair')
            seen.add(key)
        result={'schema':'strix-gpu-visibility-review-v1','production_qualified':False,'pairs':pairs,
            'coverage':{'reuse':any(x['reused_raw_buffers_read'] for x in pairs),
                'duplex':any(x['duplex_pairs'] for x in pairs),
                'producer_delay':any(x['producer_delay_exercised'] for x in pairs),
                'consumer_delay':any(x['consumer_delay_exercised'] for x in pairs)},
            'still_requires':['pinned kernel/module build','independent native report provenance review',
                'multi-slot engine scheduling (more than one operation per direction)',
                'unplug/suspend/reconnect with GPU work','unprovable quarantine safety',
                'property notification traces','physical lane training diagnostics']}
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (OSError,ValueError,TypeError,KeyError) as e:
        print(f'GPU visibility evidence rejected: {e}',file=sys.stderr);return 1

if __name__=='__main__': raise SystemExit(main())

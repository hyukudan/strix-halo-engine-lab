#!/usr/bin/env python3
"""Validate two operator-supplied queue receipts, not their authenticity.

No receipt, test_mode string or passing parser grants production qualification.
The parser never loads a driver, changes device state or runs supplied commands.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

U64_MAX = (1 << 64) - 1
MAX_REPORT_BYTES = 8 * 1024 * 1024
class EvidenceError(ValueError):
    pass

def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)

def integer(value: Any, low: int = 0, high: int = U64_MAX) -> int:
    require(type(value) is int and low <= value <= high, 'invalid bounded integer')
    return value

def text(value: Any, limit: int = 512) -> str:
    require(type(value) is str and 0 < len(value) <= limit and all(32 <= ord(c) <= 126 for c in value), 'invalid identity text')
    return value

def hex_id(value: Any, size: int) -> str:
    require(type(value) is str and re.fullmatch('[0-9a-f]{'+str(size)+'}', value) is not None and int(value,16) != 0, 'invalid canonical hex identity')
    return value

def strict_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON member')
        result[key] = value
    return result

def load_report(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    require(len(data) <= MAX_REPORT_BYTES, 'receipt exceeds size limit')
    def reject_constant(_: str) -> None:
        raise EvidenceError('nonfinite JSON number')
    try:
        result = json.loads(data, object_pairs_hook=strict_object, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise EvidenceError('invalid JSON receipt') from error
    require(type(result) is dict, 'receipt must be one object')
    return result

def validate(report: dict[str, Any]) -> dict[tuple[int,int,int,int], dict[str,Any]]:
    require(type(report) is dict, 'receipt is not an object')
    try:
        require(report['schema'] == 'strix-gpu-queue-visibility-v1' and report['test_mode'] == 'native_gpu', 'wrong receipt type')
        require(report['production_qualified'] is False, 'receipt cannot grant qualification')
        require(report['cleanup'] == 'released' and report['error'] == '', 'failed or retained execution')
        for name in ('campaign','run','kernel','physical_link'):
            text(report[name])
        for name in ('local','peer','gpu_uuid','device_instance'):
            hex_id(report[name],32)
        require(report['local'] != report['peer'], 'identical endpoints')
        require(report['architecture'] == 'gfx1151', 'wrong native architecture')
        build = text(report['driver_build'],128)
        require(len(build)%2==0 and re.fullmatch('[0-9a-f]+',build) is not None,'invalid module build ID')
        for name in ('runtime_version','gpu_driver_version'):
            integer(report[name],1,(1<<31)-1)
        epoch=integer(report['epoch'],1);depth=integer(report['depth'],2,8)
        windows=integer(report['windows'],1,128);size=integer(report['bytes'],1)
        rings=integer(report['ring_frames'],32,4096);directions=integer(report['directions'],1,3)
        recover=integer(report['recovery_every'],0,128)
        integer(report['request_id'],1);integer(report['seed']);integer(report['elapsed_ns'],1)
        count = depth*windows
        require(size <= ((rings-depth)//depth)*4096 and size*count*(2 if directions==3 else 1) <= 128*1024*1024, 'payload geometry/budget differs')
        recovery_count=(windows-1)//recover if recover else 0
        reset_requested=report.get('controller_reset_requested',False)
        network_requested=report.get('network_recovery_requested',False)
        require(type(reset_requested) is bool and type(network_requested) is bool,'invalid recovery flags')
        require(not network_requested or reset_requested,'network recovery without controller reset')
        require(not reset_requested or recovery_count>0,'reset requested without reset boundary')
        require(integer(report.get('controller_resets',0))==(recovery_count if reset_requested else 0),'controller reset count differs')
        require(integer(report['recoveries'])==recovery_count and integer(report['final_epoch'],1)==epoch+recovery_count <= U64_MAX,'recovery epoch schedule differs')
        require(integer(report['completed_windows'])==windows,'incomplete batch windows')
        for d, suffix in ((0,'tx'),(1,'rx')):
            active=bool(directions&(1<<d))
            require(integer(report['completed_'+suffix])==(count if active else 0),'incomplete direction')
            require(integer(report['peak_outstanding_'+suffix])==(depth if active else 0),'wrong software outstanding count')
        canonical='\n'.join(str(report[k]) for k in ('campaign','run','physical_link','request_id','seed','bytes','depth','windows','recovery_every'))+'\nqueue-gpu-v1'
        if reset_requested: canonical+='\ncontroller-reset-v1\n'+str(int(network_requested))
        require(hex_id(report['contract_sha256'],64)==hashlib.sha256(canonical.encode('ascii')).hexdigest(),'campaign contract hash differs')
        transfers=report['transfers']
        require(type(transfers) is list and len(transfers)==count*(2 if directions==3 else 1),'wrong receipt coverage')
        expected=set()
        for w in range(windows):
            group=w//recover if recover else 0
            within=w%recover if recover else w
            for slot in range(depth):
                for d in range(2):
                    if directions&(1<<d):
                        expected.add((epoch+group,within*depth+slot+1,slot,d))
        actual={};seen_handles=set();seen_nonces=set();generation_by_epoch={}
        for transfer in transfers:
            require(type(transfer) is dict,'invalid transfer object')
            ep=integer(transfer['epoch'],1);seq=integer(transfer['sequence'],1)
            slot=integer(transfer['slot'],0,depth-1);direction=integer(transfer['direction'],0,1)
            key=(ep,seq,slot,direction)
            require(key in expected and key not in actual,'duplicate, stale or foreign transfer')
            handle=integer(transfer['handle'],1);generation=integer(transfer['import_generation'],1)
            require(handle not in seen_handles,'reused operation handle');seen_handles.add(handle)
            if ep in generation_by_epoch:
                require(generation_by_epoch[ep]==generation,'import changed inside an epoch')
            generation_by_epoch[ep]=generation
            if direction==0:
                require(transfer['producer_complete'] is True,'missing GPU producer completion')
                require(transfer['terminal_verified'] is False and transfer['payload_verified'] is False and transfer['negative_control'] is False,'sender claims receiver-only proof')
            else:
                require(transfer['producer_complete'] is False,'receiver claims producer proof')
                require(all(transfer[k] is True for k in ('terminal_verified','payload_verified','negative_control')),'missing GPU consumer verification')
                for name in ('raw','copied'):
                    result=transfer[name];require(type(result) is dict,'invalid GPU result')
                    require(integer(result['epoch'],1)==ep and integer(result['sequence'],1)==seq,'stale GPU result')
                    nonce=integer(result['nonce'],1)
                    # Each retained slot owns a separate probe/nonce sequence.
                    require((slot,nonce) not in seen_nonces,'reused result nonce in the same slot');seen_nonces.add((slot,nonce))
                    require(integer(result['checked_bytes'],1)==size and integer(result['mismatches'])==0 and integer(result['first_mismatch'])==U64_MAX,'incomplete or mismatching payload')
                    integer(result['observed_digest']);integer(result['consumed_digest'])
                require(transfer['raw']['observed_digest']==transfer['copied']['observed_digest'] and transfer['raw']['consumed_digest']==transfer['copied']['consumed_digest'],'raw and unpacked GPU payload differ')
            actual[key]=transfer
        require(set(actual)==expected,'missing window/slot coverage')
        generations=[generation_by_epoch[e] for e in sorted(generation_by_epoch)]
        require(all(a<b for a,b in zip(generations,generations[1:])),'recovery reused an import generation')
        return actual
    except KeyError as error:
        raise EvidenceError('missing receipt field: '+str(error)) from error

def pair_reports(a: dict[str,Any], b: dict[str,Any]) -> dict[str,Any]:
    aa=validate(a);bb=validate(b)
    require(a['local']==b['peer'] and a['peer']==b['local'],'peer identities do not reverse')
    require(a['device_instance']!=b['device_instance'],'same per-open instance supplied twice')
    require(b['directions']==((a['directions']&1)<<1)|((a['directions']&2)>>1),'directions are not complementary')
    for name in ('campaign','run','contract_sha256','epoch','final_epoch','depth','windows','bytes','ring_frames','recoveries','recovery_every'):
        require(a[name]==b[name],'paired schedule differs: '+name)
    for flag in ('controller_reset_requested','network_recovery_requested','controller_resets'):
        require(a.get(flag,False if flag.endswith('requested') else 0)==b.get(flag,False if flag.endswith('requested') else 0),'paired reset policy differs')
    for key in aa:
        ep,seq,slot,d=key
        require((ep,seq,slot,1-d) in bb,'missing opposite peer transfer')
    return dict(schema='strix-gpu-queue-evidence-summary-v1',structurally_consistent=True,
                authenticity_verified=False,production_qualified=False,
                campaign=a['campaign'],run=a['run'],depth=a['depth'],windows=a['windows'],
                duplex=a['directions']==3,recoveries=a['recoveries'],
                verified_payload_bytes=a['bytes']*a['depth']*a['windows']*(2 if a['directions']==3 else 1))

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host_a',type=Path);parser.add_argument('host_b',type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(pair_reports(load_report(args.host_a),load_report(args.host_b)),sort_keys=True));return 0
    except (EvidenceError,OSError) as error:
        print(json.dumps({'schema':'strix-gpu-queue-evidence-error-v1','production_qualified':False,'error':str(error)}));return 1
if __name__=='__main__':
    raise SystemExit(main())

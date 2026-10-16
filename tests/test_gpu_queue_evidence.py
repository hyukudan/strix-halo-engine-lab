"""Synthetic reports test parser rejection; they are not hardware receipts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpu_queue_evidence',ROOT/'tools/qualification/gpu_queue_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixtures(duplex=True,recovery=0):
    def report(local,peer,instance,directions):
        r=dict(schema='strix-gpu-queue-visibility-v1',test_mode='native_gpu',production_qualified=False,
               cleanup='released',error='',campaign='fixture',run='fixture',kernel='fixture',physical_link='fixture',
               local=local*32,peer=peer*32,device_instance=instance*32,gpu_uuid=instance*32,architecture='gfx1151',driver_build='aabb',
               runtime_version=1,gpu_driver_version=1,epoch=7,depth=8,windows=3,bytes=4097,ring_frames=32,directions=directions,
               recovery_every=recovery,request_id=9,seed=2,elapsed_ns=1000000,recoveries=2//recovery if recovery else 0,
               completed_windows=3,completed_tx=24 if directions&1 else 0,completed_rx=24 if directions&2 else 0,
               peak_outstanding_tx=8 if directions&1 else 0,peak_outstanding_rx=8 if directions&2 else 0,transfers=[])
        r['final_epoch']=r['epoch']+r['recoveries']
        canonical='\n'.join(str(r[k]) for k in ('campaign','run','physical_link','request_id','seed','bytes','depth','windows','recovery_every'))+'\nqueue-gpu-v1'
        r['contract_sha256']=hashlib.sha256(canonical.encode()).hexdigest();handle=0
        for w in range(3):
            for d in range(2):
                if not directions&(1<<d):continue
                for slot in range(8):
                    handle+=1;epoch=7+(w//recovery if recovery else 0);seq=(w%recovery if recovery else w)*8+slot+1
                    result=lambda n:dict(epoch=epoch,sequence=seq,nonce=n,checked_bytes=4097,mismatches=0,first_mismatch=m.U64_MAX,observed_digest=9,consumed_digest=10)
                    r['transfers'].append(dict(epoch=epoch,sequence=seq,slot=slot,direction=d,handle=handle,import_generation=epoch,
                        producer_complete=d==0,terminal_verified=d==1,payload_verified=d==1,negative_control=d==1,
                        raw=result(w*8+1),copied=result(w*8+2)))
        return r
    return report('1','2','3',3 if duplex else 1),report('2','1','4',3 if duplex else 2)
class QueueEvidence(unittest.TestCase):
    def test_controller_reset(self):
        a,b=fixtures(True,1)
        for r in (a,b):
            r.update(controller_reset_requested=True,network_recovery_requested=True,controller_resets=2)
            canonical='\n'.join(str(r[k]) for k in ('campaign','run','physical_link','request_id','seed','bytes','depth','windows','recovery_every'))+'\nqueue-gpu-v1\ncontroller-reset-v1\n1'
            r['contract_sha256']=hashlib.sha256(canonical.encode()).hexdigest()
        self.assertTrue(m.pair_reports(a,b)['structurally_consistent'])
        b['controller_resets']=1
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_network_without_reset(self):
        a,b=fixtures();b['network_recovery_requested']=True
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_controller_reset_without_schedule(self):
        a,b=fixtures();b['controller_reset_requested']=True
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_duplex(self):
        r=m.pair_reports(*fixtures());self.assertTrue(r['duplex']);self.assertFalse(r['production_qualified']);self.assertFalse(r['authenticity_verified'])
    def test_oneway(self):self.assertFalse(m.pair_reports(*fixtures(False))['duplex'])
    def test_recovery(self):self.assertEqual(m.pair_reports(*fixtures(True,1))['recoveries'],2)
    def test_bad_fields(self):
        for key,value in [('epoch',True),('cleanup','retained'),('test_mode','api_double'),('completed_windows',2),('directions',1),('production_qualified',True),('contract_sha256','f'*64),('seed',7)]:
            with self.subTest(key=key):
                a,b=fixtures();b[key]=value
                with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_missing_slot(self):
        a,b=fixtures();b['transfers'].pop()
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_duplicate(self):
        a,b=fixtures();b['transfers'][9]=copy.deepcopy(b['transfers'][8])
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_coverage(self):
        a,b=fixtures();b['transfers'][8]['raw']['checked_bytes']=4096
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_stale_nonce(self):
        a,b=fixtures();b['transfers'][24]['raw']['nonce']=b['transfers'][8]['raw']['nonce']
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_wrong_gpu_epoch(self):
        a,b=fixtures();b['transfers'][8]['raw']['epoch']=1
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_payload_mismatch(self):
        a,b=fixtures();b['transfers'][8]['raw']['mismatches']=1
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_consumer(self):
        a,b=fixtures();b['transfers'][8]['negative_control']=False
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_peer(self):
        a,b=fixtures();b['peer']='5'*32
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
    def test_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for data in ('{"a":1,"a":2}','{"a":NaN}','[]','{'):
                p.write_text(data)
                with self.assertRaises(m.EvidenceError):m.load_report(p)
    def test_generation(self):
        a,b=fixtures(True,1)
        for t in b['transfers']:t['import_generation']=1
        with self.assertRaises(m.EvidenceError):m.pair_reports(a,b)
if __name__=='__main__':unittest.main()

"""Versioned integer anatomy rebuilt from source counts, never baseline weights."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/conditioning'
VERSION='malecns-integer-v1'
ARTIFACTS=['anatomy-v1.npz','circuit-v1.npz','excluded-boundary-v1.npz','physiology-v1.json','cells.json','mapping.json']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def topology_hash(arrays):
    h=hashlib.sha256()
    for key in ['ids','indptr','indices','counts']:
        a=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()

def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    expected=json.loads((DATA/'manifest.json').read_text()) if (DATA/'manifest.json').exists() else None
    if expected and all((DATA/name).exists() for name in ARTIFACTS):return verify()
    original=json.loads((ROOT/'data/manifest.json').read_text())
    hashes={k:digest(ROOT/'raw'/f'{k}.feather') for k in original['sha256']}
    if hashes!=original['sha256']:raise ValueError('Raw source hashes do not match baseline provenance')
    ann=pd.read_feather(ROOT/'raw/annotations.feather');ann=ann[ann.status=='Traced'].sort_values('bodyId').reset_index(drop=True)
    ids=ann.bodyId.to_numpy(np.int64);n=len(ids);rs=[];cs=[];ws=[]
    with pa.memory_map(str(ROOT/'raw/weights.feather'),'r') as source:
        reader=ipc.open_file(source)
        for k in range(reader.num_record_batches):
            b=reader.get_batch(k);pre=b.column(0).to_numpy();post=b.column(1).to_numpy();count=b.column(2).to_numpy()
            if not np.all(count==count.astype(np.int64)) or np.any(count<0):raise ValueError('Invalid anatomical counts')
            i=np.searchsorted(ids,pre);j=np.searchsorted(ids,post)
            keep=(i<n)&(j<n)&(ids[np.minimum(i,n-1)]==pre)&(ids[np.minimum(j,n-1)]==post)
            rs.append(i[keep].astype(np.int32));cs.append(j[keep].astype(np.int32));ws.append(count[keep].astype(np.int64))
    graph=csr_matrix((np.concatenate(ws),(np.concatenate(rs),np.concatenate(cs))),shape=(n,n),dtype=np.int64)
    graph.sum_duplicates();graph.sort_indices()
    arrays=dict(ids=ids,indptr=graph.indptr.astype(np.int32),indices=graph.indices.astype(np.int32),counts=graph.data)
    np.savez_compressed(DATA/'anatomy-v1.npz',**arrays)
    types=ann.type.fillna('').to_numpy();targets=np.flatnonzero(np.isin(types,['MBON11','MBON12']))
    kc=np.flatnonzero(np.char.startswith(types.astype(str),'KC'))
    connected=np.array([k for k in kc if np.isin(graph.indices[graph.indptr[k]:graph.indptr[k+1]],targets).any()],np.int32)
    selected=np.sort(np.unique(np.concatenate([connected,targets,np.flatnonzero(types=='PPL101')]))).astype(np.int32)
    sub=graph[selected][:,selected].tocsr()
    circuit=dict(ids=ids[selected],indptr=sub.indptr.astype(np.int32),indices=sub.indices.astype(np.int32),counts=sub.data)
    np.savez_compressed(DATA/'circuit-v1.npz',**circuit,global_indices=selected)
    # Preserve every excluded boundary pair explicitly, in global anatomical indices.
    pre=np.repeat(np.arange(n,dtype=np.int32),np.diff(graph.indptr));inside=np.zeros(n,bool);inside[selected]=True
    boundary=inside[pre]^inside[graph.indices]
    np.savez_compressed(DATA/'excluded-boundary-v1.npz',pre=pre[boundary],post=graph.indices[boundary],counts=graph.data[boundary])
    nt=pd.read_feather(ROOT/'raw/neurotransmitters.feather').set_index('body').reindex(ids)
    names=nt.consensus_nt.fillna(nt.predicted_nt).fillna('unknown').astype(str)
    physiology={'scale_mv_per_synapse':.275,'rest_mv':-52,'threshold_mv':-45,'tau_membrane_ms':20,'tau_synapse_ms':5,'delay_ms':1.8,'refractory_ms':2.2,'labels':names.tolist(),'inhibitory_labels':['gaba','glutamate','histamine'],'uncertain_policy':'Baseline sign convention retained; not receptor-specific physiology','target_PPL101_channel':'compartmental modulation only; no generic fast excitation'}
    (DATA/'physiology-v1.json').write_text(json.dumps(physiology))
    records=[]
    for i in selected:
        row=ann.iloc[i];records.append(dict(global_index=int(i),body_id=int(row.bodyId),type=str(row.type),instance=str(row.instance),vfb_id=str(row.vfbId)))
    # Curated type/compartment mapping uses annotation instance labels, not somaSide.
    expected={10704:('MBON11','L'),11402:('MBON11','R'),11327:('PPL101','R'),11900:('PPL101','L'),13728:('MBON12','L'),13768:('MBON12','L'),519368:('MBON12','R'),521086:('MBON12','R')}
    verified=[]
    for body,(kind,side) in expected.items():
        row=ann[ann.bodyId==body]
        if len(row)!=1:raise ValueError('Unverified circuit identity')
        r=row.iloc[0];label=str(r.instance)
        if r.type!=kind or not label.endswith('_'+side) or ('y1ped' if kind!='MBON12' else "y2a'1") not in label:raise ValueError('Unverified compartment annotation')
        verified.append(dict(body_id=body,type=kind,hemisphere=side,instance=label,vfb_id=str(r.vfbId)))
    evidence={'mapping_basis':'MaleCNS v1.0 compartment-qualified instance annotation plus published cell-type mapping; somaSide is not used. No synapse-coordinate localization is claimed.','sources':['https://elifesciences.org/articles/62576','https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/','https://www.virtualflybrain.org/term/mbon11-fbbt_00100246/'],'cells':verified,'granularity':'Directed pairs aggregated across anatomical synapses'}
    (DATA/'mapping.json').write_text(json.dumps(evidence,indent=2));(DATA/'cells.json').write_text(json.dumps(records))
    manifest=dict(version=VERSION,source_hashes=hashes,topology_hash=topology_hash(arrays),circuit_hash=topology_hash(circuit),neurons=n,edges=len(graph.data),synapses=int(graph.data.sum()),circuit_neurons=len(selected),circuit_edges=len(sub.data),excluded_boundary_pairs=int(boundary.sum()),selection='KC inputs of MBON11/12, both target/control MBON populations and PPL101; every induced edge retained',files={name:digest(DATA/name) for name in ARTIFACTS})
    if expected and any(manifest[k]!=expected[k] for k in ['version','source_hashes','topology_hash','circuit_hash']):raise ValueError('Rebuilt anatomy differs from the versioned manifest')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def verify():
    m=json.loads((DATA/'manifest.json').read_text())
    for k,v in m['source_hashes'].items():
        if digest(ROOT/'raw'/f'{k}.feather')!=v:raise ValueError('Source anatomy changed: '+k)
    for k,v in m['files'].items():
        if digest(DATA/k)!=v:raise ValueError('Versioned anatomy/physiology changed: '+k)
    return m
if __name__=='__main__':print(json.dumps(prepare(),indent=2))

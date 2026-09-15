"""Build a traced-neuron MaleCNS graph from the unmodified public Feather tables."""
from pathlib import Path
import argparse, json, hashlib, urllib.request
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[1]
BASE='https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES={'annotations':'body-annotations-male-cns-v1.0-minconf-0.5.feather','weights':'connectome-weights-male-cns-v1.0-minconf-0.5.feather','neurotransmitters':'body-neurotransmitters-male-cns-v1.0.feather'}
def main(cache):
 cache.mkdir(parents=True,exist_ok=True); (ROOT/'data').mkdir(exist_ok=True)
 for key,file in FILES.items():
  p=cache/(key+'.feather')
  if not p.exists():
   print('Downloading',file,flush=True); urllib.request.urlretrieve(BASE+file,p.with_suffix('.part')); p.with_suffix('.part').replace(p)
 ann=pd.read_feather(cache/'annotations.feather'); ann=ann[ann.status=='Traced'].sort_values('bodyId').reset_index(drop=True)
 ids=ann.bodyId.to_numpy(np.int64); n=len(ids); print('Traced neurons',n,flush=True)
 nt=pd.read_feather(cache/'neurotransmitters.feather').set_index('body').reindex(ids)
 nt_names=nt.consensus_nt.fillna(nt.predicted_nt).fillna('unknown').to_numpy()
 signs=np.where(np.isin(nt_names,['gaba','glutamate','histamine']),-1,1).astype(np.float32)
 print('NT labels',dict(zip(*np.unique(nt_names,return_counts=True))),flush=True)
 rows=[]; cols=[]; weights=[]; raw_rows=0
 with pa.memory_map(str(cache/'weights.feather'),'r') as source:
  reader=ipc.open_file(source)
  for batch_id in range(reader.num_record_batches):
   b=reader.get_batch(batch_id); pre=b.column(0).to_numpy();post=b.column(1).to_numpy();w=b.column(2).to_numpy();raw_rows+=len(pre)
   i=np.searchsorted(ids,pre);j=np.searchsorted(ids,post)
   keep=(i<n)&(j<n); safe_i=np.minimum(i,n-1);safe_j=np.minimum(j,n-1);keep&=(ids[safe_i]==pre)&(ids[safe_j]==post)
   rows.append(i[keep].astype(np.int32));cols.append(j[keep].astype(np.int32));weights.append(w[keep].astype(np.float32))
   if batch_id%500==0: print('Read rows',raw_rows,flush=True)
 r=np.concatenate(rows); c=np.concatenate(cols); w=np.concatenate(weights); del rows,cols,weights
 synapses=int(w.sum(dtype=np.float64)); graph=csr_matrix((w*signs[r]*.275,(r,c)),shape=(n,n),dtype=np.float32);graph.sum_duplicates();graph.sort_indices()
 # No edge pruning, row normalization, or learned changes to the biological graph.
 np.savez_compressed(ROOT/'data/connectome.npz',ids=ids,indptr=graph.indptr.astype(np.int32),indices=graph.indices.astype(np.int32),weights=graph.data,signs=signs)
 positions=[];pos_ids=[];pos_kinds=[]
 for i,row in ann.iterrows():
  p=row.somaLocation; kind=0
  if p is None or not isinstance(p,(list,np.ndarray)): p=row.tosomaLocation;kind=1
  if isinstance(p,(list,np.ndarray)) and len(p)==3:
   positions.append(p);pos_ids.append(i);pos_kinds.append(kind)
 positions=np.array(positions,dtype=np.float32); pos_ids=np.array(pos_ids,dtype=np.uint32)
 np.savez_compressed(ROOT/'data/anatomy.npz',positions=positions,indices=pos_ids)
 positions.astype('<f4').tofile(ROOT/'dist/positions.bin');pos_ids.astype('<u4').tofile(ROOT/'dist/position-indices.bin')
 sensory=ann['type'].isin(['L1','L2']) & ann.assignedOlHex1.notna() & ann.assignedOlHex2.notna()
 si=np.flatnonzero(sensory).astype(np.int32); xy=ann.loc[sensory,['assignedOlHex1','assignedOlHex2']].to_numpy(np.float32).copy()
 # Fixed hex-column coordinate mapping to the game image. This artificial retina is an explicit adapter.
 xy-=xy.min(axis=0);xy/=np.maximum(xy.max(axis=0),1)
 readout=np.flatnonzero(ann.superclass.isin(['descending_neuron','vnc_motor','cb_motor'])).astype(np.int32)
 np.savez_compressed(ROOT/'data/interfaces.npz',sensory=si,retina_xy=xy,readout=readout)
 meta={'dataset':'MaleCNS v1.0','neurons':n,'directed_edges':int(graph.nnz),'anatomical_synapses':synapses,'raw_graph_rows':raw_rows,'spatial_neurons':len(pos_ids),'missing_coordinates':n-len(pos_ids),'soma_positions':int((np.array(pos_kinds)==0).sum()),'root_positions':int((np.array(pos_kinds)==1).sum()),'input_neurons':len(si),'readout_neurons':len(readout),'inhibitory_neurons':int((signs<0).sum()),'unknown_nt':int((nt_names=='unknown').sum()),'selection':"status == 'Traced'; all directed connections between these neurons retained",'coordinate_units_nm':8,'sources':{k:BASE+v for k,v in FILES.items()},'license':'CC BY 4.0','model_reference':'https://www.nature.com/articles/s41586-024-07763-9','model_parameters':{'dt_ms':.1,'rest_mv':-52,'threshold_mv':-45,'tau_membrane_ms':20,'tau_synapse_ms':5,'delay_ms':1.8,'refractory_ms':2.2,'weight_mv_per_synapse':.275}}
 meta['sha256']={}
 for key in FILES:
  h=hashlib.sha256()
  with (cache/(key+'.feather')).open('rb') as source:
   for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
  meta['sha256'][key]=h.hexdigest()
 (ROOT/'data/manifest.json').write_text(json.dumps(meta,indent=2));(ROOT/'dist/manifest.json').write_text(json.dumps(meta,indent=2))
 annotations=[{'body_id':int(row.bodyId),'type':str(row['type']) if pd.notna(row['type']) else 'untyped','class':str(row.superclass),'nt':str(nt_names[i])} for i,row in ann.iterrows()]
 (ROOT/'data/neurons.json').write_text(json.dumps(annotations,separators=(',',':')))
 print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'raw');main(p.parse_args().cache)

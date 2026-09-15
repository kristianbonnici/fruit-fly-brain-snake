import {checkedBytes} from './replay.mjs';
export function decodeAdjacency(buffer,index,direction,total,efficacies=null){
  if(buffer.byteLength<24||buffer.byteLength%4)throw Error('Invalid connection buffer.');
  const u=new Uint32Array(buffer),f=new Float32Array(buffer),[version,start,count,edges]=u;
  const header=5+count;
  if(version!==1||!count||start+count>total||index<start||index>=start+count||u.length!==header+edges*3||u[4]!==0||u[4+count]!==edges)throw Error('Invalid connection coverage.');
  for(let i=0;i<count;i++)if(u[4+i]>u[5+i])throw Error('Invalid connection offsets.');
  if(efficacies!==null&&(!(efficacies instanceof Float32Array)||efficacies.length!==edges))throw Error('Invalid learned efficacy coverage.');
  const rows=[];
  for(let i=u[4+index-start];i<u[5+index-start];i++){
    const k=header+3*i,other=u[k],synapses=u[k+1],eff=efficacies===null?f[k+2]:efficacies[i];
    if(other>=total||!synapses||!Number.isFinite(eff)||eff<.049999||eff>4.000001)throw Error('Invalid measured connection.');
    rows.push(direction==='incoming'?[other,index,synapses,eff]:[index,other,synapses,eff]);
  }
  return rows;
}
export function createConnectionLoader(anatomy,commit,fetcher=fetch){
  let generation=0;const cache=new Map();
  const load=async index=>{
    const current=++generation,meta=anatomy.connections;
    try{
    const both=await Promise.all(['incoming','outgoing'].map(async direction=>{
      const entry=meta[direction][Math.floor(index/meta.block_size)];
      if(!entry||!/^data\/full-v90\/edges\/(incoming|outgoing)-\d+\.bin$/.test(entry.file))throw Error('Invalid connection shard.');
      let bytes=cache.get(entry.file);
      if(!bytes){bytes=await checkedBytes(await fetcher(entry.file),entry.sha256);cache.set(entry.file,bytes);if(cache.size>4)cache.delete(cache.keys().next().value);}
      let efficacy=null;
      if(entry.efficacy_file){
        if(!/^data\/full-selected\/efficacy\/(incoming|outgoing)-\d+\.f32$/.test(entry.efficacy_file))throw Error('Invalid efficacy shard.');
        let overlay=cache.get(entry.efficacy_file);
        if(!overlay){overlay=await checkedBytes(await fetcher(entry.efficacy_file),entry.efficacy_sha256);cache.set(entry.efficacy_file,overlay);if(cache.size>4)cache.delete(cache.keys().next().value);}
        efficacy=new Float32Array(overlay);
      }
      return decodeAdjacency(bytes,index,direction,anatomy.nodes.length,efficacy);
    }));
    const rows=[...both[0],...both[1].filter(e=>e[0]!==e[1])].sort((a,b)=>b[2]-a[2]||a[0]-b[0]||a[1]-b[1]);
    if(current!==generation)return false;commit(rows);return true;
    }catch(error){if(current!==generation)return false;throw error;}
  };
  load.cancel=()=>{generation++;cache.clear();};return load;
}

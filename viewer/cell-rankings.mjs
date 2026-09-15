// Uniform 100 ms decision endpoints, excluding the synthetic initial zero state.
export async function recordingStats(activity, cells, isCurrent=()=>true, yieldTask=()=>new Promise(resolve=>setTimeout(resolve,0))) {
  if(!Number.isInteger(cells)||cells<=0||!activity.length||activity.length%cells)throw Error('Invalid ranking activity shape');
  const magnitude=new Float64Array(cells),mean=new Float64Array(cells),variance=new Float64Array(cells);
  const frames=activity.length/cells;
  for(let f=0;f<frames;f++){
    if(!isCurrent())return null;
    const offset=f*cells;
    for(let i=0;i<cells;i++){
      const value=activity[offset+i];
      if(!Number.isFinite(value))throw Error('Invalid ranking activity value');
      magnitude[i]+=Math.abs(value);
      const delta=value-mean[i];mean[i]+=delta/(f+1);variance[i]+=delta*(value-mean[i]);
    }
    if(f%8===7)await yieldTask();
  }
  if(!isCurrent())return null;
  for(let i=0;i<cells;i++){magnitude[i]/=frames;variance[i]=Math.sqrt(Math.max(0,variance[i]/frames));}
  return {active:magnitude,variable:variance};
}
// Keep only the best results, without sorting 165k cells on every replay sample.
export function topCells(candidates, scores, limit=200, absolute=false){
  const heap=[],score=i=>absolute?Math.abs(scores[i]):scores[i];
  const better=(a,b)=>score(a)>score(b)||(score(a)===score(b)&&a<b);
  for(const i of candidates){
    if(heap.length<limit){
      heap.push(i);let k=heap.length-1;
      while(k){const p=(k-1)>>1;if(!better(heap[p],heap[k]))break;[heap[p],heap[k]]=[heap[k],heap[p]];k=p;}
    }else if(better(i,heap[0])){
      heap[0]=i;let k=0;
      for(;;){let child=k*2+1;if(child>=heap.length)break;if(child+1<heap.length&&better(heap[child],heap[child+1]))child++;if(!better(heap[k],heap[child]))break;[heap[k],heap[child]]=[heap[child],heap[k]];k=child;}
    }
  }
  return heap.sort((a,b)=>score(b)-score(a)||a-b);
}

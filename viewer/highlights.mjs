// Display membership only. Signal buffers and recording state are read-only.
export function highlightMask(nodes,populations,directions,values,output=new Uint8Array(nodes.length)) {
  if(!populations.size&&!directions.size)return null;
  for(let i=0;i<nodes.length;i++){
    const populationMatches=!populations.size||populations.has(nodes[i].population);
    const directionMatches=!directions.size||(values[i]>0&&directions.has('rising'))||(values[i]<0&&directions.has('falling'));
    output[i]=populationMatches&&directionMatches?1:0;
  }
  return output;
}

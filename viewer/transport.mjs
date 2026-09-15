// Lossless delivery only. The replay loader still checks every original SHA-256.
const hex=bytes=>Array.from(new Uint8Array(bytes),x=>x.toString(16).padStart(2,'0')).join('');
export function createPackedFetch(fetcher=globalThis.fetch.bind(globalThis),progress=()=>{}){
  let indexPromise;
  return async (path,options={})=>{
    indexPromise??=fetcher('data/pack-index.json').then(async r=>{if(!r.ok)throw Error('Could not load the showcase catalog.');return r.json();});
    const index=await indexPromise;options.signal?.throwIfAborted();
    if(index.format!=='snake-showcase-gzip-v1')throw Error('Unsupported showcase data.');
    const entry=index.files[path];if(!entry)return fetcher(path,options);
    if(!Number.isSafeInteger(entry.bytes)||entry.bytes<0||entry.bytes>512*1024**2||!Array.isArray(entry.parts))throw Error('Invalid showcase buffer.');
    const output=new Uint8Array(entry.bytes);let next=0,completed=0,offset=0;
    const parts=entry.parts.map(part=>{const start=offset;offset+=part.bytes;return {...part,start};});
    if(offset!==entry.bytes)throw Error('Incomplete showcase buffer.');
    async function worker(){
      while(next<parts.length){
        const part=parts[next++];options.signal?.throwIfAborted();
        if(!/^data\/packed\/[a-f0-9]{64}\.gz$/.test(part.file))throw Error('Invalid showcase chunk path.');
        const response=await fetcher(part.file,options);if(!response.ok)throw Error('Could not download this replay. Please try again.');
        const zipped=await response.arrayBuffer();
        if(zipped.byteLength!==part.compressed_bytes||hex(await crypto.subtle.digest('SHA-256',zipped))!==part.sha256)throw Error('Showcase download failed its integrity check.');
        const decoded=await new Response(new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        options.signal?.throwIfAborted();
        if(decoded.byteLength!==part.bytes)throw Error('Incomplete showcase activity.');
        output.set(new Uint8Array(decoded),part.start);completed+=part.bytes;progress(path,completed/entry.bytes);
      }
    }
    await Promise.all([worker(),worker()]);options.signal?.throwIfAborted();
    // Avoid Response's extra full-size copy for large neural buffers.
    return {ok:true,status:200,arrayBuffer:async()=>output.buffer,json:async()=>JSON.parse(new TextDecoder().decode(output))};
  };
}
export const siteFetch=createPackedFetch(undefined,(path,fraction)=>{
  if(path.startsWith('data/games/')&&path.endsWith('.f32')){
    const status=document.querySelector('#status');
    if(status)status.textContent=`Loading ${path.includes('.10ms.')?'detailed ':''}recorded activity · ${Math.round(fraction*100)}%`;
  }
});

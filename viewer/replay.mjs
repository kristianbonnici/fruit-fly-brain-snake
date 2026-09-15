export const FORMAT = 'snake-viewer-v1';
export class Replay {
  constructor(record, activity, substeps=null) {
    this.record=record; this.activity=activity; this.substeps=substeps;
    this.substepMoves=substeps?substeps.length/(10*record.activity_shape[1]):0;
    this.fineCount=this.substepMoves*10; this.totalSamples=this.fineCount+this.length-this.substepMoves;
    this.sampleIndex=0; this.remainder=0; this.playing=false;
    this.zero=new Float32Array(record.activity_shape[1]);
    this.changeBuffer=new Float32Array(record.activity_shape[1]);this.changeSample=-1;
  }
  get length() { return this.record.frames.length; }
  get sampleMs() { return this.fineCount && this.sampleIndex<=this.fineCount?10:this.record.tick_ms; }
  get index() { return this.sampleIndex<=this.fineCount?Math.floor(this.sampleIndex/10):this.substepMoves+this.sampleIndex-this.fineCount; }
  get phaseMs() { return this.sampleIndex<this.fineCount?(this.sampleIndex%10)*10:0; }
  get frame() { return this.index ? this.record.frames[this.index-1] : null; }
  get inputFrame() { return this.phaseMs ? this.record.frames[this.index] : this.frame; }
  get inputIndex() { return this.index+(this.phaseMs?1:0); }
  get board() { return this.frame?.after ?? this.record.initial; }
  sampleTime(index) { return index<=this.fineCount?index*10:(this.substepMoves+index-this.fineCount)*100; }
  get time() { return this.sampleTime(this.sampleIndex); }
  valuesAt(index) {
    const n=this.record.activity_shape[1];if(index===0)return this.zero;
    return index<=this.fineCount?this.substeps.subarray((index-1)*n,index*n):this.activity.subarray((this.substepMoves+index-this.fineCount-1)*n,(this.substepMoves+index-this.fineCount)*n);
  }
  get values() { return this.valuesAt(this.sampleIndex); }
  // Compare exact recorded samples 100 ms apart across both sampling regimes.
  // Before 100 ms, use the known initial zero state and expose the shorter span.
  get changeStartIndex() { return Math.max(0,this.sampleIndex-(this.sampleIndex<=this.fineCount?10:1)); }
  get changeSpanMs() { return this.time-this.sampleTime(this.changeStartIndex); }
  get changes() {
    if(this.changeSample!==this.sampleIndex){
      const now=this.values,before=this.valuesAt(this.changeStartIndex);
      for(let i=0;i<now.length;i++)this.changeBuffer[i]=now[i]-before[i];
      this.changeSample=this.sampleIndex;
    }
    return this.changeBuffer;
  }
  seek(index) { const i=Math.max(0,Math.min(this.length,Math.trunc(index)));this.sampleIndex=Math.min(i,this.substepMoves)*10+Math.max(0,i-this.substepMoves);this.remainder=0;if(i===this.length)this.playing=false; }
  advance(elapsed,speed) {
    if(!this.playing || !Number.isFinite(elapsed) || elapsed<0)return false;
    const desired=this.time+this.remainder+elapsed*speed, fineMs=this.substepMoves*100,old=this.sampleIndex;
    this.sampleIndex=Math.min(this.totalSamples,desired<=fineMs?Math.floor((desired+1e-8)/10):this.fineCount+Math.floor((desired-fineMs+1e-8)/100));
    this.remainder=desired-this.time;
    if(this.index===this.length){this.playing=false;this.remainder=0;}
    return old!==this.sampleIndex;
  }
}
export function validateSubsteps(meta,buffer,record,activity) {
  const [moves,n]=record.activity_shape, captured=meta.shape?.[0];
  const prefix=meta.format==='snake-viewer-substeps-v2' && meta.coverage==='contiguous_prefix' && Number.isInteger(captured) && captured>0 && captured<=moves;
  if((meta.format!=='snake-viewer-substeps-v1' && !prefix) || meta.game_id!==record.id
     || meta.recording_identity!==record.recording_identity || meta.source_sha256!==record.source_sha256
     || meta.checkpoint_sha256!==record.checkpoint_sha256 || meta.anatomy_identity!==record.anatomy_identity
     || meta.sample_ms!==10 || JSON.stringify(meta.shape)!==JSON.stringify([prefix?captured:moves,10,n])
     || meta.file!==`data/games/${record.id}.10ms.f32` || buffer.byteLength!==captured*10*n*4
     || !['exact_endpoints','exact_logits','exact_actions','checkpoint_unchanged'].every(k=>meta.validation?.[k]===true))
    throw Error('Incompatible neural substep capture.');
  const samples=new Float32Array(buffer);
  if(samples.some(v=>!Number.isFinite(v)||Math.abs(v)>1.000001))throw Error('Invalid neural substeps.');
  for(let i=0;i<captured;i++)for(let j=0;j<n;j++)
    if(samples[(i*10+9)*n+j]!==activity[i*n+j])throw Error('Neural substeps do not match recorded decisions.');
  return samples;
}
export async function checkedBytes(response, expected) {
  if(!response.ok)throw Error(`Could not load recording (${response.status}).`);
  const bytes=await response.arrayBuffer();
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==expected)throw Error('Data integrity check failed. Please re-export the frozen recordings.');
  return bytes;
}
export function validateRecord(record,buffer,manifest,entry) {
  const shape=record.activity_shape;
  if(record.format!==FORMAT || record.id!==entry.id || record.checkpoint_sha256!==manifest.checkpoint_sha256 || record.anatomy_identity!==manifest.anatomy_identity
     || !Array.isArray(shape) || shape.length!==2 || shape[0]!==entry.moves || shape[1]!==manifest.coverage.simulated
     || record.tick_ms!==100 || record.frames.length!==shape[0] || buffer.byteLength!==shape[0]*shape[1]*4
     || record.activity_file!==`data/games/${entry.id}.f32`) throw Error('Incompatible or incomplete recording.');
  const activity=new Float32Array(buffer);
  if(activity.some(v=>!Number.isFinite(v)||Math.abs(v)>1.000001))throw Error('Invalid graded activity values.');
  record.frames.forEach((f,i)=>{
    if(f.observation_ms!==i*100 || f.endpoint_ms!==(i+1)*100 || f.before.moves!==i || f.after.moves!==i+1
       || f.sensors.length!==5 || f.logits.length!==3 || ![0,1,2].includes(f.action)
       || f.action!==f.logits.indexOf(Math.max(...f.logits)) || (i<record.frames.length-1 && f.after.done)) throw Error('Invalid decision alignment.');
  });
  if(manifest.signals?.input==='image'){
    if(record.input_kind!=='image'||record.frames.some(f=>!Array.isArray(f.image)||f.image.length!==256||f.image.some(v=>!Number.isFinite(v)||v<0||v>1)||f.sensory_drive?.length!==256||f.sensory_drive.some(v=>!Number.isFinite(v))))throw Error('Invalid recorded image inputs.');
  }
  if(!record.frames.at(-1)?.after.done)throw Error('A complete terminal game is required.');
  return {record,activity};
}
export function createLoader(manifest,commit,fetcher=fetch) {
  let generation=0,controller;
  const load=async entry=>{
    const current=++generation; controller?.abort(); controller=new AbortController(); const signal=controller.signal;
    try {
      const jsonBytes=await checkedBytes(await fetcher(entry.file,{signal}),entry.sha256);
      const record=JSON.parse(new TextDecoder().decode(jsonBytes));
      if(record.activity_file!==`data/games/${entry.id}.f32`)throw Error('Unexpected recording data path.');
      const bytes=await checkedBytes(await fetcher(record.activity_file,{signal}),record.activity_sha256);
      const loaded=validateRecord(record,bytes,manifest,entry);
      if(entry.substeps){
        if(entry.substeps.file!==`data/games/${entry.id}.10ms.json`)throw Error('Unexpected substep metadata path.');
        const metadataBytes=await checkedBytes(await fetcher(entry.substeps.file,{signal}),entry.substeps.sha256);
        const meta=JSON.parse(new TextDecoder().decode(metadataBytes));
        if(meta.file!==`data/games/${entry.id}.10ms.f32`)throw Error('Unexpected substep data path.');
        const substepBytes=await checkedBytes(await fetcher(meta.file,{signal}),meta.sha256);
        loaded.substeps=validateSubsteps(meta,substepBytes,record,loaded.activity);
      }
      if(current!==generation)return false;
      commit(loaded); return true;
    } catch(error) { if(current!==generation)return false; throw error; }
  };
  load.cancel=()=>{generation++;controller?.abort();};
  return load;
}
export function sensoryDrive(anatomy,sensors,slot) {
  return 3*Math.tanh(anatomy.encoder_bias[slot]+anatomy.encoder_weights[slot].reduce((sum,w,i)=>sum+w*sensors[i],0));
}

// A display transfer curve only: zero remains zero; source values stay unchanged.
export function activityIntensity(value,contrast=2.5,scale='linear',range=1) {
  const magnitude=Math.min(1,Math.abs(value)/range);
  const scaled=scale==='log'?Math.log1p(magnitude*1e6)/Math.log1p(1e6):magnitude;
  return Math.pow(scaled,1/Math.max(1,Math.min(4,contrast)));
}

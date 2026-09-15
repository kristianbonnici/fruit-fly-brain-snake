import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import {Replay,FORMAT,validateRecord,createLoader,sensoryDrive} from '../viewer/replay.mjs';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
function fixture(id='game-28'){
  const state=i=>({moves:i,score:i===2?1:0,done:i===3,body:i===2?[[1,0],[0,0],[0,1],[0,2]]:[[0,0],[0,1],[0,2]],food:[3,3]});
  const frames=Array.from({length:3},(_,i)=>({before:state(i),after:state(i+1),observation_ms:i*100,endpoint_ms:(i+1)*100,sensors:[.5,0,1,0,0],action:1,logits:[0,2,0]}));
  const buffer=new Float32Array([.1,.2,.3,.4,.5,.6]).buffer;
  const record={format:FORMAT,id,checkpoint_sha256:'policy',anatomy_identity:'anatomy',tick_ms:100,initial:state(0),frames,activity_shape:[3,2],activity_file:`data/games/${id}.f32`,activity_sha256:sha(Buffer.from(buffer))};
  const bytes=Buffer.from(JSON.stringify(record));const entry={id,moves:3,file:`data/games/${id}.json`,sha256:sha(bytes)};
  const manifest={checkpoint_sha256:'policy',anatomy_identity:'anatomy',coverage:{simulated:2}};
  return {record,buffer,entry,manifest,bytes};
}
test('shared decision clock holds endpoint samples, clamps seeks, stops at terminal',()=>{
  const {record,buffer}=fixture(),replay=new Replay(record,new Float32Array(buffer));
  assert.equal(replay.frame,null);assert.equal(replay.board.moves,0);assert.deepEqual([...replay.values],[0,0]);
  replay.playing=true;assert.equal(replay.advance(99,1),false);assert.equal(replay.index,0);
  assert.equal(replay.advance(1,1),true);assert.equal(replay.index,1);assert.equal(replay.frame.before.moves,0);assert.equal(replay.board.moves,1);
  assert.ok(Math.abs(replay.values[0]-.1)<1e-6);replay.advance(10,1);assert.equal(replay.index,1);
  replay.seek(2);assert.equal(replay.board.score,1);assert.equal(replay.frame.endpoint_ms,200);
  replay.seek(-12);assert.equal(replay.index,0);replay.playing=true;replay.advance(3000,.1);assert.equal(replay.index,3);assert.equal(replay.playing,false);assert.equal(replay.board.done,true);
  replay.seek(999);assert.equal(replay.index,3);assert.equal(replay.time,300);
});
test('buffer shape, identity, values and timestamp mismatches reject',()=>{
  const {record,buffer,manifest,entry}=fixture();assert.equal(validateRecord(record,buffer,manifest,entry).activity.length,6);
  assert.throws(()=>validateRecord(record,buffer.slice(0,4),manifest,entry));
  assert.throws(()=>validateRecord({...record,checkpoint_sha256:'wrong'},buffer,manifest,entry));
  const bad=new Float32Array(buffer.slice(0));bad[0]=NaN;assert.throws(()=>validateRecord(record,bad.buffer,manifest,entry));
  const corrupt=structuredClone(record);corrupt.frames[1].endpoint_ms=100;assert.throws(()=>validateRecord(corrupt,buffer,manifest,entry));
});
test('late older recording cannot replace a newer selection even if fetch ignores abort',async()=>{
  const a=fixture('game-1'),b=fixture('game-2');let release;const gate=new Promise(resolve=>release=resolve);let first=true;const committed=[];
  const fetcher=async path=>{const f=path.includes('game-1')?a:b;if(path.endsWith('.json')&&first){first=false;await gate;}return new Response(path.endsWith('.json')?f.bytes:f.buffer);};
  const load=createLoader(a.manifest,data=>committed.push(data.record.id),fetcher);
  const old=load(a.entry);assert.equal(await load(b.entry),true);release();assert.equal(await old,false);assert.deepEqual(committed,['game-2']);
});
test('corrupted transport bytes do not commit a partial recording',async()=>{
  const f=fixture();let count=0;const load=createLoader(f.manifest,()=>count++,async path=>new Response(path.endsWith('.json')?f.bytes:new Uint8Array(24)));
  await assert.rejects(load(f.entry),/integrity/);assert.equal(count,0);
});
test('derived projection uses actual sensor weights and bias',()=>{
  const a={encoder_weights:[[1,2,3,4,5]],encoder_bias:[.25]};assert.equal(sensoryDrive(a,[1,0,0,0,0],0),3*Math.tanh(1.25));
});

function substepFixture(id='game-28') {
  const f=fixture(id),samples=new Float32Array(3*10*2);
  for(let i=0;i<3;i++)for(let k=0;k<10;k++)for(let n=0;n<2;n++)samples[(i*10+k)*2+n]=new Float32Array(f.buffer)[i*2+n]*(k+1)/10;
  const meta={format:'snake-viewer-substeps-v1',game_id:id,recording_identity:f.record.recording_identity,source_sha256:f.record.source_sha256,checkpoint_sha256:'policy',anatomy_identity:'anatomy',sample_ms:10,shape:[3,10,2],file:`data/games/${id}.10ms.f32`,sha256:sha(Buffer.from(samples.buffer)),validation:{exact_endpoints:true,exact_logits:true,exact_actions:true,checkpoint_unchanged:true}};
  const metadata=Buffer.from(JSON.stringify(meta));f.entry.substeps={file:`data/games/${id}.10ms.json`,sha256:sha(metadata)};
  return {...f,samples,meta,metadata};
}
const {validateSubsteps,activityIntensity}=await import('../viewer/replay.mjs');
test('10 ms samples advance neural state and input while the board waits for the action',()=>{
  const f=substepFixture(),r=new Replay(f.record,new Float32Array(f.buffer),f.samples);
  r.playing=true;r.advance(9,1);assert.equal(r.sampleIndex,0);assert.equal(r.inputFrame,null);
  r.advance(1,1);assert.equal(r.sampleIndex,1);assert.equal(r.index,0);assert.equal(r.board.moves,0);assert.equal(r.frame,null);assert.equal(r.inputFrame,f.record.frames[0]);assert.equal(r.inputIndex,1);assert.equal(r.phaseMs,10);
  r.advance(90,1);assert.equal(r.index,1);assert.equal(r.board.moves,1);assert.equal(r.phaseMs,0);assert.equal(r.inputFrame,f.record.frames[0]);assert.equal(r.values[0],new Float32Array(f.buffer)[0]);
  r.advance(10,1);assert.equal(r.board.moves,1);assert.equal(r.inputFrame,f.record.frames[1]);assert.equal(r.frame,f.record.frames[0]);
  r.seek(2);assert.equal(r.sampleIndex,20);assert.equal(r.time,200);assert.equal(r.board.score,1);assert.equal(r.inputFrame,f.record.frames[1]);
  r.playing=true;r.advance(1000,1);assert.equal(r.sampleIndex,30);assert.equal(r.board.done,true);assert.equal(r.inputFrame,f.record.frames[2]);assert.equal(r.playing,false);
  r.seek(0);assert.deepEqual([...r.values],[0,0]);assert.equal(r.inputFrame,null);
});
test('substeps reject mismatched endpoints, identity, shape and nonfinite values',()=>{
  const f=substepFixture(),endpoints=new Float32Array(f.buffer);
  assert.equal(validateSubsteps(f.meta,f.samples.buffer,f.record,endpoints).length,60);
  assert.throws(()=>validateSubsteps({...f.meta,game_id:'game-1'},f.samples.buffer,f.record,endpoints));
  assert.throws(()=>validateSubsteps({...f.meta,sample_ms:20},f.samples.buffer,f.record,endpoints));
  const bad=f.samples.slice();bad[18]=.9;assert.throws(()=>validateSubsteps(f.meta,bad.buffer,f.record,endpoints),/match/);
  bad[18]=endpoints[0];bad[0]=NaN;assert.throws(()=>validateSubsteps(f.meta,bad.buffer,f.record,endpoints),/Invalid/);
});
test('switching games rejects a late in-flight 10 ms capture',async()=>{
  const a=substepFixture('game-1'),b=substepFixture('game-2');let release,started;const gate=new Promise(r=>release=r),pending=new Promise(r=>started=r);const commits=[];
  const fetcher=async path=>{const f=path.includes('game-1')?a:b;if(path==='data/games/game-1.10ms.f32'){started();await gate;}
    return new Response(path.endsWith('.10ms.json')?f.metadata:path.endsWith('.10ms.f32')?f.samples.buffer:path.endsWith('.json')?f.bytes:f.buffer);};
  const load=createLoader(a.manifest,v=>commits.push([v.record.id,v.substeps.length]),fetcher);
  const old=load(a.entry);await pending;await load(b.entry);release();assert.equal(await old,false);assert.deepEqual(commits,[['game-2',60]]);
});
test('contrast reveals low activity without changing signed measurements or inventing zero activity',()=>{
  assert.equal(activityIntensity(0,4),0);assert.equal(activityIntensity(1,4),1);assert.equal(activityIntensity(-.04,1),.04);
  assert.equal(activityIntensity(-.04,2),activityIntensity(.04,2));assert.ok(activityIntensity(.04,2)>.04);
  const f=substepFixture(),r=new Replay(f.record,new Float32Array(f.buffer),f.samples);r.seek(1);const before=r.values.slice();
  for(const v of r.values)activityIntensity(v,4);assert.deepEqual(r.values,before);assert.equal(r.index,1);
});

test('partial full-network substeps transition to held endpoints without adding samples',()=>{
  const f=substepFixture(),prefix=f.samples.slice(0,20),r=new Replay(f.record,new Float32Array(f.buffer),prefix);
  const meta={...f.meta,format:'snake-viewer-substeps-v2',coverage:'contiguous_prefix',shape:[1,10,2]};
  assert.equal(validateSubsteps(meta,prefix.buffer,f.record,new Float32Array(f.buffer)).length,20);
  r.playing=true;r.advance(100,1);assert.equal(r.index,1);assert.equal(r.sampleIndex,10);assert.equal(r.time,100);
  const held=r.values.slice();r.advance(99,1);assert.equal(r.index,1);assert.deepEqual(r.values,held);
  r.advance(1,1);assert.equal(r.index,2);assert.equal(r.sampleIndex,11);assert.equal(r.sampleMs,100);assert.equal(r.time,200);assert.equal(r.board.score,1);
  r.advance(100,1);assert.equal(r.sampleIndex,12);assert.equal(r.index,3);assert.equal(r.board.done,true);assert.equal(r.playing,false);
  assert.equal(r.valuesAt(10)[0],new Float32Array(f.buffer)[0]);assert.equal(r.valuesAt(11)[0],new Float32Array(f.buffer)[2]);
  r.seek(1);assert.equal(r.sampleIndex,10);r.seek(2);assert.equal(r.time,200);r.seek(0);assert.deepEqual(r.values,r.zero);
  assert.throws(()=>validateSubsteps({...meta,shape:[4,10,2]},prefix.buffer,f.record,new Float32Array(f.buffer)));
});
test('changing collections invalidates a pending recording even when transport ignores abort',async()=>{
  const f=fixture();let started,release;const pending=new Promise(r=>started=r),gate=new Promise(r=>release=r);const commits=[];
  const load=createLoader(f.manifest,v=>commits.push(v),async path=>{if(path.endsWith('.f32')){started();await gate;}return new Response(path.endsWith('.json')?f.bytes:f.buffer);});
  const promise=load(f.entry);await pending;load.cancel();release();assert.equal(await promise,false);assert.equal(commits.length,0);
});
test('image collection requires actual finite pixel input and recorded sensory drive',()=>{
  const f=fixture();f.manifest.signals={input:'image'};
  assert.throws(()=>validateRecord(f.record,f.buffer,f.manifest,f.entry));
  f.record.input_kind='image';f.record.frames.forEach(frame=>{frame.image=Array(256).fill(.5);frame.sensory_drive=Array(256).fill(0);});
  assert.equal(validateRecord(f.record,f.buffer,f.manifest,f.entry).activity.length,6);
  f.record.frames[0].image[0]=2;assert.throws(()=>validateRecord(f.record,f.buffer,f.manifest,f.entry));
});

test('fixed logarithmic activity scale reveals small magnitudes without inventing values',()=>{
  const samples=[0,1e-8,1e-6,1e-4,.01,.1,1];
  const scaled=samples.map(v=>activityIntensity(v,2.5,'log'));
  assert.equal(scaled[0],0);assert.equal(scaled.at(-1),1);
  for(let i=1;i<scaled.length;i++)assert.ok(scaled[i]>scaled[i-1]);
  assert.equal(activityIntensity(-.00009,4,'log'),activityIntensity(.00009,4,'log'));
  assert.ok(activityIntensity(.00009,4,'log')>7*activityIntensity(.00009,4,'linear'));
  const f=substepFixture(),r=new Replay(f.record,new Float32Array(f.buffer),f.samples);r.seek(2);const before=r.values.slice();
  for(const scale of ['linear','log'])for(const v of r.values)activityIntensity(v,4,scale);
  assert.deepEqual(r.values,before);assert.equal(r.index,2);
});

test('change view uses a fixed 100 ms interval across fine and coarse samples, including seeks',()=>{
  const f=substepFixture(),raw=new Float32Array(f.buffer),original=raw.slice(),fine=f.samples.slice(),r=new Replay(f.record,raw,f.samples.slice(0,40));
  assert.equal(r.changeSpanMs,0);assert.deepEqual([...r.changes],[0,0]);
  r.playing=true;r.advance(50,1);
  assert.equal(r.changeSpanMs,50);assert.deepEqual(r.changes,r.values);
  r.advance(100,1);assert.equal(r.time,150);assert.equal(r.changeSpanMs,100);
  assert.equal(r.changes[0],Math.fround(f.samples[28]-f.samples[8]));
  const held=r.changes.slice();r.advance(1,1);assert.deepEqual(r.changes,held);
  r.seek(2);assert.equal(r.changeSpanMs,100);assert.equal(r.changes[0],Math.fround(raw[2]-raw[0]));
  r.playing=true;r.advance(100,1);assert.equal(r.sampleMs,100);assert.equal(r.time,300);assert.equal(r.changeSpanMs,100);
  assert.equal(r.changes[0],Math.fround(raw[4]-raw[2]));assert.equal(r.board.done,true);
  r.seek(0);assert.deepEqual([...r.changes],[0,0]);
  r.seek(1);assert.deepEqual(r.changes,raw.slice(0,2));
  assert.deepEqual(raw,original);assert.deepEqual(f.samples,fine);
});

test('change preserves sign flips and zero changes, and resets for a new recording',()=>{
  const f=fixture(),raw=new Float32Array([1,-1,-1,1,-1,1]),r=new Replay(f.record,raw);
  r.seek(2);assert.deepEqual([...r.changes],[-2,2]);
  for(const scale of ['linear','log']){
    assert.equal(activityIntensity(r.changes[0],1,scale,2),1);
    assert.equal(activityIntensity(r.changes[1],1,scale,2),1);
    assert.equal(activityIntensity(0,4,scale,2),0);
  }
  r.seek(3);assert.deepEqual([...r.changes],[0,0]);
  const next=new Replay(f.record,new Float32Array(6));assert.deepEqual([...next.changes],[0,0]);next.seek(2);assert.deepEqual([...next.changes],[0,0]);
  assert.deepEqual([...raw],[1,-1,-1,1,-1,1]);
});

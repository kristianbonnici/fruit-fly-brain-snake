import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import {decodeAdjacency,createConnectionLoader} from '../viewer/connections.mjs';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
function fixture(){const u=new Uint32Array([1,0,2,2,0,1,2,1,4,0,0,7,0]);const f=new Float32Array(u.buffer);f[9]=1.5;f[12]=.5;return u;}
test('adjacency shard preserves incoming/outgoing direction, counts and frozen efficacy',()=>{
  const u=fixture();assert.deepEqual(decodeAdjacency(u.buffer,0,'incoming',2),[[1,0,4,1.5]]);assert.deepEqual(decodeAdjacency(u.buffer,1,'outgoing',2),[[1,0,7,.5]]);
  assert.throws(()=>decodeAdjacency(u.buffer,2,'incoming',2));assert.throws(()=>decodeAdjacency(u.buffer.slice(0,20),0,'incoming',2));
  const bad=u.slice();bad[8]=0;assert.throws(()=>decodeAdjacency(bad.buffer,0,'incoming',2));
});
test('slow previous neuron and cancelled collection cannot commit stale connections',async()=>{
  const u=fixture(),sha=createHash('sha256').update(new Uint8Array(u.buffer)).digest('hex');let release,started;
  const gate=new Promise(r=>release=r),pending=new Promise(r=>started=r);let calls=0;const commits=[];
  const anatomy={nodes:[{},{}],connections:{block_size:2,incoming:[{file:'data/full-v90/edges/incoming-0.bin',sha256:sha}],outgoing:[{file:'data/full-v90/edges/outgoing-0.bin',sha256:sha}]}};
  const load=createConnectionLoader(anatomy,r=>commits.push(r),async()=>{if(calls++===0){started();await gate;}return new Response(u);});
  const old=load(0);await pending;await load(1);release();assert.equal(await old,false);assert.equal(commits.length,1);assert.equal(commits[0][0][2],7);
  load.cancel();
});

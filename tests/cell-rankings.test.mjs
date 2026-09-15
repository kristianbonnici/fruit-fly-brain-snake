import test from 'node:test';
import assert from 'node:assert/strict';
import {recordingStats,topCells} from '../viewer/cell-rankings.mjs';
test('negative activity counts; a steady strong cell is not variable',async()=>{
  const data=new Float32Array([-1,-.5,0, -1,.5,0, -1,-.5,0, -1,.5,0]);
  const stats=await recordingStats(data,3);
  assert.deepEqual([...stats.active],[1,.5,0]);assert.deepEqual([...stats.variable],[0,.5,0]);
  assert.deepEqual(topCells([0,1,2],stats.active,2),[0,1]);
  assert.deepEqual(topCells([0,1,2],stats.variable,2),[1,0]);
  assert.deepEqual([...data],[-1,-.5,0,-1,.5,0,-1,-.5,0,-1,.5,0]);
});
test('top cells matches full sorting, filters and stable ties',()=>{
  const scores=Float64Array.from({length:2000},(_,i)=>Math.sin(i)*.5),candidates=Array.from({length:2000},(_,i)=>i).filter(i=>i%3);
  assert.deepEqual(topCells(candidates,scores,200,true),candidates.slice().sort((a,b)=>Math.abs(scores[b])-Math.abs(scores[a])||a-b).slice(0,200));
  assert.deepEqual(topCells([3,2,1,0],[0,0,0,0],2),[0,1]);assert.deepEqual(topCells([],scores),[]);
});
test('cancel statistics from a superseded recording',async()=>{
  let current=true;const result=await recordingStats(new Float32Array(20),1,()=>current,async()=>{current=false;});assert.equal(result,null);
});
test('reject invalid buffers',async()=>{
  await assert.rejects(recordingStats(new Float32Array(5),2));await assert.rejects(recordingStats(new Float32Array([NaN]),1));
});

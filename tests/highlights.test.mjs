import test from 'node:test';
import assert from 'node:assert/strict';
import {highlightMask} from '../viewer/highlights.mjs';

const nodes=[{population:'optic',position:[1,2,3]},{population:'central',position:null},{population:'optic',position:null},{population:'cord',position:[4,5,6]}];
test('population emphasis combines groups, includes missing coordinates and has no default highlight',()=>{
  const values=new Float32Array([0,0,0,0]),none=new Set();
  assert.equal(highlightMask(nodes,none,none,values),null);
  const selected=new Set(['optic','central']);
  assert.deepEqual([...highlightMask(nodes,selected,none,values)],[1,1,1,0]);
  selected.delete('optic');assert.deepEqual([...highlightMask(nodes,selected,none,values)],[0,1,0,0]);
  selected.clear();assert.equal(highlightMask(nodes,selected,none,values),null);
});

test('change direction intersects selected populations, excludes zero and follows new recorded values',()=>{
  const values=new Float32Array([.1,-.2,-.3,0]),before=values.slice(),scratch=new Uint8Array(nodes.length),none=new Set();
  assert.deepEqual([...highlightMask(nodes,none,new Set(['falling']),values,scratch)],[0,1,1,0]);
  assert.deepEqual([...highlightMask(nodes,new Set(['optic']),new Set(['falling']),values,scratch)],[0,0,1,0]);
  assert.deepEqual([...highlightMask(nodes,none,new Set(['rising','falling']),values,scratch)],[1,1,1,0]);
  const next=new Float32Array([-.1,.2,.3,-0]);
  assert.deepEqual([...highlightMask(nodes,none,new Set(['falling']),next,scratch)],[1,0,0,0]);
  assert.deepEqual(values,before);assert.deepEqual([...next],[-Math.fround(.1),Math.fround(.2),Math.fround(.3),-0]);
  assert.equal(highlightMask(nodes,none,none,next,scratch),null);
});

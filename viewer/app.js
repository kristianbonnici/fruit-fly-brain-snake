import {recordingStats,topCells} from './cell-rankings.mjs';
import {siteFetch as fetch} from './transport.mjs';
import {Replay,createLoader,checkedBytes,FORMAT,sensoryDrive,activityIntensity} from './replay.mjs';
import {createBrain} from './brain.js?v=clear-selection-1';
import {createConnectionLoader} from './connections.mjs';
import {highlightMask} from './highlights.mjs';
const $=id=>document.getElementById(id);
const sensorNames=['Food · forward','Food · right','Obstacle · left','Obstacle · forward','Obstacle · right'];
const actionNames=['Left','Straight','Right'];
const lcdPalette={background:'#c7d4b0',grid:'#aebf9840',body:'#526543',head:'#354b2f',food:'#354b2f'};
// A display illustration inspired by blue/green sensitivity, not receptor responses.
// These structured-input recordings contain no spectral or ultraviolet measurements.
const flyContextPalette={background:'#102c46',grid:'#83b5d51c',body:'#49b881',head:'#9ce9ad',food:'#77c9ff'};
let manifest,anatomy,brain,replay,loader,selected=null,selectedSensor=null,loading=true,lastWall=0,dirty=true;
let contrast=4,activityScale='log',activityView='change',collections,collectionGeneration=0,edgeLoader,selectedEdges=[],edgeReady=false,edgeLimit=200,gridPage=0;
let rankingStats=null,rankingTask=null,rankedReplay=null,candidateCache=null,lastRankingSample=-1,chooseRankLeader=false;
const gridPageSize=()=>anatomy.nodes.length>4096?1024:anatomy.nodes.length;
const isVisual=()=>manifest?.signals.input==='image';
let rendered=0,elapsedRender=0,lastFps=0,populationMap,adjacency;
const sensorButtons=[],GRID_COLUMNS=64;
const highlightedPopulations=new Set(),highlightedDirections=new Set();
let highlights=null,highlightScratch,highlightRevision=0,lastHighlightRevision=-1,lastHighlightSample=-1,highlightReplay;
function refreshHighlightButtons(){
  for(const button of document.querySelectorAll('[data-population]'))button.setAttribute('aria-pressed',String(highlightedPopulations.has(button.dataset.population)));
  for(const button of document.querySelectorAll('[data-change-direction]'))button.setAttribute('aria-pressed',String(highlightedDirections.has(button.dataset.changeDirection)));
  const extraSelected=$('otherPopulationLegend').querySelectorAll('[aria-pressed="true"]').length;
  $('otherPopulations').querySelector('summary').textContent=extraSelected?`More populations · ${extraSelected} selected`:'More populations';
  $('clearHighlights').disabled=!highlightedPopulations.size&&!highlightedDirections.size;
  const chosen=[...highlightedPopulations].map(key=>populationMap[key].label).concat([...highlightedDirections].map(key=>key==='rising'?'Rising':'Falling'));
  $('highlightSummary').textContent=chosen.length===0?'None selected':chosen.length===1?chosen[0]:`${chosen.length} selected`;
  $('highlightSummary').title=chosen.join(', ');
}
function toggleHighlight(set,key){set.has(key)?set.delete(key):set.add(key);highlightRevision++;refreshHighlightButtons();dirty=true;}
function updateHighlights(values){
  if(highlightRevision===lastHighlightRevision&&highlightReplay===replay&&(!highlightedDirections.size||lastHighlightSample===replay.sampleIndex))return;
  if(highlightScratch?.length!==anatomy.nodes.length)highlightScratch=new Uint8Array(anatomy.nodes.length);
  highlights=highlightMask(anatomy.nodes,highlightedPopulations,highlightedDirections,values,highlightScratch);
  brain?.highlight(highlights);
  let matched=0,positioned=0;
  if(highlights)for(let i=0;i<highlights.length;i++)if(highlights[i]){matched++;if(anatomy.nodes[i].position)positioned++;}
  $('highlightStatus').textContent=highlights?`${matched.toLocaleString()} highlighted · ${positioned.toLocaleString()} in 3D · ${(matched-positioned).toLocaleString()} without coordinates. Selection outlines mark anatomy; activity values stay unchanged.`:'No highlights selected · select one or more labels to focus the brain and grid.';
  lastHighlightRevision=highlightRevision;lastHighlightSample=replay.sampleIndex;highlightReplay=replay;
}
function node(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function signed(value,digits=3){const text=value!==0&&Math.abs(value)<10**-digits?value.toExponential(2):value.toFixed(digits);return `${value>=0?'+':''}${text}`;}
function jsonFrom(bytes){return JSON.parse(new TextDecoder().decode(bytes));}
function fail(error){$('error').textContent=error.message;$('error').hidden=false;$('status').textContent='Could not load this recording';loading=true;setPlaying(false);enableControls(false);}
function enableControls(value){for(const id of ['play','previous','next','restart','timeline'])$(id).disabled=!value;}
function setPlaying(value){if(replay)replay.playing=value;lastWall=0;$('play').textContent=value?'Ⅱ Pause':'▶ Play';if(!loading)$('status').textContent=value?'Playing recorded activity':'Paused · explore any decision';}
function drawBoard(canvas,g,mini=false){
  const palette=mini?flyContextPalette:lcdPalette;
  const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.fillStyle=palette.background;c.fillRect(0,0,w,h);if(!g)return;
  const size=g.size,cell=w/size;
  c.strokeStyle=palette.grid;c.lineWidth=.65;for(let i=0;i<=size;i++){c.beginPath();c.moveTo(i*cell,0);c.lineTo(i*cell,h);c.stroke();c.beginPath();c.moveTo(0,i*cell);c.lineTo(w,i*cell);c.stroke();}
  g.body.forEach(([x,y],i)=>{c.fillStyle=i===0?palette.head:palette.body;const pad=mini?1:2;c.fillRect(x*cell+pad,y*cell+pad,cell-pad*2,cell-pad*2);});
  if(g.food){const [x,y]=g.food;c.fillStyle=palette.food;c.fillRect((x+.25)*cell,(y+.25)*cell,cell*.5,cell*.5);}
  if(!mini&&g.done){c.fillStyle='#c7d4b0e5';c.fillRect(0,h/2-29,w,58);c.fillStyle='#354b2f';c.font='17px ui-monospace, monospace';c.textAlign='center';c.fillText(g.reason==='Collision'?'GAME OVER':g.reason.toUpperCase(),w/2,h/2+5);}
}
function fitCanvas(canvas){const ratio=Math.min(devicePixelRatio,2),r=canvas.getBoundingClientRect();const w=Math.round(r.width*ratio),h=Math.round(r.height*ratio);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return [w,h];}
function populationColor(n){return populationMap[n.population].color;}
function gridColor(n,value,emphasized){const change=activityView==='change',hex=change?(value>=0?'#d5742b':'#18899b'):populationColor(n),base=[parseInt(hex.slice(1,3),16),parseInt(hex.slice(3,5),16),parseInt(hex.slice(5,7),16)];const a=activityIntensity(value,contrast,activityScale,change?2:1)*(emphasized?1:.08),background=emphasized?225:244;return `rgb(${base.map(v=>Math.round(background+(v-background)*a)).join(',')})`;}
function drawGrid(values){
  const canvas=$('activityGrid'),[w,h]=fitCanvas(canvas),c=canvas.getContext('2d');c.clearRect(0,0,w,h);
  const pageSize=gridPageSize(),start=gridPage*pageSize,end=Math.min(start+pageSize,anatomy.nodes.length);
  const rows=Math.ceil((end-start)/GRID_COLUMNS),dx=w/GRID_COLUMNS,dy=h/rows;
  anatomy.nodes.slice(start,end).forEach((n,k)=>{const i=start+k,x=(k%GRID_COLUMNS)*dx,y=Math.floor(k/GRID_COLUMNS)*dy;c.fillStyle=gridColor(n,values[i],!highlights||highlights[i]);c.fillRect(x+1,y+1,Math.max(1,dx-2),Math.max(1,dy-2));
    if(!n.position){c.strokeStyle='#9d9eaa';c.lineWidth=.6;c.strokeRect(x+1.5,y+1.5,Math.max(1,dx-3),Math.max(1,dy-3));}
    if(i===selected){c.strokeStyle='#182c43';c.lineWidth=2;c.strokeRect(x+.5,y+.5,dx-1,dy-1);}
    else if(selectedSensor!==null&&n.sensory_slot>=0){c.strokeStyle='#0071e3';c.lineWidth=.9;c.strokeRect(x+1,y+1,dx-2,dy-2);}
  });
  $('gridPaging').hidden=pageSize===anatomy.nodes.length;$('gridPrevious').disabled=start===0;$('gridNext').disabled=end===anatomy.nodes.length;$('gridPageLabel').textContent=`${(start+1).toLocaleString()}–${end.toLocaleString()} of ${anatomy.nodes.length.toLocaleString()} neurons`;
  canvas.setAttribute('aria-label',selected===null?`All ${anatomy.nodes.length} simulated neurons. No cell selected. Click a cell or use arrow keys to select one.`:`All ${anatomy.nodes.length} simulated neurons. Selected ${anatomy.nodes[selected].id}, activity ${signed(replay.values[selected],4)}.${activityView==='change'?` Change ${signed(values[selected],4)} over ${replay.changeSpanMs} ms.`:''} Arrow keys select; outlined cells have no 3D coordinates.`);
}
function readableActivity(value){
  if(value===0)return '0';
  if(Math.abs(value)<1e-8)return 'Near zero';
  return `${value>0?'+':'−'}${Math.abs(value).toLocaleString('en-US',{useGrouping:false,maximumSignificantDigits:4})}`;
}
function drawHistory(){
  if(!replay||selected===null)return;
  const canvas=$('history'),[w,h]=fitCanvas(canvas),c=canvas.getContext('2d'),ratio=Math.min(devicePixelRatio,2);
  const count=Math.min(replay.sampleIndex,40),start=replay.sampleIndex-count;
  const samples=Array.from({length:count+1},(_,k)=>replay.valuesAt(start+k)[selected]);
  const peak=Math.max(...samples.map(Math.abs)),zoom=$('historyZoom').checked,limit=zoom&&peak>0?peak*1.1:1;
  const axisLabel=value=>Math.abs(value)<1e-8&&value!==0?value.toExponential(1):value.toLocaleString('en-US',{useGrouping:false,maximumSignificantDigits:2});
  c.clearRect(0,0,w,h);c.strokeStyle='#e4e6eb';c.lineWidth=1;c.beginPath();c.moveTo(0,h/2);c.lineTo(w,h/2);c.stroke();
  c.fillStyle='#90949b';c.font=`${9*ratio}px -apple-system`;c.fillText('+'+axisLabel(limit),0,12*ratio);c.fillText('0',0,h/2-3*ratio);c.fillText('−'+axisLabel(limit),0,h-2);
  c.strokeStyle=populationColor(anatomy.nodes[selected]);c.lineWidth=2;c.beginPath();
  const fromTime=replay.sampleTime(start),span=replay.time-fromTime||1,left=62*ratio;
  const y=value=>h/2-value/limit*(h/2-12*ratio);
  c.moveTo(left,y(samples[0]));
  for(let k=0;k<count;k++){const x=left+(w-left-5*ratio)*(replay.sampleTime(start+k+1)-fromTime)/span;c.lineTo(x,y(samples[k]));c.lineTo(x,y(samples[k+1]));}
  c.stroke();
  const value=replay.values[selected];
  $('historyLabel').textContent=count?`Last ${((replay.time-fromTime)/1000).toFixed(2)} seconds · ${count} recorded samples`:'Before the first recorded sample';
  $('neuronValue').textContent=readableActivity(value);$('rawNeuronValue').textContent=String(value);
  $('signalMeaning').textContent=!replay.sampleIndex?'Initial state — playback has not reached a recorded sample.':value===0?'At the model’s zero reference level.':Math.abs(value)<.001?'Very close to zero: less than 0.001 in magnitude.':value>0?'Above the model’s zero reference level.':'Below the model’s zero reference level.';
  $('historyStart').textContent=(fromTime/1000).toFixed(2)+' s';$('historyEnd').textContent=(replay.time/1000).toFixed(2)+' s';
  $('historyRange').textContent=zoom&&peak>0?`Zoomed scale: −${axisLabel(limit)} to +${axisLabel(limit)}. Fits this cell’s visible history; values are unchanged.`:zoom?'No nonzero signal in this window. Showing the full −1 to +1 scale.':'Full scale: −1 to +1, the same for every cell.';
  $('historyExplanation').textContent='Each step is a recorded sample. Zero is the model’s reference level. '+(zoom?'Zoom enlarges small changes; use full scale to compare signal sizes across cells.':'Small signals can look flat at full scale; zoom in to see their changes.');
  canvas.setAttribute('aria-label',`Recorded activity for ${anatomy.nodes[selected].type||'selected cell'}, from ${(fromTime/1000).toFixed(2)} to ${(replay.time/1000).toFixed(2)} seconds. Current value ${value}. Vertical scale minus ${limit} to plus ${limit}.`);
}
function updateProjection(){
  if(selected===null)return;
  const n=anatomy.nodes[selected],slot=n.sensory_slot;
  if(slot<0){$('projection').textContent=selectedSensor===null?'Select a sensor to inspect its learned projection into the model.':'This neuron does not directly receive the structured input projection.';return;}
  const weight=selectedSensor===null?'':`${sensorNames[selectedSensor]} weight ${signed(anatomy.encoder_weights[slot][selectedSensor],5)}. `;
  if(isVisual()){const value=replay?.inputFrame?.sensory_drive[slot];$('projection').textContent=weight+(value===undefined?'No input yet. ':`Recorded sensory drive ${signed(value,4)}. `)+'Frozen learned projection from the recorded image encoder estimates.';return;}
  const value=replay?.inputFrame?`Derived drive ${signed(sensoryDrive(anatomy,replay.inputFrame.sensors,slot),4)}. `:'';
  $('projection').textContent=weight+value+'Drive = 3 × tanh(weighted inputs + bias), derived from recorded inputs and frozen weights.';
}
function drawImage(canvas,pixels){const c=canvas.getContext('2d'),cell=canvas.width/16;c.fillStyle='#000';c.fillRect(0,0,canvas.width,canvas.height);if(!pixels)return;pixels.forEach((v,i)=>{const gray=Math.round(v*255);c.fillStyle=`rgb(${gray},${gray},${gray})`;c.fillRect((i%16)*cell,Math.floor(i/16)*cell,cell,cell);});}
function paint(){
  if(!replay||loading)return;
  const f=replay.frame,input=replay.inputFrame,g=replay.board,values=replay.values;
  drawBoard($('game'),g);
  if(isVisual()){const current=input?.image??replay.record.frames[0].image;const previous=input&&replay.inputIndex>1?replay.record.frames[replay.inputIndex-2].image:null;drawImage($('inputBoard'),current);drawImage($('previousInputBoard'),previous);}else drawBoard($('inputBoard'),input?.before ?? replay.record.initial,true);
  $('score').textContent=String(g.score).padStart(3,'0');$('gameState').textContent=g.done?'FINISHED':replay.index?'REPLAY':'READY';
  $('game').setAttribute('aria-label',`Resulting Snake board. Completed decisions ${replay.index}. Score ${g.score}. ${g.done?g.reason:''}`);
  $('boardTime').textContent=replay.index?`After move ${replay.index}`:'Initial board';
  $('inputTime').textContent=input?`Input for move ${replay.inputIndex}`:'No decision yet';
  sensorButtons.forEach((button,i)=>{button.querySelector('strong').textContent=input?(i<2?signed(input.sensors[i]):input.sensors[i].toFixed(3)):'—';button.querySelector('i').style.width=`${input?Math.abs(input.sensors[i])*100:0}%`;});
  $('foodDot').style.opacity=input?'1':'0';if(input){$('foodDot').setAttribute('cx',String(50+input.sensors[1]*30));$('foodDot').setAttribute('cy',String(38-input.sensors[0]*23));}
  [...$('actions').children].forEach((el,i)=>{el.classList.toggle('selected',!!f&&f.action===i);el.title=f?`Recorded logit: ${f.logits[i].toFixed(5)} (not a probability)`:'';});
  $('clock').replaceChildren(document.createTextNode(`${(replay.time/1000).toFixed(replay.sampleMs===10?2:1)} s `),node('small',`/ ${(replay.length/10).toFixed(1)} s`));
  $('sampleBadge').textContent=`${replay.sampleMs} ms samples · graded activity${replay.substepMoves && replay.substepMoves<replay.length?' · 10 ms through move '+replay.substepMoves:''}`;
  $('neuralTime').textContent=replay.phaseMs?`Move ${replay.inputIndex} · ${replay.phaseMs} / 100 ms` : replay.index?`Move ${replay.index} · neural endpoint`:'Before the first input';
  $('decision').textContent=replay.phaseMs?`${replay.index} / ${replay.length} completed · processing move ${replay.inputIndex}`:replay.index?`Decision ${replay.index} / ${replay.length}${g.done?' · '+g.reason:''}`:'Initial state · before the first decision';
  $('timeline').value=String(replay.index);$('timeline').setAttribute('aria-valuetext',`${replay.index} of ${replay.length} completed decisions`);
  $('previous').disabled=replay.sampleIndex===0;$('next').disabled=replay.index===replay.length;
  const change=activityView==='change',displayValues=change?replay.changes:values;
  updateHighlights(displayValues);
  $('viewNote').textContent=change?(replay.time===0?'No change before the first input.':replay.changeSpanMs<100?`Change since the initial state · ${replay.changeSpanMs} ms.`:'Change from 100 ms earlier.')+' Warm = rising · cool = falling recorded values.':'Population colours · brightness shows recorded activity strength.';
  $('changeLegend').hidden=!change;
  $('gridViewLabel').textContent=change?'Rising / falling · change over up to 100 ms':'Population colours · contrast-adjusted activity';
  $('neuronChange').hidden=!change||selected===null;
  if(change&&selected!==null)$('neuronChange').textContent=`Derived change ${signed(displayValues[selected],5)} over ${replay.changeSpanMs} ms. Activity and history above show the original signed values.`;
  if($('cellRanking').value==='changing'&&lastRankingSample!==replay.sampleIndex)refreshList();
  brain?.update(displayValues,contrast,activityScale,activityView);drawGrid(displayValues);drawHistory();updateProjection();
  // Visible telemetry is also useful for repeatable browser checks; no control API.
  document.body.dataset.decision=String(replay.index);document.body.dataset.game=replay.record.id;document.body.dataset.sample=String(replay.sampleIndex);document.body.dataset.sampleMs=String(replay.sampleMs);
  document.body.dataset.recordingIdentity=replay.record.recording_identity;
  document.body.dataset.activityView=activityView;
}
function requestRankingStats(){
  if(!replay||loading||rankingStats||rankingTask)return;
  const target=replay;
  const task=recordingStats(target.activity,anatomy.nodes.length,()=>replay===target&&!loading);
  rankingTask=task;
  task.then(result=>{if(rankingTask!==task)return;rankingTask=null;if(!result)return;rankingStats=result;refreshList();}).catch(error=>{if(rankingTask!==task)return;rankingTask=null;$('rankingStatus').textContent='Could not calculate rankings: '+error.message;});
}
function refreshList(){
  if(!anatomy)return;
  if(rankedReplay!==replay){rankedReplay=replay;rankingStats=null;rankingTask=null;lastRankingSample=-1;}
  const mode=$('cellRanking').value,query=$('neuronSearch').value.trim().toLowerCase(),filter=$('coordinateFilter').value;
  if(!candidateCache||candidateCache.anatomy!==anatomy||candidateCache.query!==query||candidateCache.filter!==filter){
    const matches=[];
    for(let i=0;i<anatomy.nodes.length;i++){
      const n=anatomy.nodes[i];
      if((!query||`${n.id} ${n.type} ${n.population} ${populationMap[n.population].label}`.toLowerCase().includes(query))
        &&(filter==='all'||filter==='missing'&&!n.position||filter==='positioned'&&n.position||filter==='sensory'&&n.sensory_slot>=0||filter==='output'&&n.output))matches.push(i);
    }
    candidateCache={anatomy,query,filter,matches};
  }
  const matches=candidateCache.matches;
  const help={browse:'Choose a ranking to find cells with strong or changing signals.',active:'Highest average signal magnitude across this recording’s 100 ms samples. Negative signals count too.',variable:'Largest fluctuations across this recording’s 100 ms samples, measured by standard deviation.',changing:'Largest absolute signal change over the last 100 ms. Updates with playback; pause to browse a stable list.'};
  $('rankingHelp').textContent=help[mode]+(mode==='browse'?'':' Activity does not measure importance for playing Snake.');
  if(loading||!replay||(mode!=='browse'&&mode!=='changing'&&!rankingStats)){
    $('rankingStatus').textContent=loading||!replay?'Waiting for the recording…':'Calculating this recording’s rankings…';
    $('neuronList').replaceChildren();$('matchCount').textContent='';
    if(!loading&&(mode==='active'||mode==='variable'))requestRankingStats();
    return;
  }
  const scores=mode==='changing'?replay.changes:rankingStats?.[mode];
  const shown=mode==='browse'?matches.slice(0,200):topCells(matches,scores,200,mode==='changing');
  const noChange=mode==='changing'&&(!replay.sampleIndex||shown.every(i=>scores[i]===0));
  $('rankingStatus').textContent=mode==='browse'?'':noChange?'No recorded changes at this moment.':mode==='changing'?`At ${(replay.time/1000).toFixed(2)} s · change over ${replay.changeSpanMs} ms`:'Whole recording · order stays fixed during playback';
  if(noChange)shown.length=0;
  if(chooseRankLeader){chooseRankLeader=false;if(shown.length){selectNeuron(shown[0]);return;}}
  const options=shown.map((i,rank)=>{
    const n=anatomy.nodes[i],metric=mode==='browse'?'':` · ${Math.abs(scores[i]).toPrecision(3)}`;
    const option=node('option',`${mode==='browse'?'':`${rank+1}. `}${n.type||'Unnamed'} · ${n.id}${metric}${!n.position?' ◌':''}`);
    option.value=String(i);return option;
  });
  if(!shown.includes(selected)&&matches.includes(selected)){
    const n=anatomy.nodes[selected],option=node('option',`Selected: ${n.type||'Unnamed'} · ${n.id}${!n.position?' ◌':''}`);option.value=String(selected);options.push(option);
  }
  $('neuronList').replaceChildren(...options);$('neuronList').value=String(selected);
  $('matchCount').textContent=`${matches.length.toLocaleString()} matching cells${matches.length>200?` · ${mode==='browse'?'first':'top'} 200 shown`:''}`;
  lastRankingSample=replay.sampleIndex;
}
function connectionRows(){
  if(selected===null){brain?.connections([]);$('edgeRows').replaceChildren();$('edgeCoverage').textContent='';$('connectionSummary').textContent='Connections to other cells';$('moreConnections').hidden=true;return;}
  const direction=$('edgeDirection').value;
  if(!edgeReady){$('connectionSummary').textContent='Loading connections to other cells…';$('edgeRows').replaceChildren();$('edgeCoverage').textContent='';$('moreConnections').hidden=true;brain?.connections([]);return;}
  const all=selectedEdges.filter(e=>direction==='both'||(direction==='outgoing'?e[0]===selected:e[1]===selected));
  const valid=all.filter(e=>anatomy.nodes[e[0]].position&&anatomy.nodes[e[1]].position&&e[0]!==e[1]);
  const outgoing=valid.filter(e=>e[0]===selected).slice(0,32),incoming=valid.filter(e=>e[0]!==selected).slice(0,32);
  brain?.connections([...outgoing,...incoming]);
  $('edgeCoverage').textContent=`${outgoing.length+incoming.length} lines · ${all.length.toLocaleString()} connections · ${Math.min(edgeLimit,all.length)} listed`;
  $('connectionSummary').textContent=`Connections to other cells · ${selectedEdges.length.toLocaleString()} links`;
  const rows=all.slice(0,edgeLimit).map(e=>{const outgoing=e[0]===selected,other=outgoing?e[1]:e[0],n=anatomy.nodes[other],tr=node('tr');tr.append(node('td',outgoing?'Outgoing →':'Incoming ←'));const cell=node('td'),button=node('button',`${n.type||'Untyped'} · ${n.id}${!n.position?' ◌':''}`);button.onclick=()=>selectNeuron(other);cell.append(button);tr.append(cell,node('td',e[2].toLocaleString()),node('td',anatomy.nodes[e[0]].assigned_sign>0?'Excitatory (+)':'Inhibitory (−)'),node('td',e[3].toFixed(4)));return tr;});
  $('edgeRows').replaceChildren(...rows);$('moreConnections').hidden=all.length<=edgeLimit;$('moreConnections').textContent=`Show more · ${Math.min(edgeLimit,all.length)} of ${all.length} connections`;
}
function selectNeuron(index){
  if(!anatomy)return;
  if(index===null){
    selected=null;selectedSensor=null;chooseRankLeader=false;edgeLoader?.cancel();selectedEdges=[];edgeReady=false;
    brain?.select(null);brain?.connections([]);brain?.showRecipients(null);
    sensorButtons.forEach(button=>button.setAttribute('aria-pressed','false'));
    $('sensorHint').textContent=isVisual()?'Actual previous/current grayscale frames. Select an encoder estimate to inspect its projection.':'Select a sensor to locate its receiving neurons.';
    $('selectionLabel').textContent='No cell selected · click a cell to explore it';
    $('neuronName').textContent='Select a cell';$('neuronPopulation').textContent='';$('neuronId').textContent='';
    $('cellDescription').textContent='Click a cell in the brain or activity grid, or choose one from the list.';
    $('neuronFacts').replaceChildren();$('projection').textContent='';$('cellReadout').hidden=true;
    $('focusNeuron').disabled=true;$('inspectSelected').disabled=true;$('clearCellSelection').disabled=true;
    $('cellTechnical').open=false;$('connections').open=false;
    refreshList();connectionRows();dirty=true;return;
  }
  if(!Number.isInteger(index)||index<0||index>=anatomy.nodes.length)return;
  $('cellReadout').hidden=false;$('inspectSelected').disabled=false;$('clearCellSelection').disabled=false;
  selected=Math.max(0,Math.min(anatomy.nodes.length-1,index));
  gridPage=Math.floor(selected/gridPageSize());
  const n=anatomy.nodes[selected];$('neuronPopulation').textContent=populationMap[n.population].label.toUpperCase();$('neuronPopulation').style.color=populationColor(n);
  $('selectionLabel').textContent=n.position?`Selected ${n.type||'neuron'} · ${n.id} · ring marks location, not cell size`:'Selected neuron has no 3D coordinates · its connections are listed in the inspector';
  $('neuronName').textContent=n.type||'Unnamed cell';
  $('cellDescription').textContent=`One simulated cell in the ${populationMap[n.population].label} group. `+(n.position?'Use “Show in brain” to find its location.':'Its 3D location is unavailable, but its activity and connections can still be explored.');$('neuronId').textContent=`Body ID ${n.id}`;
  const facts=[n.position?'Mapped in 3D':'No 3D coordinates',n.assigned_sign>0?'Assigned excitatory':'Assigned inhibitory',`Transmitter: ${n.nt||'unknown'}`,n.core?'Core network':'Surrounding network'];
  if(n.sensory_slot>=0)facts.push('Sensory recipient');if(n.output)facts.push('Readout neuron');
  $('neuronFacts').replaceChildren(...facts.map(v=>node('span',v,'fact')));$('focusNeuron').disabled=!n.position||!brain;
  refreshList();$('neuronList').value=String(selected);brain?.select(selected);edgeLimit=200;
  if(edgeLoader){edgeReady=false;edgeLoader(selected).catch(error=>{$('connectionSummary').textContent=error.message;});}else{selectedEdges=adjacency[selected];edgeReady=true;}connectionRows();dirty=true;
}
function selectSensor(index){
  selectedSensor=selectedSensor===index?null:index;
  sensorButtons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===selectedSensor)));
  brain?.showRecipients(selectedSensor);
  if(selectedSensor===null){$('sensorHint').textContent=isVisual()?'Numbers are recorded visual-encoder estimates. Select one to locate its receiving neurons.':'Select a sensor to locate its receiving neurons.';}
  else{
    const slots=anatomy.sensory_nodes.filter(j=>anatomy.encoder_weights[anatomy.nodes[j].sensory_slot][index]!==0);
    const positioned=slots.filter(j=>anatomy.nodes[j].position).length;
    $('sensorHint').textContent=`${slots.length} learned recipients · ${positioned} in 3D · ${slots.length-positioned} in the grid only. Each sensor can affect every recipient.`;
    const strongest=slots.reduce((best,j)=>Math.abs(anatomy.encoder_weights[anatomy.nodes[j].sensory_slot][index])>Math.abs(anatomy.encoder_weights[anatomy.nodes[best].sensory_slot][index])?j:best,slots[0]);
    selectNeuron(strongest);
  }dirty=true;
}
async function loadGame(id){
  const entry=manifest.catalog.find(g=>g.id===id);if(!entry)return;
  loading=true;rankingStats=null;rankingTask=null;setPlaying(false);enableControls(false);refreshList();$('error').hidden=true;$('status').textContent='Verifying complete recording…';
  try{await loader(entry);}catch(error){fail(error);}
}
function refreshDisplaySummary(){ $('displaySummary').textContent=`${activityScale==='log'?'Logarithmic':'Linear'} · ${contrast}×`; }
function initControls(){
  $('play').onclick=()=>{if(!replay||loading)return;if(replay.index===replay.length)replay.seek(0);setPlaying(!replay.playing);dirty=true;};
  const seek=idx=>{setPlaying(false);replay?.seek(idx);dirty=true;};
  $('previous').onclick=()=>seek(replay.index-(replay.phaseMs?0:1));$('next').onclick=()=>seek(replay.index+1);$('restart').onclick=()=>seek(0);
  $('timeline').oninput=()=>seek(Number($('timeline').value));$('collection').onchange=()=>loadCollection($('collection').value).catch(fail);
  $('gridPrevious').onclick=()=>{gridPage=Math.max(0,gridPage-1);dirty=true;};$('gridNext').onclick=()=>{gridPage=Math.min(Math.ceil(anatomy.nodes.length/gridPageSize())-1,gridPage+1);dirty=true;};
  $('moreConnections').onclick=()=>{edgeLimit+=200;connectionRows();};$('recording').onchange=()=>loadGame($('recording').value);
  $('activityContrast').oninput=()=>{contrast=Number($('activityContrast').value);refreshDisplaySummary();$('contrastValue').textContent=`${contrast}×`;$('activityContrast').setAttribute('aria-valuetext',`${contrast} times display contrast; recorded values unchanged`);dirty=true;};
  $('activityScale').onchange=()=>{activityScale=$('activityScale').value;refreshDisplaySummary();$('scaleNote').textContent=activityScale==='log'?'Log scale reveals small signals · zero stays quiet · numeric values are unchanged.':'Linear magnitude scale · contrast curve applied · numeric values are unchanged.';dirty=true;};
  $('activityView').onchange=()=>{activityView=$('activityView').value;highlightedDirections.clear();highlightRevision++;refreshHighlightButtons();dirty=true;};
  $('clearCellSelection').onclick=()=>selectNeuron(null);
  $('activityGrid').addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();selectNeuron(null);}});
  document.addEventListener('click',e=>{if(e.target.matches('body,main,.stage,.brain-card,.left-column'))selectNeuron(null);});
  $('clearHighlights').onclick=()=>{highlightedPopulations.clear();highlightedDirections.clear();highlightRevision++;refreshHighlightButtons();dirty=true;};
  for(const button of document.querySelectorAll('[data-change-direction]'))button.onclick=()=>toggleHighlight(highlightedDirections,button.dataset.changeDirection);
  $('speed').onchange=()=>{lastWall=0;};
  $('brainPreset').onclick=()=>{brain?.setPreset('brain');$('brainPreset').setAttribute('aria-pressed','true');$('cnsPreset').setAttribute('aria-pressed','false');};
  $('cnsPreset').onclick=()=>{brain?.setPreset('cns');$('brainPreset').setAttribute('aria-pressed','false');$('cnsPreset').setAttribute('aria-pressed','true');};
  $('resetCamera').onclick=()=>brain?.reset();$('focusNeuron').onclick=()=>{brain?.focus();$('brain').scrollIntoView({block:'center',behavior:'instant'});};
  $('cellRanking').onchange=()=>{chooseRankLeader=$('cellRanking').value!=='browse';refreshList();};
  $('historyZoom').onchange=()=>{dirty=true;};
  $('neuronSearch').oninput=refreshList;$('coordinateFilter').onchange=refreshList;
  $('neuronList').onchange=()=>{if($('neuronList').selectedIndex>=0)selectNeuron(Number($('neuronList').value));};$('edgeDirection').onchange=connectionRows;
  $('inspectSelected').onclick=()=>{$('inspector').scrollIntoView({behavior:'instant',block:'start'});$('neuronList').focus({preventScroll:true});};
  $('activityGrid').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();const col=Math.floor((e.clientX-r.left)/r.width*GRID_COLUMNS),row=Math.floor((e.clientY-r.top)/r.height*Math.ceil(Math.min(gridPageSize(),anatomy.nodes.length-gridPage*gridPageSize())/GRID_COLUMNS));selectNeuron(gridPage*gridPageSize()+row*GRID_COLUMNS+col);};
  $('activityGrid').onkeydown=e=>{const offset={ArrowLeft:-1,ArrowRight:1,ArrowUp:-GRID_COLUMNS,ArrowDown:GRID_COLUMNS}[e.key];if(offset){e.preventDefault();selectNeuron(selected===null?gridPage*gridPageSize():Math.max(0,Math.min(anatomy.nodes.length-1,selected+offset)));}};
  $('about').onclick=()=>$('notes').showModal();$('closeNotes').onclick=()=>$('notes').close();
  addEventListener('visibilitychange',()=>{lastWall=0;});addEventListener('resize',()=>{dirty=true;});
}
function animate(now){
  const elapsed=lastWall?now-lastWall:0;lastWall=now;
  if(!document.hidden){
    if(!loading&&replay?.advance(elapsed,Number($('speed').value))){dirty=true;if(!replay.playing)setPlaying(false);}
    if(dirty){paint();dirty=false;}
    brain?.render();
    if(lastFps){rendered++;elapsedRender+=now-lastFps;}lastFps=now;
    if(elapsedRender>=1000){$('fps').textContent=`${Math.round(rendered*1000/elapsedRender)} FPS`;rendered=0;elapsedRender=0;}
  }else{lastFps=0;}
  requestAnimationFrame(animate);
}
async function loadCollection(id){
  const generation=++collectionGeneration;loader?.cancel();edgeLoader?.cancel();loading=true;setPlaying(false);enableControls(false);$('recording').disabled=true;$('error').hidden=true;$('status').textContent='Loading recorded anatomy…';
  try{
  const item=collections.collections.find(v=>v.id===id);if(!item)throw Error('Unknown recording collection.');
  const response=await fetch(item.file);if(!response.ok)throw Error('No exported recordings found.');const incoming=await response.json();
  if(incoming.format!==FORMAT||incoming.signals.live||!['structured','image'].includes(incoming.signals.input))throw Error('Unsupported recording collection.');
  const [anatomyBytes,positionBytes]=await Promise.all([checkedBytes(await fetch(incoming.anatomy_file),incoming.anatomy_sha256),checkedBytes(await fetch(incoming.positions_file),incoming.positions_sha256)]);
  const newAnatomy=jsonFrom(anatomyBytes);if(newAnatomy.format!==FORMAT||newAnatomy.identity!==incoming.anatomy_identity||newAnatomy.nodes.length!==incoming.coverage.simulated||positionBytes.byteLength!==incoming.positions_count*12)throw Error('Anatomy coverage/identity mismatch.');
  if(generation!==collectionGeneration)return;
  brain?.dispose();brain=null;replay=null;manifest=incoming;anatomy=newAnatomy;selected=null;selectedSensor=null;gridPage=0;selectedEdges=[];edgeReady=false;
  highlightedPopulations.clear();highlightedDirections.clear();highlightRevision++;highlights=null;refreshHighlightButtons();
  $('neuronSearch').value='';$('coordinateFilter').value='all';$('brainPreset').setAttribute('aria-pressed','true');$('cnsPreset').setAttribute('aria-pressed','false');
  populationMap=Object.fromEntries(anatomy.populations.map(p=>[p.key,p]));
  if(anatomy.connections){adjacency=null;edgeLoader=createConnectionLoader(anatomy,rows=>{if(generation!==collectionGeneration)return;selectedEdges=rows;edgeReady=true;connectionRows();},fetch);}
  else{edgeLoader=null;adjacency=anatomy.nodes.map(()=>[]);anatomy.edges.forEach(e=>{adjacency[e[0]].push(e);if(e[0]!==e[1])adjacency[e[1]].push(e);});adjacency.forEach(edges=>edges.sort((a,b)=>b[2]-a[2]||a[0]-b[0]||a[1]-b[1]));}
  $('brainFallback').hidden=true;for(const id of ['brainPreset','cnsPreset','resetCamera'])$(id).disabled=false;
  try{brain=createBrain($('brain'),anatomy,new Float32Array(positionBytes),selectNeuron);}catch(error){console.error(error);$('brainFallback').hidden=false;for(const id of ['brainPreset','cnsPreset','resetCamera'])$(id).disabled=true;}
  const c=manifest.coverage,visual=isVisual();$('brainCount').textContent=`${c.simulated.toLocaleString()} simulated neurons`;$('coverage').textContent=`${c.simulated.toLocaleString()} simulated · ${c.simulated_positioned.toLocaleString()} positioned · ${c.simulated_unpositioned.toLocaleString()} without coordinates`;
  $('footerCoverage').textContent=`${c.omitted.toLocaleString()} neurons outside this simulation · ${c.positioned.toLocaleString()} total positions`;
  $('modelLabel').textContent=visual?'V90 · full-network internal learning':'Frozen policy · V73 correction';$('modelSummary').textContent=visual?'Image input · Full network · Recorded replay':'Structured inputs · Reduced anatomical model · Recorded replay';
  $('sensorsHeading').textContent=visual?'Visual input':'Structured sensors';$('previousImageContext').hidden=!visual;$('visionNote').hidden=!visual;$('estimateLabel').hidden=!visual;$('scenePalette').hidden=visual;
  $('sceneContextLabel').textContent=visual?'Current frame':'Input scene context';$('inputBoard').setAttribute('aria-label',visual?'Exact current image supplied to the visual encoder':'Pre-action board illustrated in blue and green');
  if(visual)$('inputBoard').removeAttribute('aria-describedby');else $('inputBoard').setAttribute('aria-describedby','scenePalette');
  $('sensorHint').textContent=visual?'Actual previous/current grayscale frames. Numbers are learned encoder estimates, not supplied game geometry. Select one to inspect its projection.':'Select a sensor to locate its receiving neurons.';
  const populationButton=p=>{const chip=node('button',undefined,'population-chip legend-button'),dot=node('i');dot.style.background=p.color;chip.style.setProperty('--chip-color',p.color);chip.style.setProperty('--chip-tint',p.color+'16');chip.append(dot,document.createTextNode(p.label));chip.title=`${p.count.toLocaleString()} simulated neurons · select to highlight`;chip.dataset.population=p.key;chip.setAttribute('aria-pressed','false');chip.onclick=()=>toggleHighlight(highlightedPopulations,p.key);return chip;};
  $('populationLegend').replaceChildren(...anatomy.populations.filter(p=>p.count>=500||!visual).map(populationButton));
  const otherPopulations=anatomy.populations.filter(p=>p.count<500&&visual);
  $('otherPopulationLegend').replaceChildren(...otherPopulations.map(populationButton));$('otherPopulations').hidden=!otherPopulations.length;$('otherPopulations').open=false;
  $('anatomyLegendNote').textContent=c.omitted?'Faint backdrop: omitted anatomy':'Full measured network · quiet cells remain visible';
  sensorButtons.length=0;$('sensorValues').replaceChildren();sensorNames.forEach((name,i)=>{const button=node('button',undefined,'sensor-button');button.setAttribute('aria-pressed','false');button.title=visual?'Recorded estimate from the learned two-frame visual encoder. No exact game-geometry sensor enters this policy.':i<2?'Egocentric food displacement divided by 15':'Reciprocal distance to occupied body or wall within 4 cells; the current tail counts as occupied. This is not a legal-action flag.';button.append(node('i'),node('span',name),node('strong','—'));button.onclick=()=>selectSensor(i);sensorButtons.push(button);$('sensorValues').append(button);});
  $('recording').replaceChildren(...manifest.catalog.map(g=>{const option=node('option',`${(g.label??'Game '+(Number(g.id.slice(5))+1)).replace(/\s*·\s*\d+\s+food\b/gi,'').replace('Stronger selected game','Stronger game')} · ${g.score} food · ${g.moves} moves`);option.value=g.id;return option;}));$('recording').value=manifest.default_game;$('recording').disabled=false;
  $('notesTitle').textContent=visual?'Recorded play through the full network.':'A real recording, a reduced model.';
  $('notesModel').textContent=visual?'The V90 policy processes current and previous grayscale images through a frozen learned encoder. Its five estimates feed a fixed sensory interface into all 165,122 neurons and 25,563,197 measured connections. Expert demonstrations trained internal efficacy; no teacher chooses actions during these games. This viewer executes no neural model.':'The policy receives five structured sensor values and acts through a reduced anatomical model of 1,055 neurons. This viewer replays its measured states. It does not train or execute the neural model.';
  $('notesAnatomy').textContent=visual?'All 165,122 neurons are recorded. 141,000 have coordinates; 24,122 remain inspectable through search and the paged activity grid. Full-network inclusion does not imply every neuron is active or that every pathway has a proven causal role.':'The full anatomical backdrop provides context. Neurons outside the reduced simulation have no recorded activity. Missing 3D coordinates are shown in the linked grid, never assigned invented positions.';
  $('notesActivity').textContent=visual?'Activity is signed graded deviation in [−1, 1]. Every game has measured 100 ms endpoints. Game 1 also has real 10 ms samples through move 32; the sample interval changes afterwards. Values hold between measured samples. Connection arrows show direction, not travelling signals. Learned efficacy stays fixed throughout replay.':'Activity is signed graded deviation in [−1, 1]. Separate frozen-model 10 ms captures exactly match the original decision endpoints. These are not spikes; lines show measured connectivity, not axon paths or travelling signals.';
  $('notesPalette').hidden=visual;
  $('notesContrast').textContent='In Population activity, colour identifies the annotated population. The default logarithmic scale reveals small recorded magnitudes with log(1 + |activity| × 1,000,000) / log(1,000,001), followed by the selected contrast curve. The same fixed scale applies to every cell and every frame; there is no per-frame normalization. Zero produces no coloured activity. A separate faint layer shows anatomy. Choose Linear for direct magnitudes before contrast. Signed numeric activity and history are unchanged.';
  $('notesInput').textContent=visual?'The main board shows the last completed action. The Visual input panel shows the exact current and previous grayscale frames for the decision being processed, with a blank previous frame only at game start. The frozen encoder evaluates eight rotated/reflected views of this pair. At an endpoint it retains that decision’s pre-action input. The five numbers are recorded visual-encoder estimates; ground-truth geometry and teacher actions are not supplied during play.':'The board shows the result of the last decision. During a decision window, the sensor panel shows the input being processed while the board waits for the completed action. At each endpoint it retains that decision’s earlier input scene. The scene image is context; this policy consumes the five numbers.';
  $('evaluationNotes').textContent=`V${manifest.evaluation.version} development evaluation: ${manifest.evaluation.games} games, mean ${manifest.evaluation.mean.toFixed(2)} food, median ${manifest.evaluation.median}. One training seed. `+(visual?'The matched frozen control scored zero; these eight previously used development games are preliminary evidence. Final useful-play targets and full-network causal validation remain incomplete.':'Internal correction improved mean; the five-food fraction and expert-prediction accuracy regressed.');
  // Collection-specific provenance and sampling notes are plain text only.
  for(const [id,key] of [['notesModel','notes_model'],['modelLabel','model_label'],['modelSummary','model_summary'],['notesTitle','notes_title'],['notesActivity','notes_activity'],['evaluationNotes','evaluation_notes']]){
    const text=manifest.presentation?.[key];if(typeof text==='string')$(id).textContent=text;
  }
  // Preserve the source notes verbatim in the methods disclosure, then explain them.
  for(const [source,target] of [['notesModel','sourceModelNotes'],['notesActivity','sourceActivityNotes'],['evaluationNotes','sourceEvaluationNotes']])$(target).textContent=$(source).textContent;
  $('notesTitle').textContent='How this simulation works';
  $('notesModel').textContent=visual?'A learned visual encoder turns two consecutive grayscale board images into five numerical estimates of food direction and nearby obstacles. These estimates enter the neural network through a learned input mapping. An action readout converts network activity into a left turn, straight move, or right turn. The controller was trained using demonstrations; no teacher chooses its actions during these games.':'The model receives five numbers describing food direction and nearby obstacles. A learned input mapping feeds these values into the neural network, and an action readout converts its activity into a left turn, straight move, or right turn.';
  $('notesAnatomy').textContent=`This recording includes ${c.simulated.toLocaleString()} simulated neurons. ${c.simulated_positioned.toLocaleString()} have mapped 3D locations; the other ${c.simulated_unpositioned.toLocaleString()} remain available in the activity grid and cell explorer.`;
  $('notesActivity').textContent=`The website plays back saved activity; it does not run or train the model. Each cell has a signed signal between −1 and +1, relative to the model’s zero reference. This is a continuously valued model state, not a count of electrical spikes. `+(manifest.signals.substeps?'The sample badge identifies the available recording interval.':'The recordings contain one neural snapshot every 100 milliseconds (10 snapshots per second).')+' The viewer holds each measured value until the next snapshot.';
  $('notesChange').textContent='Population activity colours cells by their annotated group and uses brightness to show signal magnitude. Activity change compares the signal with up to 100 milliseconds earlier: warm colours mean it rose; cool colours mean it fell. Connection lines show anatomical links, not signals travelling between cells.';
  const evaluation=manifest.evaluation;
  $('evaluationNotes').textContent=`Across ${evaluation.games} development games, the controller collected ${evaluation.mean.toFixed(2)} food on average, with a median of ${evaluation.median}. `+(typeof evaluation.at_least_five==='number'?`${(evaluation.at_least_five*100).toFixed(2)}% of games reached at least five food. `:'')+'The recordings in this viewer are selected examples. These results come from one training run, rather than independent repetitions of the experiment.';
  $('notesContrast').textContent='The logarithmic colour scale and contrast slider make small signals easier to see. They leave the recorded numbers unchanged. Signed values remain available in the cell explorer; a negative signal is separate from the cell’s assigned excitatory or inhibitory effect on its connections.';
  $('provenance').replaceChildren(...[['Checkpoint SHA-256',manifest.checkpoint_sha256],['Anatomy identity',manifest.anatomy_identity],['Activity',manifest.notes],['Geometry','141,000 soma/root overview positions; schematic straight connections'],['Input mapping',visual?'Recorded grayscale frame pair → frozen visual encoder → five estimated values → frozen 5 → 256 projection.':'Learned dense 5 → 256 projection. No biological food/obstacle cell assignment.']].flatMap(([k,v])=>[node('dt',k),node('dd',v)]));
  loader=createLoader(manifest,({record,activity,substeps})=>{if(generation!==collectionGeneration)return;replay=new Replay(record,activity,substeps);loading=false;$('timeline').max=String(replay.length);enableControls(true);setPlaying(false);refreshList();dirty=true;},fetch);
  refreshList();selectNeuron(null);await loadGame(manifest.default_game);
  }catch(error){if(generation===collectionGeneration)throw error;}
}
async function init(){
  const response=await fetch('data/collections.json');
  collections=response.ok?await response.json():{format:'snake-viewer-collections-v1',default_collection:'reduced-v74',collections:[{id:'reduced-v74',label:'Reduced network · V74',file:'data/manifest.json'}]};
  if(collections.format!=='snake-viewer-collections-v1')throw Error('Unsupported collection catalog.');
  $('collection').replaceChildren(...collections.collections.map(c=>{const o=node('option',c.label);o.value=c.id;return o;}));$('collection').value=collections.default_collection;$('collection').disabled=false;
  const multipleCollections=collections.collections.length>1;
  $('collectionField').hidden=!multipleCollections;
  $('collectionField').closest('.record-select').classList.toggle('has-collections',multipleCollections);
  initControls();requestAnimationFrame(animate);await loadCollection(collections.default_collection);
}

const modelContext=typeof document==='undefined'?null:document.modelContext;
if(modelContext?.registerTool){
  const lifecycle=new AbortController();
  addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  const describe=()=>({collection:$('collection').value,recording:replay?.record.id??null,decision:replay?.index??0,decisions:replay?.length??0,playing:replay?.playing??false});
  try{Promise.resolve(modelContext.registerTool({name:'inspect_replay',title:'Inspect the current replay',description:'Read the currently selected recording and playback position.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:describe},{signal:lifecycle.signal})).catch(()=>{});
  Promise.resolve(modelContext.registerTool({name:'seek_replay',title:'Go to a recorded decision',description:'Pause the selected replay and show a completed decision. This changes playback only.',inputSchema:{type:'object',properties:{decision:{type:'integer',minimum:0}},required:['decision'],additionalProperties:false},execute:async input=>{if(loading||!replay||!Number.isInteger(input.decision)||input.decision<0||input.decision>replay.length)throw Error('Choose a valid decision in a loaded replay.');setPlaying(false);replay.seek(input.decision);dirty=true;paint();return describe();}},{signal:lifecycle.signal})).catch(()=>{});}catch{}
}
init().catch(fail);

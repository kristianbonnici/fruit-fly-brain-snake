import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

export function createBrain(mount,anatomy,positions,onSelect) {
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0xffffff,0);
  renderer.domElement.setAttribute('aria-hidden','true'); mount.append(renderer.domElement);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.01,100);
  const controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=.09;controls.minDistance=.3;controls.maxDistance=28;controls.autoRotate=false;
  const backdropGeometry=new THREE.BufferGeometry();backdropGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  const backdrop=new THREE.Points(backdropGeometry,new THREE.PointsMaterial({size:.012,color:0x8398a7,transparent:true,opacity:.08,depthWrite:false}));scene.add(backdrop);
  const mapped=anatomy.nodes.map((n,i)=>n.position?i:null).filter(i=>i!==null);
  const activePositions=new Float32Array(mapped.flatMap(i=>anatomy.nodes[i].position));
  const colors=new Float32Array(activePositions.length),signals=new Float32Array(mapped.length),highlighted=new Float32Array(mapped.length).fill(1);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(activePositions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  geometry.setAttribute('recordedActivity',new THREE.BufferAttribute(signals,1).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute('highlightMask',new THREE.BufferAttribute(highlighted,1).setUsage(THREE.DynamicDrawUsage));
  const populationColors=Object.fromEntries(anatomy.populations.map(p=>[p.key,new THREE.Color(p.color)]));
  mapped.forEach((node,k)=>{const c=populationColors[anatomy.nodes[node].population];colors[k*3]=c.r;colors[k*3+1]=c.g;colors[k*3+2]=c.b;});
  const displayUniforms={activityContrast:{value:2.5},logScale:{value:1},changeView:{value:0},focusEnabled:{value:0},riseColor:{value:new THREE.Color('#d5742b')},fallColor:{value:new THREE.Color('#18899b')}};
  const activityMaterial=new THREE.PointsMaterial({size:anatomy.nodes.length>4096?.028:.046,vertexColors:true,transparent:true,opacity:.8,depthWrite:false});
  // Population mode uses recorded magnitude; change mode uses the signed 100 ms difference.
  // The selected signal controls opacity and size;
  // quiet anatomy lives in its own faint layer, not in opaque grey activity dots.
  activityMaterial.onBeforeCompile=shader=>{
    Object.assign(shader.uniforms,displayUniforms);
    shader.vertexShader='attribute float recordedActivity; attribute float highlightMask; uniform float focusEnabled; uniform float activityContrast; uniform float logScale; uniform float changeView; varying float signalIntensity; varying float rising; varying float emphasis;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('void main() {',`void main() {
      float magnitude = clamp(abs(recordedActivity) / mix(1.0, 2.0, changeView), 0.0, 1.0);
      rising = step(0.0, recordedActivity);
      emphasis = mix(1.0, mix(0.08, 1.0, highlightMask), focusEnabled);
      float displayed = logScale > 0.5 ? log(1.0 + magnitude * 1000000.0) / log(1000001.0) : magnitude;
      signalIntensity = pow(displayed, 1.0 / activityContrast);`);
    shader.vertexShader=shader.vertexShader.replace('gl_PointSize = size;', 'gl_PointSize = size * mix(0.45, 1.0, signalIntensity) * mix(0.7, 1.0, emphasis);');
    shader.fragmentShader='uniform float changeView; uniform vec3 riseColor; uniform vec3 fallColor; varying float signalIntensity; varying float rising; varying float emphasis;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
      float radius = length(gl_PointCoord - vec2(0.5));
      if (radius > 0.5) discard;
      if (changeView > 0.5) diffuseColor.rgb = mix(fallColor, riseColor, rising);
      diffuseColor.a *= signalIntensity * emphasis * (1.0 - smoothstep(0.3, 0.5, radius));`);
  };
  activityMaterial.customProgramCacheKey=()=> 'recorded-activity-highlights-v3';
  // Neutral outlines identify the chosen anatomy even when measured activity is zero.
  const highlightMaterial=new THREE.PointsMaterial({size:activityMaterial.size*1.16,color:0x546778,transparent:true,opacity:.16,depthWrite:false});
  highlightMaterial.onBeforeCompile=shader=>{
    shader.vertexShader='attribute float highlightMask; varying float highlighted;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('void main() {','void main() { highlighted = highlightMask;');
    shader.fragmentShader='varying float highlighted;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>',`#include <color_fragment>
      float radius = length(gl_PointCoord - vec2(0.5));
      if (highlighted < 0.5 || radius > 0.5 || radius < 0.32) discard;
      diffuseColor.a *= smoothstep(0.32, 0.38, radius) * (1.0 - smoothstep(0.44, 0.5, radius));`);
  };
  highlightMaterial.customProgramCacheKey=()=> 'anatomy-highlight-outline-v1';
  const highlightPoints=new THREE.Points(geometry,highlightMaterial);highlightPoints.visible=false;scene.add(highlightPoints);
  const activityPoints=new THREE.Points(geometry,activityMaterial);scene.add(activityPoints);
  const ring=Array.from({length:40},(_,i)=>new THREE.Vector3(.020*Math.cos(i*Math.PI/20),.020*Math.sin(i*Math.PI/20),0));
  const selection=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ring),new THREE.LineBasicMaterial({color:0x243d55,transparent:true,opacity:.85,depthTest:false}));selection.visible=false;selection.renderOrder=2;scene.add(selection);
  const recipients=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:.053,color:0x0071e3,transparent:true,opacity:.4,depthWrite:false}));recipients.visible=false;scene.add(recipients);
  const edgesGroup=new THREE.Group();scene.add(edgesGroup);
  const raycaster=new THREE.Raycaster();raycaster.params.Points.threshold=.045;
  let selected=null,preset='brain',down=null,currentHighlight=null;
  const fullBounds=new THREE.Box3().setFromBufferAttribute(backdropGeometry.attributes.position);
  const brainBounds=new THREE.Box3();
  // Fit brain populations in the preserved common coordinate frame; full CNS
  // remains present, and the expanded preset fits every available position.
  for(const i of mapped)if(['ol_intrinsic','visual_projection','cb_intrinsic','visual_centrifugal','descending_neuron'].includes(anatomy.nodes[i].population))brainBounds.expandByPoint(new THREE.Vector3(...anatomy.nodes[i].position));
  function fit(bounds){
    const size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
    const span=Math.max(size.y,size.x/camera.aspect);
    const z=span/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*1.23+size.z*.35;
    controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(0,0,z));controls.update();
  }
  function setPreset(value){preset=value;fit(value==='cns'?fullBounds:brainBounds);}
  let viewportAspect=null;
  function resize(){
    const w=mount.clientWidth,h=mount.clientHeight;if(!w||!h)return;
    const aspect=w/h;
    if(viewportAspect!==null){
      // Preserve orbit, pan and relative zoom as the responsive panel changes.
      const size=(preset==='cns'?fullBounds:brainBounds).getSize(new THREE.Vector3());
      const ratio=Math.max(size.y,size.x/aspect)/Math.max(size.y,size.x/viewportAspect);
      camera.position.sub(controls.target).multiplyScalar(ratio).add(controls.target);
    }
    viewportAspect=aspect;renderer.setSize(w,h);camera.aspect=aspect;camera.updateProjectionMatrix();
  }
  const observer=new ResizeObserver(resize);observer.observe(mount);resize();setPreset('brain');
  function update(values,contrast=2.5,scale='log',view='population'){
    displayUniforms.activityContrast.value=Math.max(1,Math.min(4,contrast));displayUniforms.logScale.value=scale==='log'?1:0;
    displayUniforms.changeView.value=view==='change'?1:0;
    for(let k=0;k<mapped.length;k++)signals[k]=values[mapped[k]];
    geometry.attributes.recordedActivity.needsUpdate=true;
  }
  function select(node){selected=node;const p=anatomy.nodes[node]?.position;selection.visible=!!p;if(p)selection.position.set(...p);}
  function highlight(mask){
    currentHighlight=mask;highlightPoints.visible=!!mask;displayUniforms.focusEnabled.value=mask?1:0;
    backdrop.material.opacity=mask ? .03 : .08;
    for(let k=0;k<mapped.length;k++)highlighted[k]=mask?mask[mapped[k]]:1;
    geometry.attributes.highlightMask.needsUpdate=true;
  }
  function focus(){const p=anatomy.nodes[selected]?.position;if(!p)return;const direction=camera.position.clone().sub(controls.target).normalize();controls.target.set(...p);camera.position.copy(controls.target).add(direction.multiplyScalar(1.3));controls.update();}
  function showRecipients(sensor){
    recipients.visible=sensor!==null;
    if(sensor===null)return;
    const coords=anatomy.sensory_nodes.filter(i=>anatomy.nodes[i].position && anatomy.encoder_weights[anatomy.nodes[i].sensory_slot][sensor]!==0).flatMap(i=>anatomy.nodes[i].position);
    recipients.geometry.dispose();recipients.geometry=new THREE.BufferGeometry();recipients.geometry.setAttribute('position',new THREE.Float32BufferAttribute(coords,3));
  }
  function connections(edges){
    while(edgesGroup.children.length){const item=edgesGroup.children[0];edgesGroup.remove(item);item.geometry?.dispose();item.material?.dispose();}
    const lines=[],lineColors=[];const color=new THREE.Color();
    for(const e of edges){
      const a=anatomy.nodes[e[0]].position,b=anatomy.nodes[e[1]].position;if(!a||!b||e[0]===e[1])continue;
      color.set(e[0]===selected?'#bc762c':'#3978b8');lines.push(...a,...b);lineColors.push(color.r,color.g,color.b,color.r,color.g,color.b);
      const av=new THREE.Vector3(...a),bv=new THREE.Vector3(...b),direction=bv.clone().sub(av).normalize();
      const cone=new THREE.Mesh(new THREE.ConeGeometry(.016,.055,6),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.7,depthWrite:false}));
      cone.position.copy(av.lerp(bv,.72));cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction);edgesGroup.add(cone);
    }
    if(lines.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));g.setAttribute('color',new THREE.Float32BufferAttribute(lineColors,3));edgesGroup.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.35,depthWrite:false})));}
  }
  renderer.domElement.addEventListener('pointerdown',e=>{down=e.isPrimary&&e.button===0?[e.clientX,e.clientY]:null;});
  renderer.domElement.addEventListener('pointercancel',()=>{down=null;});
  renderer.domElement.addEventListener('pointerup',e=>{
    const start=down;down=null;if(!start||!e.isPrimary||e.button!==0||Math.hypot(e.clientX-start[0],e.clientY-start[1])>5)return;
    const r=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);
    const hits=raycaster.intersectObject(activityPoints),hit=currentHighlight?hits.find(h=>currentHighlight[mapped[h.index]])??hits[0]:hits[0];onSelect(hit?mapped[hit.index]:null);
  });
  const keyHandler=e=>{
    if(e.key==='Escape'){e.preventDefault();onSelect(null);return;}
    if(e.target!==mount)return;
    const offset=camera.position.clone().sub(controls.target),spherical=new THREE.Spherical().setFromVector3(offset);
    if(e.key==='ArrowLeft')spherical.theta-=.1;else if(e.key==='ArrowRight')spherical.theta+=.1;else if(e.key==='ArrowUp')spherical.phi=Math.max(.05,spherical.phi-.1);else if(e.key==='ArrowDown')spherical.phi=Math.min(Math.PI-.05,spherical.phi+.1);else if(e.key==='+'||e.key==='=')spherical.radius=Math.max(.3,spherical.radius*.9);else if(e.key==='-')spherical.radius=Math.min(28,spherical.radius*1.1);else return;
    e.preventDefault();camera.position.copy(controls.target).add(offset.setFromSpherical(spherical));controls.update();
  };
  mount.addEventListener('keydown',keyHandler);
  return {update,highlight,select,focus,showRecipients,connections,setPreset,reset:()=>setPreset(preset),
    render:()=>{controls.update();selection.quaternion.copy(camera.quaternion);renderer.render(scene,camera);},
    dispose:()=>{observer.disconnect();mount.removeEventListener('keydown',keyHandler);controls.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});renderer.dispose();renderer.domElement.remove();},
    cameraState:()=>({position:camera.position.toArray(),target:controls.target.toArray(),preset})};
}

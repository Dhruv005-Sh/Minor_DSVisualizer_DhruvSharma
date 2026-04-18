'use strict';
/**
 * DS Visualizer v3 — main.js (Tree & Graph Edition)
 * ===================================================
 * Data Structures : Tree, Graph
 * Tree Algorithms : BFS, DFS, Inorder/Preorder/Postorder Traversal
 * Graph Algorithms: BFS, DFS, Kruskal MST
 * ML Models       : KNN Regressor (Smart Insert), Linear Regression (Complexity)
 *
 * VISUALIZATION ONLY — all DS and ML logic lives in Python backend.
 */

const API    = 'http://localhost:5001';
const NODE_R = 24;

// Color palette
const C = {
  purple:    '#a855f7',
  purpleL:   '#c084fc',
  emerald:   '#10b981',
  amber:     '#f59e0b',
  rose:      '#f43f5e',
  sky:       '#38bdf8',
  text:      '#f0ebff',
  muted:     '#7c6fa0',
  nodeFill:  'rgba(168,85,247,0.12)',
  nodeHL:    'rgba(245,158,11,0.30)',
  nodeFound: 'rgba(16,185,129,0.30)',
  nodeDFS:   'rgba(56,189,248,0.28)',
  nodeMST:   'rgba(16,185,129,0.22)',
  edgeColor: 'rgba(168,85,247,0.42)',
  mstEdge:   'rgba(16,185,129,0.75)',
};

// ── UI state ──────────────────────────────────────────────
let curDS       = 'graphs';   // start on Graphs
let animRunning = false;
let animSpeed   = 500;

// Live Python state (JS reads, never mutates logic)
let pyState = {
  tree:  { tree: null, size: 0 },
  graph: { nodes: [1,2,3,4,5], edges:[[1,2,8],[1,3,5],[2,4,12],[3,4,7],[4,5,3]], size:5, next_id:6 },
};

// Algorithm animation state
let algoHighlight   = [];     // nodes currently highlighted by an algo step
let algoMSTEdges    = [];     // MST edges found so far (for Kruskal animation)
let treeSearchPath  = [];     // search highlight path in tree

// Graph visual state (positions & selection) — purely for rendering
let graphPos   = {};
let gSelected  = null;
let gDragging  = null;
let gDragOff   = { x:0, y:0 };
let gDragMoved = false;
let hoveredNode = null;
let treePos = {}; // store computed screen positions for tree nodes

// ML op tracking
let opHistory = [];
let opCount   = 0;

// ═══════════════════════════════════════════════
// API HELPERS
// ═══════════════════════════════════════════════
async function pyPost(endpoint, payload = {}) {
  try {
    const r = await fetch(API + endpoint, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload), signal: AbortSignal.timeout(5000),
    });
    return await r.json();
  } catch {
    return { ok:false, error:'Cannot reach Python backend — run: python app.py' };
  }
}

async function pyGet(endpoint) {
  try {
    const r = await fetch(API + endpoint, { signal: AbortSignal.timeout(4000) });
    return await r.json();
  } catch { return null; }
}

function applyResponse(res, dsKey) {
  if (!res) return;
  if (res.state) pyState[dsKey] = res.state;
  draw();
  updateStatePanel(dsKey);
}

// ═══════════════════════════════════════════════
// CANVAS SETUP
// ═══════════════════════════════════════════════
const canvas = document.getElementById('vizCanvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth - 24, H = wrap.clientHeight - 24;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W; canvas.height = H; initGraphPos();
  }
}

function initGraphPos() {
  const nodes = pyState.graph.nodes || [];
  const W = canvas.width, H = canvas.height, n = nodes.length;
  if (!n) return;
  const rad = Math.min(W, H) * 0.33;
  nodes.forEach((nd, i) => {
    if (!graphPos[nd]) {
      const a = (2 * Math.PI * i / n) - Math.PI / 2;
      graphPos[nd] = { x: W/2 + rad * Math.cos(a), y: H/2 + rad * Math.sin(a) };
    }
  });
}

resizeCanvas();
window.addEventListener('resize', () => { resizeCanvas(); draw(); });

// ═══════════════════════════════════════════════
// DRAWING PRIMITIVES
// ═══════════════════════════════════════════════
function rrect(x,y,w,h,r,fill,stroke,lw=1.8){
  ctx.beginPath(); ctx.roundRect(x,y,w,h,r);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke();}
}
function circ(x,y,r,fill,stroke,lw=2){
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw;ctx.stroke();}
}
function lbl(t,x,y,color,size,align='center'){
  ctx.fillStyle=color; ctx.font=`600 ${size}px 'JetBrains Mono',monospace`;
  ctx.textAlign=align; ctx.textBaseline='middle'; ctx.fillText(String(t),x,y);
}
function emptyMsg(msg){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  lbl(msg, canvas.width/2, canvas.height/2, C.muted, 13);
}

// ═══════════════════════════════════════════════
// DRAW DISPATCHER
// ═══════════════════════════════════════════════
function draw(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (curDS === 'trees')  drawTree(canvas.width, canvas.height);
  if (curDS === 'graphs') drawGraph(canvas.width, canvas.height);
  updateBadge();
}

// ── TREE Drawing ───────────────────────────────
function drawTree(W, H){
  const root = pyState.tree.tree;
  if(!root){ emptyMsg('Empty Tree — Add Node to start'); return; }
  lbl('BINARY TREE', W/2, 18, C.muted, 11);

  const xPos={}, yPos={};
  let ctr = 0;
  function inOrder(n, d){ if(!n)return; inOrder(n.left, d+1); xPos[n.val]=ctr++; yPos[n.val]=d; inOrder(n.right, d+1); }
  inOrder(root, 0);

  const n   = ctr;
  const sx  = n > 1 ? Math.min(72, (W-100)/(n-1)) : 0;
  const ox  = (W - (n>1?(n-1)*sx:0)) / 2;
  const yvs = 80;

  function scr(v){ return { x: ox + xPos[v]*sx, y: yvs + yPos[v]*80 }; }

  const algoSet  = new Set(algoHighlight);
  const pathSet  = new Set(treeSearchPath);
  const lastAlgo = algoHighlight.length > 0 ? algoHighlight[algoHighlight.length-1] : null;
  const lastPath = treeSearchPath.length > 0 ? treeSearchPath[treeSearchPath.length-1] : null;

  // Draw edges
  function drawEdges(node){
    if(!node) return;
    const ps = scr(node.val);
    [node.left, node.right].forEach(c => {
      if(!c) return;
      const pc = scr(c.val);
      // Check if this edge is on the search path
      const pathIdx = treeSearchPath.indexOf(c.val);
      const onPath  = pathIdx > 0 && treeSearchPath[pathIdx-1] === node.val;
      ctx.strokeStyle = onPath ? C.amber : C.edgeColor;
      ctx.lineWidth   = onPath ? 2.5 : 1.5;
      ctx.beginPath(); ctx.moveTo(ps.x, ps.y + NODE_R); ctx.lineTo(pc.x, pc.y - NODE_R); ctx.stroke();
      drawEdges(c);
    });
  }
  drawEdges(root);

  // Draw nodes
  function drawNodes(node){
    if(!node) return;
    const p = scr(node.val);
    const isLastAlgo = node.val === lastAlgo;
    const inAlgo     = algoSet.has(node.val) && !isLastAlgo;
    const isLastPath = node.val === lastPath && treeSearchPath.length > 0;
    const inPath     = pathSet.has(node.val) && !isLastPath;

    let fill  = C.nodeFill;
    let stroke = C.purple;
    let textCol = C.text;

    if(isLastAlgo || isLastPath){ fill = C.nodeFound; stroke = C.emerald; textCol = C.emerald; }
    else if(inAlgo)              { fill = C.nodeDFS;  stroke = C.sky;     textCol = C.sky; }
    if(isLastAlgo || isLastPath){
      ctx.shadowColor = C.emerald; ctx.shadowBlur = 18;
    }
    
    let rad = NODE_R;
    if(String(node.val) === String(hoveredNode)){
       ctx.shadowColor = stroke; ctx.shadowBlur = 20;
       rad = NODE_R + 6;
       fill = C.nodeHL;
       textCol = C.text;
    }
    
    circ(p.x, p.y, rad, fill, stroke);
    ctx.shadowBlur = 0;
    lbl(node.val, p.x, p.y, textCol, 14);
    treePos[node.val] = p; // store for hit test
    drawNodes(node.left); drawNodes(node.right);
  }
  drawNodes(root);
}

// ── GRAPH Drawing ──────────────────────────────
function drawGraph(W, H){
  const g = pyState.graph;
  if(!g.nodes || !g.nodes.length){ emptyMsg('Click canvas to add node 1'); return; }
  initGraphPos();
  lbl('UNDIRECTED WEIGHTED GRAPH', W/2, 18, C.muted, 11);

  const algoSet = new Set(algoHighlight);
  const mstSet  = new Set(algoMSTEdges.map(e => `${Math.min(e[0],e[1])}-${Math.max(e[0],e[1])}`));

  // Draw all edges
  g.edges.forEach(([a, b, w]) => {
    const pa = graphPos[a], pb = graphPos[b];
    if(!pa || !pb) return;
    const key = `${Math.min(a,b)}-${Math.max(a,b)}`;
    const isMST = mstSet.has(key);

    ctx.strokeStyle = isMST ? C.mstEdge : C.edgeColor;
    ctx.lineWidth   = isMST ? 3.2 : 1.8;
    if(isMST){ ctx.shadowColor = C.emerald; ctx.shadowBlur = 8; }
    ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw weight label at midpoint
    const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
    const weightColor = isMST ? C.emerald : C.muted;
    // Small background pill
    ctx.save();
    ctx.fillStyle = 'rgba(15,10,30,0.82)';
    ctx.beginPath();
    ctx.roundRect(mx-9, my-8, 18, 16, 4);
    ctx.fill();
    ctx.restore();
    lbl(w, mx, my, weightColor, 10);
  });

  // Selection ring
  if(gSelected != null && graphPos[gSelected]){
    const p = graphPos[gSelected];
    ctx.strokeStyle = C.amber; ctx.lineWidth = 2; ctx.setLineDash([6,4]);
    ctx.beginPath(); ctx.arc(p.x, p.y, NODE_R+9, 0, Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw nodes
  g.nodes.forEach(nd => {
    const p = graphPos[nd]; if(!p) return;
    const sel     = nd === gSelected;
    const isLast  = algoHighlight.length > 0 && nd === algoHighlight[algoHighlight.length-1];
    const inAlgo  = algoSet.has(nd) && !isLast;

    let fill   = sel ? C.nodeHL   : C.nodeFill;
    let stroke = sel ? C.amber    : C.purple;
    let textCol = C.text;
    if(isLast)       { fill = C.nodeFound; stroke = C.emerald; textCol = C.emerald; ctx.shadowColor = C.emerald; ctx.shadowBlur = 18; }
    else if(inAlgo)  { fill = C.nodeDFS;   stroke = C.sky;     textCol = C.sky; }

    let rad = NODE_R;
    if(String(nd) === String(hoveredNode)){
       ctx.shadowColor = stroke; ctx.shadowBlur = 20;
       rad = NODE_R + 6;
       fill = C.nodeHL;
       textCol = C.text;
    }

    circ(p.x, p.y, rad, fill, stroke);
    ctx.shadowBlur = 0;
    lbl(nd, p.x, p.y, textCol, 14);
  });

  if(gSelected != null)
    lbl(`Node ${gSelected} selected — click another to connect`, W/2, H-18, C.amber, 11);
}

// ═══════════════════════════════════════════════
// GRAPH MOUSE EVENTS
// ═══════════════════════════════════════════════
function hitNode(mx, my){
  for(const nd of (pyState.graph.nodes || [])){
    const p = graphPos[nd];
    if(p && Math.hypot(mx-p.x, my-p.y) <= NODE_R) return nd;
  }
  return null;
}
function getMouse(e){ const r = canvas.getBoundingClientRect(); return { mx:e.clientX-r.left, my:e.clientY-r.top }; }

function getHoveredTarget(mx, my){
  if(curDS === 'graphs'){ return hitNode(mx, my); }
  if(curDS === 'trees'){
    for(const [val, p] of Object.entries(treePos)){
      if(Math.hypot(mx-p.x, my-p.y) <= (NODE_R+6)) return val;
    }
  }
  return null;
}

canvas.addEventListener('mousedown', e => {
  if(curDS !== 'graphs') return;
  const { mx, my } = getMouse(e);
  const hit = hitNode(mx, my);
  if(hit != null){ gDragging=hit; gDragMoved=false; gDragOff={x:mx-graphPos[hit].x, y:my-graphPos[hit].y}; canvas.style.cursor='grabbing'; }
});
canvas.addEventListener('mousemove', e => {
  const { mx, my } = getMouse(e);
  
  if(curDS === 'graphs'){
    if(gDragging != null){ gDragMoved=true; graphPos[gDragging]={x:mx-gDragOff.x, y:my-gDragOff.y}; draw(); return; }
    canvas.style.cursor = hitNode(mx,my) != null ? 'grab' : 'crosshair';
  } else {
    canvas.style.cursor = getHoveredTarget(mx,my) != null ? 'pointer' : 'default';
  }

  // Hover detection for both Tree and Graph
  const hit = getHoveredTarget(mx, my);
  if(String(hit) !== String(hoveredNode)){
    hoveredNode = hit;
    draw();
  }
});
canvas.addEventListener('mouseup', e => {
  if(curDS !== 'graphs') return;
  const wasMoved = gDragMoved;
  gDragging=null; gDragMoved=false; canvas.style.cursor='crosshair';
  if(wasMoved){ draw(); return; }
  handleGraphClick(e);
});

async function handleGraphClick(e){
  const { mx, my } = getMouse(e);
  const hit = hitNode(mx, my);
  if(hit != null){
    if(gSelected == null){
      gSelected=hit; logOp('selected', hit); draw();
    } else if(gSelected === hit){
      gSelected=null; draw();
    } else {
      // Connect two nodes — Python assigns random weight if not given
      const res = await pyPost('/ds/graph/add_edge', { node_a:gSelected, node_b:hit });
      if(res.ok){
        const w = res.edge?.[2] ?? '?';
        logOp('edge', `${gSelected}—${hit} (w=${w})`, 'lg');
        setStatus(`Edge ${gSelected}—${hit} added (weight=${w})`, 'status-ok');
        applyResponse(res, 'graph');
        await runML(null);
      } else {
        logOp('err', res.error, 'le');
        setStatus(res.error, 'status-err');
      }
      gSelected = null; draw();
    }
  } else {
    if(gSelected != null){ gSelected=null; draw(); return; }
    const res = await pyPost('/ds/graph/add_node', {});
    if(res.ok){
      const nid = res.node;
      graphPos[nid] = { x:mx, y:my };
      logOp('add node', nid, 'lg');
      setStatus('Node '+nid+' added at cursor', 'status-ok');
      applyResponse(res, 'graph');
      await runML(nid);
    } else {
      setStatus(res.error, 'status-err');
    }
  }
}

// ═══════════════════════════════════════════════
// BADGE & PANELS
// ═══════════════════════════════════════════════
function updateBadge(){
  const size = curDS==='trees' ? (pyState.tree.size||0) : (pyState.graph.size||0);
  document.querySelector('.badge').innerHTML =
    `<div class="badge-dot"></div>size: ${size} · KNN + Linear Regression`;
}

function updateStatePanel(dsKey){
  const box = document.getElementById('stateBox');
  const s   = pyState[dsKey] || {};
  let h = '';
  if(dsKey === 'tree'){
    h = `<div class="state-row"><span class="state-key">size</span><span class="state-val">${s.size??0}</span></div>
         <div class="state-row"><span class="state-key">root</span><span class="state-val">${s.tree ? s.tree.val : 'null'}</span></div>`;
  } else if(dsKey === 'graph'){
    h = `<div class="state-row"><span class="state-key">nodes</span><span class="state-val">[${(s.nodes||[]).join(', ')}]</span></div>
         <div class="state-row"><span class="state-key">edges</span><span class="state-val">${(s.edges||[]).length}</span></div>
         <div class="state-row"><span class="state-key">size</span><span class="state-val">${s.size??0}</span></div>
         <div class="state-row"><span class="state-key">next_id</span><span class="state-val">${s.next_id??'-'}</span></div>`;
  }
  box.innerHTML = h;
}

// ═══════════════════════════════════════════════
// ML RENDERERS
// ═══════════════════════════════════════════════

/** Run KNN + Linear Regression after every operation */
async function runML(newValue) {
  opCount++;
  const g = pyState.graph;
  const t = pyState.tree;

  // Build inputs based on current DS
  let array=[], nNodes=0, nEdges=0, depth=1;
  if(curDS === 'trees'){
    const collect = (n) => n ? [n.val, ...collect(n.left), ...collect(n.right)] : [];
    array  = collect(t.tree);
    nNodes = t.size || 0;
    nEdges = Math.max(0, nNodes - 1);  // binary tree: n-1 edges
    depth  = Math.ceil(Math.log2(nNodes+1));
  } else {
    array  = [...(g.nodes||[])];
    nNodes = g.size || 0;
    nEdges = (g.edges||[]).length;
    depth  = Math.max(1, Math.ceil(Math.log2(nNodes+1)));
  }

  const val = (newValue !== null && newValue !== undefined && !isNaN(Number(newValue))) ? Number(newValue) : 50;

  const [knnRes, lrRes] = await Promise.all([
    pyPost('/ml/knn-insert',        { value:val, array }),
    pyPost('/ml/linear-complexity', { n_nodes:nNodes, n_edges:nEdges, depth, op_count:opCount }),
  ]);

  renderKNN(knnRes, val);
  renderLinearRegression(lrRes);
}

function renderKNN(d, val){
  const el = document.getElementById('knn-result');
  if(!d || d.suggested_index === undefined){
    el.innerHTML = '<div class="ml-placeholder">Run an op to activate</div>'; return;
  }
  const sr = d.sorted_ratio ?? 0;
  el.innerHTML = `
    <div class="insert-idx-display">
      <div class="insert-idx-label">Suggested Insert Position</div>
      <div class="insert-idx-val">[${d.suggested_index}]</div>
    </div>
    <div class="insert-reasoning">${d.reasoning}</div>
    <div class="sorted-bar" style="margin-top:.65rem">
      <div class="sorted-bar-label"><span>Sorted Ratio</span><span>${sr}%</span></div>
      <div class="sorted-track"><div class="sorted-fill" style="width:${sr}%"></div></div>
    </div>
    <div class="knn-badge">🔵 KNN · ${d.neighbors_used ?? 7} neighbors</div>`;
}

function renderLinearRegression(d){
  const el = document.getElementById('lr-result');
  if(!d || !d.rating){
    el.innerHTML = '<div class="ml-placeholder">Run an op to activate</div>'; return;
  }
  const colorMap = { emerald:'#10b981', cyan:'#06b6d4', amber:'#f59e0b', rose:'#f43f5e' };
  const col = colorMap[d.color] || '#c084fc';
  const pct = Math.min(100, Math.round((d.predicted_steps / Math.max(d.predicted_steps, 50)) * 100));
  el.innerHTML = `
    <div class="complexity-main">
      <div class="complexity-badge" style="background:${col}22;color:${col};border:1px solid ${col}44">${d.rating}</div>
      <div class="complexity-steps">Predicted steps: <strong style="color:${col}">${d.predicted_steps}</strong> &nbsp;·&nbsp; Theoretical min: <strong>${d.theoretical_min}</strong></div>
    </div>
    <div class="linreg-bar-wrap">
      <div class="linreg-bar-label"><span>Predicted Complexity</span><span style="color:${col}">${pct}%</span></div>
      <div class="linreg-bar-track"><div class="linreg-bar-fill" style="width:${pct}%;background:${col}"></div></div>
    </div>
    <div class="complexity-grid">
      <div class="complexity-cell"><div class="complexity-cell-label">Nodes</div><div class="complexity-cell-val">${d.inputs?.nodes??0}</div></div>
      <div class="complexity-cell"><div class="complexity-cell-label">Edges</div><div class="complexity-cell-val">${d.inputs?.edges??0}</div></div>
    </div>
    <div class="complexity-formula">📈 ${d.formula_note}</div>`;
}

// ═══════════════════════════════════════════════
// ALGO RESULT DISPLAY
// ═══════════════════════════════════════════════
function showAlgoResult(label, items, isEdges = false){
  const box = document.getElementById('algoResultBox');
  const tagged = items.map(item => {
    if(isEdges){
      return `<span class="algo-mst-edge">${item[0]}—${item[1]} (w=${item[2]})</span>`;
    }
    return `<span class="algo-node">${item}</span>`;
  }).join('');
  box.innerHTML = `<span style="color:var(--muted);font-size:.62rem">${label}:&nbsp;</span>${tagged}`;
}

function showAlgoBanner(msg){
  const b = document.getElementById('algoBanner');
  b.textContent = msg;
  b.style.display = 'block';
  clearTimeout(b._hide);
  b._hide = setTimeout(() => { b.style.display='none'; }, 3500);
}

// ═══════════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════════
const sleep = ms => new Promise(r => setTimeout(r, ms));

function getInputVal(){
  const v = parseInt(document.getElementById('valInput').value);
  return isNaN(v) ? Math.floor(Math.random()*89)+10 : Math.max(1, Math.min(99,v));
}

function logOp(type, val, cls='lc'){
  const el = document.getElementById('opLog');
  const d  = document.createElement('div');
  d.innerHTML = cls==='le'
    ? `<span class="le">✗ ${val}</span>`
    : `<span class="${cls}">${type}</span>(<span class="lv">${val}</span>)`;
  el.appendChild(d); el.scrollTop = el.scrollHeight;
}

function setStatus(msg, cls='status-info'){
  const el = document.getElementById('statusMsg');
  el.className = cls; el.textContent = msg;
}

function toggleSection(id){
  const body = document.getElementById('ml-body-'+id);
  const btn  = document.getElementById('ml-btn-'+id);
  const open = body.style.maxHeight !== '0px' && body.style.maxHeight !== '';
  body.style.maxHeight = open ? '0px' : '600px';
  body.style.overflow  = open ? 'hidden' : 'visible';
  btn.textContent = open ? '▶' : '▼';
}

// ═══════════════════════════════════════════════
// OPERATIONS
// ═══════════════════════════════════════════════
const ops = {

  // ── TREE ────────────────────────────────────────────────
  trees: {
    'Add Node': async() => {
      const v = getInputVal();
      setStatus('Python: tree.insert('+v+')…');
      const res = await pyPost('/ds/tree/insert', { value:v });
      if(res.ok){ logOp('insert',v); setStatus('insert('+v+') size:'+(res.state?.size??'?'),'status-ok'); }
      else setStatus(res.error,'status-err');
      algoHighlight=[]; treeSearchPath=[];
      applyResponse(res,'tree');
      await runML(v);
    },

    'Search': async() => {
      const v = getInputVal();
      setStatus('Python: tree.search('+v+')…');
      const res = await pyPost('/ds/tree/search', { value:v });
      if(!res.ok && res.error){ setStatus(res.error,'status-err'); return; }
      const path = res.path || [];
      treeSearchPath=[]; algoHighlight=[];
      for(const val of path){
        treeSearchPath = path.slice(0, path.indexOf(val)+1);
        draw();
        await sleep(Math.max(animSpeed*0.7, 150));
      }
      if(res.found){ logOp('found','node('+v+')','lg'); setStatus('Found node('+v+') in path:'+path.join('→'),'status-ok'); showAlgoResult('Search path', path); }
      else          { logOp('not found',v,'le');       setStatus(v+' not found in tree','status-err'); }
      await sleep(900); treeSearchPath=[];
      applyResponse(res,'tree');
      await runML(v);
    },

    'BFS': async() => {
      const g = pyState.tree;
      if(!g.tree){ setStatus('Tree is empty — add nodes first','status-err'); return; }
      setStatus('Python: tree.bfs()…');
      const res = await pyPost('/ds/tree/bfs');
      if(!res.ok){ setStatus(res.error,'status-err'); return; }
      logOp('BFS','tree');
      algoHighlight=[]; treeSearchPath=[];
      for(const frame of (res.frames||[])){
        algoHighlight = frame;
        draw();
        await sleep(animSpeed);
      }
      await sleep(600); algoHighlight=[];
      showAlgoResult('BFS Order', res.order);
      showAlgoBanner('BFS: ' + res.order.join(' → '));
      setStatus('BFS complete: '+res.order.join(' → '),'status-ok');
      draw();
      applyResponse(res,'tree');
      await runML(null);
    },

    'DFS': async() => {
      const g = pyState.tree;
      if(!g.tree){ setStatus('Tree is empty — add nodes first','status-err'); return; }
      setStatus('Python: tree.dfs() [pre-order]…');
      const res = await pyPost('/ds/tree/dfs');
      if(!res.ok){ setStatus(res.error,'status-err'); return; }
      logOp('DFS','tree');
      algoHighlight=[]; treeSearchPath=[];
      for(const frame of (res.frames||[])){
        algoHighlight = frame;
        draw();
        await sleep(animSpeed);
      }
      await sleep(600); algoHighlight=[];
      showAlgoResult('DFS Order', res.order);
      showAlgoBanner('DFS (Pre-order): ' + res.order.join(' → '));
      setStatus('DFS complete: '+res.order.join(' → '),'status-ok');
      draw();
      applyResponse(res,'tree');
      await runML(null);
    },

    'Inorder': async() => { await runTreeTraverse('inorder'); },
    'Preorder': async() => { await runTreeTraverse('preorder'); },
    'Postorder': async() => { await runTreeTraverse('postorder'); },

    'Remove Last': async() => {
      setStatus('Python: tree.remove_last()…');
      const res = await pyPost('/ds/tree/remove_last');
      if(res.ok){ logOp('remove', res.removed); setStatus('Removed node('+res.removed+')','status-ok'); }
      else setStatus(res.error,'status-err');
      algoHighlight=[]; treeSearchPath=[];
      applyResponse(res,'tree');
      await runML(0);
    },

    'Clear': async() => {
      const res = await pyPost('/ds/tree/clear');
      if(res.ok){ logOp('clear',''); setStatus('Tree cleared','status-ok'); }
      algoHighlight=[]; treeSearchPath=[];
      applyResponse(res,'tree');
      await runML(0);
    },
  },

  // ── GRAPH ───────────────────────────────────────────────
  graphs: {
    'Add Node': async() => {
      setStatus('Python: graph.add_node()…');
      const res = await pyPost('/ds/graph/add_node', {});
      if(res.ok){
        const nid = res.node;
        const W = canvas.width, H = canvas.height;
        const angle = (Object.keys(graphPos).length * 137.5) * (Math.PI/180);
        const r = Math.min(W,H)*0.28;
        graphPos[nid] = {
          x: W/2 + r*Math.cos(angle) + (Math.random()-0.5)*40,
          y: H/2 + r*Math.sin(angle) + (Math.random()-0.5)*40,
        };
        logOp('add node', nid,'lg');
        setStatus('Node '+nid+' added','status-ok');
        algoHighlight=[]; algoMSTEdges=[];
        applyResponse(res,'graph');
        await runML(nid);
      } else {
        setStatus(res.error,'status-err');
      }
    },

    'Remove Node': async() => {
      const v = getInputVal();
      setStatus('Python: graph.remove_node('+v+')…');
      const res = await pyPost('/ds/graph/remove_node', { value:v });
      if(res.ok){
        delete graphPos[v];
        if(gSelected===v) gSelected=null;
        logOp('remove node', v);
        setStatus('Node '+v+' removed','status-ok');
        algoHighlight=[]; algoMSTEdges=[];
        applyResponse(res,'graph');
        await runML(v);
      } else {
        setStatus(res.error,'status-err');
      }
    },

    'BFS': async() => {
      const g = pyState.graph;
      if(!g.nodes||!g.nodes.length){ setStatus('Graph empty — add nodes first','status-err'); return; }
      setStatus('Python: graph.bfs()…');
      const startNode = g.nodes[0];
      const res = await pyPost('/ds/graph/bfs', { start:startNode });
      if(!res.ok){ setStatus(res.error,'status-err'); return; }
      logOp('BFS from', res.start);
      algoHighlight=[]; algoMSTEdges=[];
      for(const frame of (res.frames||[])){
        algoHighlight = frame; draw(); await sleep(animSpeed);
      }
      await sleep(600); algoHighlight=[];
      const visited  = res.visited_order || [];
      const isolated = (g.nodes||[]).filter(n => !new Set(visited).has(n));
      let msg = 'BFS: ' + visited.join(' → ');
      if(isolated.length) msg += '  |  isolated: '+isolated.join(', ');
      showAlgoResult('BFS Order', visited);
      showAlgoBanner(msg);
      setStatus(msg,'status-ok');
      draw();
      applyResponse(res,'graph');
      await runML(res.start);
    },

    'DFS': async() => {
      const g = pyState.graph;
      if(!g.nodes||!g.nodes.length){ setStatus('Graph empty — add nodes first','status-err'); return; }
      setStatus('Python: graph.dfs()…');
      const startNode = g.nodes[0];
      const res = await pyPost('/ds/graph/dfs', { start:startNode });
      if(!res.ok){ setStatus(res.error,'status-err'); return; }
      logOp('DFS from', res.start);
      algoHighlight=[]; algoMSTEdges=[];
      for(const frame of (res.frames||[])){
        algoHighlight = frame; draw(); await sleep(animSpeed);
      }
      await sleep(600); algoHighlight=[];
      const visited = res.visited_order || [];
      showAlgoResult('DFS Order', visited);
      showAlgoBanner('DFS: ' + visited.join(' → '));
      setStatus('DFS complete: '+visited.join(' → '),'status-ok');
      draw();
      applyResponse(res,'graph');
      await runML(res.start);
    },

    'Kruskal': async() => {
      const g = pyState.graph;
      if(!g.nodes||g.nodes.length<2){ setStatus('Need at least 2 nodes','status-err'); return; }
      if(!g.edges||!g.edges.length) { setStatus('No edges — connect nodes first','status-err'); return; }
      setStatus('Python: graph.kruskal()…');
      const res = await pyPost('/ds/graph/kruskal');
      if(!res.ok){ setStatus(res.error,'status-err'); return; }
      logOp('Kruskal','MST');
      algoHighlight=[]; algoMSTEdges=[];
      for(const frame of (res.frames||[])){
        algoMSTEdges = frame;
        // Highlight the nodes in the current MST
        algoHighlight = [...new Set(frame.flatMap(e => [e[0],e[1]]))];
        draw();
        await sleep(animSpeed * 1.2);
      }
      await sleep(700);
      algoHighlight = [];
      const mst = res.mst_edges || [];
      showAlgoResult('MST Edges (Kruskal)', mst, true);
      const totalW = res.total_weight ?? '?';
      const spanning = res.is_spanning ? '✅ Spanning' : '⚠ Partial';
      showAlgoBanner('Kruskal MST — total weight: '+totalW+' · '+spanning);
      setStatus('Kruskal done — MST weight: '+totalW+' · '+spanning,'status-ok');
      draw();
      applyResponse(res,'graph');
      await runML(null);
    },

    'Clear': async() => {
      setStatus('Python: graph.clear()…');
      const res = await pyPost('/ds/graph/clear');
      if(res.ok){
        graphPos={}; gSelected=null; algoHighlight=[]; algoMSTEdges=[];
        logOp('clear',''); setStatus('Graph cleared','status-ok');
        applyResponse(res,'graph');
        draw();
        await runML(0);
      } else {
        setStatus(res.error,'status-err');
      }
    },
  },
};

// ── Tree traversal helper ─────────────────────
async function runTreeTraverse(mode){
  const t = pyState.tree;
  if(!t.tree){ setStatus('Tree is empty — add nodes first','status-err'); return; }
  setStatus('Python: tree.traverse('+mode+')…');
  const res = await pyPost('/ds/tree/traverse', { mode });
  if(!res.ok){ setStatus(res.error,'status-err'); return; }
  logOp('traverse', mode, 'lb');
  algoHighlight=[]; treeSearchPath=[];
  for(const frame of (res.frames||[])){
    algoHighlight = frame; draw(); await sleep(animSpeed * 0.85);
  }
  await sleep(700); algoHighlight=[];
  showAlgoResult(`${mode.charAt(0).toUpperCase()+mode.slice(1)} Traversal`, res.order);
  showAlgoBanner(mode+': ' + res.order.join(' → '));
  setStatus(mode+': '+res.order.join(' → '),'status-ok');
  draw();
  applyResponse(res,'tree');
  await runML(null);
}

// ── Button rendering ──────────────────────────
const BTN_STYLES = {
  'Add Node':    'ob-p',
  'Remove Last': 'ob-d',
  'Remove Node': 'ob-d',
  'Clear':       'ob-d',
  'Search':      'ob-s',
  'BFS':         'ob-bfs',
  'DFS':         'ob-dfs',
  'Kruskal':     'ob-kruskal',
  'Inorder':     'ob-traverse',
  'Preorder':    'ob-traverse',
  'Postorder':   'ob-traverse',
};

function renderOpBtns(ds){
  const c = document.getElementById('opBtns');
  c.innerHTML = '';
  Object.keys(ops[ds]||{}).forEach(label => {
    const b = document.createElement('button');
    b.className = `op-btn ${BTN_STYLES[label]||'ob-s'}`;
    b.textContent = label;
    b.id = 'btn-' + label.toLowerCase().replace(' ','-');
    b.onclick = async() => {
      if(animRunning) return;
      animRunning = true;
      animSpeed = (6 - parseInt(document.getElementById('speedRange').value))*180 + 80;
      try { await ops[ds][label](); } catch(err) { console.error('Op error:', err); }
      animRunning = false;
    };
    c.appendChild(b);
  });
}

// ═══════════════════════════════════════════════
// SWITCH TAB
// ═══════════════════════════════════════════════
async function switchDS(ds, btn){
  curDS = ds; gSelected=null; gDragging=null; algoHighlight=[]; algoMSTEdges=[]; treeSearchPath=[];
  canvas.style.cursor = ds==='graphs' ? 'crosshair' : 'default';
  document.querySelectorAll('.ds-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('graphHint').style.display = ds==='graphs' ? 'block' : 'none';

  const dsKey = ds==='trees' ? 'tree' : 'graph';
  const s = await pyGet('/ds/'+dsKey+'/state');
  if(s){
    pyState[dsKey] = s;
    if(ds==='graphs') initGraphPos();
  }
  renderOpBtns(ds); draw(); updateStatePanel(dsKey);
  setStatus('Switched to '+ds,'status-info');
  document.getElementById('opLog').innerHTML = `<span class="lc">// switched to ${ds}</span>`;
  document.getElementById('algoResultBox').innerHTML = '<span style="color:var(--muted)">Run an algorithm to see the traversal order</span>';
}

// ═══════════════════════════════════════════════
// BOOTSTRAP
// ═══════════════════════════════════════════════
async function init(){
  const [t, g] = await Promise.all([
    pyGet('/ds/tree/state'),
    pyGet('/ds/graph/state'),
  ]);
  if(t) pyState.tree  = t;
  if(g){ pyState.graph = g; graphPos={}; }

  initGraphPos();
  renderOpBtns('graphs');   // start on Graphs tab

  ['knn','lr'].forEach(id => {
    document.getElementById('ml-body-'+id).style.maxHeight = '600px';
  });

  // Show graph hint since we start on graphs
  document.getElementById('graphHint').style.display = 'block';

  draw(); updateStatePanel('graph');
  setStatus('Connected to Python backend :5001', 'status-ok');

  // Initial ML run to populate panels
  await runML(null);
}

init();

"""Local browser interface for the end-to-end heart sound pipeline.

Run:
  python app/server.py

Open:
  http://127.0.0.1:8010
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)

from pipeline import HeartSoundPipeline, Thresholds  # noqa: E402


PORT = 8010
RUNS_DIR = os.path.join(ROOT, "runs")
UPLOAD_DIR = os.path.join(RUNS_DIR, "uploads")
RESULT_DIR = os.path.join(RUNS_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

PIPELINE: HeartSoundPipeline | None = None


def get_pipeline() -> HeartSoundPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = HeartSoundPipeline(Thresholds(heart=0.5, murmur=0.5))
    return PIPELINE


def json_bytes(data):
    return json.dumps(data, indent=2).encode("utf-8")


def safe_name(name):
    base = os.path.basename(name).replace("\\", "_").replace("/", "_")
    return "".join(ch if ch.isalnum() or ch in " ._-()" else "_" for ch in base).strip() or "audio.wav"


def parse_multipart_upload(handler):
    ctype = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in ctype or "boundary=" not in ctype:
        raise ValueError("Expected multipart/form-data upload")
    boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
    length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(length)
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b'Content-Disposition:' not in part or b'name="audio"' not in part:
            continue
        header, content = part.split(b"\r\n\r\n", 1)
        content = content.rsplit(b"\r\n", 1)[0]
        header_text = header.decode("utf-8", errors="replace")
        filename = "upload.wav"
        marker = 'filename="'
        if marker in header_text:
            filename = header_text.split(marker, 1)[1].split('"', 1)[0]
        return safe_name(filename), content
    raise ValueError("No audio file field named 'audio' found")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_body(200, "text/html; charset=utf-8", INDEX_HTML)
            return
        if parsed.path == "/api/health":
            ready = PIPELINE is not None
            self.send_body(200, "application/json", json_bytes({"ready": ready}))
            return
        if parsed.path == "/api/audio":
            qs = parse_qs(parsed.query)
            run_id = (qs.get("id") or [""])[0]
            result_path = os.path.join(RESULT_DIR, f"{run_id}.json")
            if not os.path.isfile(result_path):
                self.send_body(404, "application/json", json_bytes({"error": "result not found"}))
                return
            with open(result_path) as fh:
                result = json.load(fh)
            audio_path = result.get("path", "")
            if not os.path.isfile(audio_path):
                self.send_body(404, "application/json", json_bytes({"error": "audio not found"}))
                return
            with open(audio_path, "rb") as fh:
                self.send_body(200, "audio/wav", fh.read())
            return
        self.send_body(404, "text/plain", "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_body(404, "text/plain", "not found")
            return
        try:
            filename, content = parse_multipart_upload(self)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            digest = hashlib.sha1(content).hexdigest()[:10]
            run_id = f"{stamp}-{digest}"
            upload_path = os.path.join(UPLOAD_DIR, f"{run_id}-{filename}")
            with open(upload_path, "wb") as fh:
                fh.write(content)
            result = get_pipeline().run(upload_path)
            result["run_id"] = run_id
            result_path = os.path.join(RESULT_DIR, f"{run_id}.json")
            with open(result_path, "w") as fh:
                json.dump(result, fh, indent=2)
            self.send_body(200, "application/json", json_bytes(result))
        except Exception as exc:
            self.send_body(500, "application/json", json_bytes({"error": f"{type(exc).__name__}: {exc}"}))


class ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heart Sound Pipeline</title>
<style>
  :root{--s1:#2563eb;--systole:#f59e0b;--s2:#16a34a;--diastole:#db2777;--ok:#22c55e;--bad:#f43f5e;}
  *{box-sizing:border-box}
  body{margin:0;padding:22px;background:#0f172a;color:#e2e8f0;font-family:system-ui,Segoe UI,Roboto,sans-serif}
  main{max-width:1220px;margin:0 auto}
  h1{font-size:21px;margin:0 0 3px} .sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
  .card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:14px}
  .upload{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
  input[type=file]{font-size:14px;color:#cbd5e1}
  button{font-size:14px;padding:8px 12px;border-radius:8px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;cursor:pointer}
  button:hover{background:#334155} button:disabled{opacity:.5;cursor:wait}
  button.primary{background:#2563eb;border-color:#2563eb} button.primary:hover{background:#1d4ed8}
  label.chk{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #334155;background:#1e293b;display:inline-flex;align-items:center;gap:6px}
  .status{font-size:13px;color:#94a3b8;min-height:20px}
  .grid{display:grid;grid-template-columns:330px 1fr;gap:14px}
  .final{font-weight:700;font-size:17px;margin-bottom:12px;padding:8px 10px;border-radius:8px;background:#1e293b;border:1px solid #334155}
  .steps{display:grid;gap:9px}
  .step{border:1px solid #1f2937;border-radius:8px;padding:10px;background:#0d1526}
  .step h2{font-size:13px;margin:0 0 6px;display:flex;justify-content:space-between;gap:8px;align-items:center;font-weight:600}
  .pill{border-radius:999px;padding:2px 9px;font-size:11.5px;background:#334155;color:#cbd5e1;font-weight:600}
  .pill.ok{background:rgba(34,197,94,.15);color:var(--ok)}.pill.bad{background:rgba(244,63,94,.15);color:var(--bad)}
  .kv{display:grid;grid-template-columns:1fr auto;gap:4px;font-size:12.5px;color:#94a3b8}.kv b{color:#e2e8f0;font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
  td,th{border-top:1px solid #1f2937;padding:6px 4px;text-align:left}th{color:#94a3b8;font-weight:600}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  #wrap{position:relative;background:#0b1220;border:1px solid #1f2937;border-radius:8px;overflow:hidden}
  #canvas{display:block;width:100%;height:240px;cursor:crosshair}
  #ruler{display:block;width:100%;height:22px;background:#0b1220}
  #scroll{width:100%;margin:8px 0 2px}
  .empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:14px;text-align:center;padding:0 20px}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 6px;font-size:13px;color:#cbd5e1}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:13px;height:13px;border-radius:3px;display:inline-block}
  #hover{font-size:13px;color:#94a3b8;min-height:18px}
  .hint{font-size:12px;color:#64748b;margin-top:8px}
  kbd{background:#1e293b;border:1px solid #334155;border-radius:4px;padding:1px 5px;font-size:11px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
</style></head>
<body><main>
  <h1>Heart Sound End-to-End Pipeline</h1>
  <div class="sub">Heart gate &rarr; murmur classifier &rarr; segmentation &rarr; systolic/diastolic timing</div>
  <section class="card upload">
    <input id="file" type="file" accept=".wav,.flac,.ogg,audio/*">
    <button id="run" class="primary">Analyze</button>
    <span class="status" id="status">Choose an audio file.</span>
  </section>
  <section class="grid">
    <aside class="card">
      <div class="final" id="final">No result yet</div>
      <div class="steps">
        <div class="step"><h2>1. Heart Sound Gate <span class="pill" id="heartPill">waiting</span></h2><div class="kv" id="heartKv"></div></div>
        <div class="step"><h2>2. Murmur Classifier <span class="pill" id="murmurPill">waiting</span></h2><div class="kv" id="murmurKv"></div></div>
        <div class="step"><h2>3. Segmentation <span class="pill" id="segPill">waiting</span></h2><div class="kv" id="segKv"></div></div>
        <div class="step"><h2>4. Timing <span class="pill" id="timingPill">waiting</span></h2><div class="kv" id="timingKv"></div></div>
      </div>
      <table id="table"></table>
    </aside>
    <section class="card">
      <div class="controls">
        <button id="playBtn" class="primary">&#9654; Play all</button>
        <button id="stopBtn">&#9632; Stop</button>
        <label class="chk"><input type="checkbox" id="loopChk" style="width:auto"> loop segment</label>
        <button id="zoomIn">&#43; Zoom</button>
        <button id="zoomOut">&minus; Zoom</button>
        <button id="zoomReset">Reset</button>
        <label class="chk"><input type="checkbox" id="followChk" style="width:auto" checked> follow playhead</label>
      </div>
      <div id="wrap"><canvas id="canvas"></canvas><canvas id="ruler"></canvas><div id="empty" class="empty">Upload a recording and click Analyze &mdash; the waveform (and any predicted S1/Systole/S2/Diastole segments) will appear here.</div></div>
      <input type="range" id="scroll" min="0" max="1000" value="0" step="1" disabled>
      <div class="legend">
        <span><i class="sw" style="background:var(--s1)"></i>S1</span>
        <span><i class="sw" style="background:var(--systole)"></i>Systole</span>
        <span><i class="sw" style="background:var(--s2)"></i>S2</span>
        <span><i class="sw" style="background:var(--diastole)"></i>Diastole</span>
      </div>
      <div id="hover"></div>
      <div class="hint">Wheel = zoom at cursor &middot; Shift+Wheel or &larr;/&rarr; = pan &middot; drag scrollbar to pan &middot; click a segment to play it &middot; <kbd>Space</kbd> play/stop &middot; <kbd>+</kbd>/<kbd>&minus;</kbd> zoom.</div>
    </section>
  </section>
</main>
<script>
const COLOR={S1:'#2563eb',Systole:'#f59e0b',S2:'#16a34a',Diastole:'#db2777'};
const $=id=>document.getElementById(id);
const canvas=$('canvas'),ctx=canvas.getContext('2d');
const ruler=$('ruler'),rctx=ruler.getContext('2d');
const scroll=$('scroll'),followChk=$('followChk'),loopChk=$('loopChk'),hoverEl=$('hover');
let result=null,audioCtx=null,audioBuffer=null,peaks=null,globalMax=1,segments=[],duration=0;
let source=null,playing=false,playStartCtxTime=0,playStartOffset=0,loopRegion=null,rafId=null;
const MINDUR=0.15;               // most-zoomed view (seconds)
let viewStart=0,viewDur=0;       // visible window

function pct(x){return (100*x).toFixed(1)+'%'} function num(x,n=3){return Number(x).toFixed(n)}
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function timeToX(t){const W=canvas.clientWidth;return (t-viewStart)/viewDur*W}
function xToTime(x){const W=canvas.clientWidth;return viewStart+x/W*viewDur}
function setPill(id,text,kind){const el=$(id);el.textContent=text;el.className='pill '+(kind||'')}
function kv(id,rows){$(id).innerHTML=rows.map(r=>`<span>${r[0]}</span><b>${r[1]}</b>`).join('')}

// ---------- results panel ----------
function update(result){
  $('final').textContent=result.final_decision||'Done';
  const h=result.stages.heart_sound_gate||{}; setPill('heartPill',h.detected?'heart':'no heart',h.detected?'ok':'bad');
  kv('heartKv',[['p(heart)',num(h.p_heart||0)],['logreg',num(h.p_heart_logreg||0)],['svm',num(h.p_heart_svm_rbf||0)]]);
  const m=result.stages.murmur_classifier||{};
  if(m.p_present===undefined){setPill('murmurPill','skipped',''); kv('murmurKv',[['reason','no heart detected']]);}
  else{setPill('murmurPill',m.detected?'present':'absent',m.detected?'bad':'ok'); kv('murmurKv',[['p(present)',num(m.p_present)],['p(absent)',num(m.p_absent)],['windows',m.n_windows]]);}
  const s=result.stages.segmentation||{};
  if(!s.n_segments){setPill('segPill','skipped',''); kv('segKv',[['segments','-'],['cycles','-']]);}
  else{setPill('segPill','done','ok'); kv('segKv',[['segments',s.n_segments],['cycles',s.n_cycles],['systole share',pct(s.phase_fraction.Systole||0)],['diastole share',pct(s.phase_fraction.Diastole||0)]]);}
  const t=result.stages.timing||{};
  if(!t.timing){setPill('timingPill','skipped',''); kv('timingKv',[['timing','-']]);}
  else{setPill('timingPill',t.timing,t.timing==='SYSTOLIC'?'ok':'bad'); kv('timingKv',[['systolic energy',num(t.systolic_pct,1)+'%'],['diastolic energy',num(t.diastolic_pct,1)+'%']]);}
  $('table').innerHTML='<tr><th>Runtime</th><th>Seconds</th></tr>'+[
    ['heart gate',h.runtime_s],['murmur classifier',m.runtime_s],['timing/segmentation',t.runtime_s],['total',result.runtime_s]
  ].filter(r=>r[1]!==undefined).map(r=>`<tr><td>${r[0]}</td><td>${num(r[1],3)}</td></tr>`).join('');
}

// ---------- audio + interactive viewer ----------
async function decodeAudio(){
  const arr=await (await fetch('/api/audio?id='+encodeURIComponent(result.run_id))).arrayBuffer();
  if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  audioBuffer=await audioCtx.decodeAudioData(arr.slice(0));
  duration=audioBuffer.duration;
  const raw=audioBuffer.getChannelData(0); globalMax=0;
  for(let i=0;i<raw.length;i++){const v=Math.abs(raw[i]);if(v>globalMax)globalMax=v;}
  if(globalMax<=0)globalMax=1;
  segments=result.segments||[];
  $('empty').style.display='none';
  setView(0,duration);
}

function setView(start,dur){
  viewDur=clamp(dur,MINDUR,duration||dur);
  viewStart=clamp(start,0,Math.max(0,duration-viewDur));
  computePeaks(); syncScroll(); draw();
}
function syncScroll(){
  const span=Math.max(1e-9,duration-viewDur);
  scroll.value=Math.round(viewStart/span*1000)||0;
  scroll.disabled=viewDur>=duration;
}
function computePeaks(){
  if(!audioBuffer)return;
  const raw=audioBuffer.getChannelData(0),sr=audioBuffer.sampleRate;
  const W=Math.max(50,canvas.clientWidth||900);
  const s0=Math.floor(viewStart*sr),s1=Math.min(raw.length,Math.floor((viewStart+viewDur)*sr));
  const n=Math.max(1,s1-s0),step=n/W;
  peaks=new Float32Array(W);
  for(let i=0;i<W;i++){let mx=0;const a=s0+Math.floor(i*step),b=s0+Math.floor((i+1)*step);
    for(let j=a;j<b;j++){const v=Math.abs(raw[j]);if(v>mx)mx=v;}peaks[i]=mx/globalMax;}
}
function hexA(hex,a){hex=hex.replace('#','');
  const r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16);return`rgba(${r},${g},${b},${a})`;}

function draw(){
  const dpr=window.devicePixelRatio||1,W=canvas.clientWidth,H=240;
  canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.height=H+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
  if(!duration){drawRuler();return;}
  for(const s of segments){
    if(s.end<viewStart||s.start>viewStart+viewDur)continue;
    const x0=Math.max(0,timeToX(s.start)),x1=Math.min(W,timeToX(s.end)),c=COLOR[s.name]||'#888';
    ctx.fillStyle=hexA(c,0.30);ctx.fillRect(x0,0,Math.max(1,x1-x0),H);
    ctx.fillStyle=c;ctx.fillRect(x0,0,Math.max(1,x1-x0),4);
  }
  if(peaks){ctx.fillStyle='rgba(226,232,240,0.88)';const mid=H/2,bw=W/peaks.length;
    for(let i=0;i<peaks.length;i++){const h=peaks[i]*(H*0.42);ctx.fillRect(i*bw,mid-h,Math.max(1,bw),h*2);}}
  ctx.font='11px system-ui';ctx.textAlign='center';
  for(const s of segments){
    if(s.end<viewStart||s.start>viewStart+viewDur)continue;
    const x0=timeToX(s.start);
    if(x0>=0&&x0<=W){ctx.strokeStyle='rgba(148,163,184,0.30)';ctx.beginPath();ctx.moveTo(x0,0);ctx.lineTo(x0,H);ctx.stroke();}
    const w=timeToX(s.end)-timeToX(s.start),cx=timeToX(s.start)+w/2;
    if(w>20&&cx>0&&cx<W){ctx.fillStyle=COLOR[s.name]||'#888';ctx.fillText(s.name,cx,H-8);}
  }
  if(playing){const t=playStartOffset+(audioCtx.currentTime-playStartCtxTime);
    if(t>=viewStart&&t<=viewStart+viewDur){const x=timeToX(t);
      ctx.strokeStyle='#f8fafc';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();ctx.lineWidth=1;}}
  drawRuler();
}
function drawRuler(){
  const dpr=window.devicePixelRatio||1,W=ruler.clientWidth,H=22;
  ruler.width=W*dpr;ruler.height=H*dpr;ruler.style.height=H+'px';
  rctx.setTransform(dpr,0,0,dpr,0,0);rctx.clearRect(0,0,W,H);
  if(!duration)return;
  const targets=[0.05,0.1,0.2,0.25,0.5,1,2,5,10,15,30,60];
  let step=targets[targets.length-1];
  for(const t of targets){if(viewDur/t<=12){step=t;break;}}
  rctx.fillStyle='#64748b';rctx.strokeStyle='rgba(100,116,139,0.4)';rctx.font='10px system-ui';rctx.textAlign='left';
  const first=Math.ceil(viewStart/step)*step;
  for(let t=first;t<=viewStart+viewDur;t+=step){const x=timeToX(t);
    rctx.beginPath();rctx.moveTo(x,0);rctx.lineTo(x,6);rctx.stroke();
    rctx.fillText(t.toFixed(step<1?2:(step<10?1:0))+'s',x+3,15);}
}

// ---------- playback ----------
function playRange(a,b){
  if(!audioBuffer)return;stop();if(audioCtx.state==='suspended')audioCtx.resume();
  source=audioCtx.createBufferSource();source.buffer=audioBuffer;source.connect(audioCtx.destination);
  source.start(0,a,Math.max(0.01,b-a));playStartCtxTime=audioCtx.currentTime;playStartOffset=a;playing=true;
  source.onended=()=>{if(loopRegion&&playing){playRange(loopRegion.start,loopRegion.end);}else{playing=false;source=null;draw();}};
  animate();
}
function playAll(){if(!audioBuffer)return;loopRegion=null;playRange(viewStart,duration);}
function stop(){if(source){try{source.onended=null;source.stop();}catch(e){}source=null;}playing=false;
  if(rafId){cancelAnimationFrame(rafId);rafId=null;}loopRegion=null;draw();}
function animate(){
  if(playing&&followChk.checked){
    const t=playStartOffset+(audioCtx.currentTime-playStartCtxTime);
    if(viewDur<duration&&(t>viewStart+viewDur||t<viewStart)){setView(t-viewDur*0.1,viewDur);}
  }
  draw();if(playing)rafId=requestAnimationFrame(animate);
}
function zoomAt(centerT,factor){
  const nd=clamp(viewDur*factor,MINDUR,duration);
  const frac=(centerT-viewStart)/viewDur;
  setView(centerT-frac*nd,nd);
}

// ---------- events ----------
canvas.addEventListener('wheel',e=>{
  if(!duration)return;e.preventDefault();
  if(e.shiftKey){setView(viewStart+(e.deltaY>0?1:-1)*viewDur*0.15,viewDur);}
  else{const rect=canvas.getBoundingClientRect();zoomAt(xToTime(e.clientX-rect.left),e.deltaY<0?0.8:1.25);}
},{passive:false});
canvas.addEventListener('click',ev=>{
  if(!duration)return;const rect=canvas.getBoundingClientRect();
  const t=xToTime(ev.clientX-rect.left);
  const s=segments.find(x=>t>=x.start&&t<x.end);
  if(s){loopRegion=loopChk.checked?{start:s.start,end:s.end}:null;playRange(s.start,s.end);}
  else{playRange(t,duration);}
});
canvas.addEventListener('mousemove',ev=>{
  if(!duration){hoverEl.textContent='';return;}const rect=canvas.getBoundingClientRect();
  const t=xToTime(ev.clientX-rect.left);const s=segments.find(x=>t>=x.start&&t<x.end);
  hoverEl.textContent=s?`${s.name}  -  ${s.start.toFixed(3)}-${s.end.toFixed(3)}s (${(s.end-s.start).toFixed(3)}s)  |  view ${viewStart.toFixed(2)}-${(viewStart+viewDur).toFixed(2)}s`
                      :`t = ${t.toFixed(3)}s  |  view ${viewStart.toFixed(2)}-${(viewStart+viewDur).toFixed(2)}s`;
});
canvas.addEventListener('mouseleave',()=>hoverEl.textContent='');
scroll.addEventListener('input',()=>{const span=Math.max(0,duration-viewDur);setView(scroll.value/1000*span,viewDur);});
$('zoomIn').onclick=()=>{if(duration)zoomAt(viewStart+viewDur/2,0.6);};
$('zoomOut').onclick=()=>{if(duration)zoomAt(viewStart+viewDur/2,1.7);};
$('zoomReset').onclick=()=>{if(duration)setView(0,duration);};
$('playBtn').onclick=playAll;
$('stopBtn').onclick=stop;
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(!duration)return;
  if(e.code==='Space'){e.preventDefault();playing?stop():playAll();}
  else if(e.key==='ArrowLeft'){setView(viewStart-viewDur*0.2,viewDur);}
  else if(e.key==='ArrowRight'){setView(viewStart+viewDur*0.2,viewDur);}
  else if(e.key==='+'||e.key==='='){zoomAt(viewStart+viewDur/2,0.6);}
  else if(e.key==='-'){zoomAt(viewStart+viewDur/2,1.7);}
});
window.addEventListener('resize',()=>{if(audioBuffer){computePeaks();draw();}});

$('run').onclick=async()=>{
  const f=$('file').files[0]; if(!f){$('status').textContent='Choose a file first.'; return;}
  stop(); $('run').disabled=true; $('status').textContent='Loading models and analyzing... first run can take a bit.';
  const fd=new FormData(); fd.append('audio',f);
  try{
    const res=await fetch('/api/analyze',{method:'POST',body:fd});
    result=await res.json(); if(result.error) throw new Error(result.error);
    update(result); await decodeAudio(); $('status').textContent='Done.';
  }catch(e){$('status').textContent='Error: '+e.message;}
  finally{$('run').disabled=false;}
};
</script></body></html>"""


def main():
    print("Heart Sound End-to-End Pipeline")
    print(f"  open: http://127.0.0.1:{PORT}")
    print("  loading models on first analysis")
    with ThreadingServer(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()

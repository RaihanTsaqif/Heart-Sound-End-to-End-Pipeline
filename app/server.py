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
  :root{--bg:#f7f8fb;--ink:#172033;--muted:#65728a;--line:#d9dfeb;--panel:#fff;--s1:#2563eb;--sys:#f59e0b;--s2:#16a34a;--dia:#db2777;--ok:#0f8f5f;--bad:#bf2c34;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,Roboto,sans-serif}
  main{max-width:1180px;margin:0 auto;padding:22px}
  header{display:flex;justify-content:space-between;gap:16px;align-items:flex-end;margin-bottom:16px}
  h1{font-size:24px;margin:0} .sub{color:var(--muted);font-size:13px;margin-top:4px}
  .upload{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
  input[type=file]{font-size:14px} button{border:1px solid #b9c2d2;background:#172033;color:#fff;border-radius:7px;padding:9px 13px;cursor:pointer}
  button:disabled{opacity:.5;cursor:wait}.status{font-size:13px;color:var(--muted);min-height:20px}
  .grid{display:grid;grid-template-columns:340px 1fr;gap:14px;margin-top:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
  .steps{display:grid;gap:10px}.step{border:1px solid var(--line);border-radius:8px;padding:10px;background:#fbfcfe}
  .step h2{font-size:14px;margin:0 0 6px;display:flex;justify-content:space-between;gap:8px}.pill{border-radius:999px;padding:2px 8px;font-size:12px;background:#eef2f7;color:#364257}
  .pill.ok{background:#e8f7ef;color:var(--ok)}.pill.bad{background:#fdecee;color:var(--bad)}.kv{display:grid;grid-template-columns:1fr auto;gap:4px;font-size:13px;color:var(--muted)}
  .final{font-weight:700;font-size:18px;margin-bottom:10px}.meter{height:8px;background:#e8edf5;border-radius:99px;overflow:hidden;margin-top:5px}.bar{height:100%;background:#2663eb;width:0}
  audio{width:100%;margin:8px 0 12px} #wrap{position:relative;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#0b1220}
  canvas{display:block;width:100%;height:250px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--muted)}
  .sw{width:13px;height:13px;border-radius:3px;display:inline-block;margin-right:5px;vertical-align:-2px}.empty{height:250px;display:flex;align-items:center;justify-content:center;color:#94a3b8;background:#0b1220}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}td,th{border-top:1px solid var(--line);padding:7px;text-align:left}th{color:var(--muted);font-weight:600}
  @media(max-width:850px){.grid{grid-template-columns:1fr}header{display:block}}
</style></head>
<body><main>
  <header>
    <div><h1>Heart Sound End-to-End Pipeline</h1><div class="sub">Heart gate -> murmur classifier -> segmentation -> systolic/diastolic timing</div></div>
  </header>
  <section class="upload">
    <input id="file" type="file" accept=".wav,.flac,.ogg,audio/*">
    <button id="run">Analyze</button>
    <span class="status" id="status">Choose an audio file.</span>
  </section>
  <section class="grid">
    <aside class="panel">
      <div class="final" id="final">No result yet</div>
      <div class="steps">
        <div class="step"><h2>1. Heart Sound Gate <span class="pill" id="heartPill">waiting</span></h2><div class="kv" id="heartKv"></div></div>
        <div class="step"><h2>2. Murmur Classifier <span class="pill" id="murmurPill">waiting</span></h2><div class="kv" id="murmurKv"></div></div>
        <div class="step"><h2>3. Segmentation <span class="pill" id="segPill">waiting</span></h2><div class="kv" id="segKv"></div></div>
        <div class="step"><h2>4. Timing <span class="pill" id="timingPill">waiting</span></h2><div class="kv" id="timingKv"></div></div>
      </div>
    </aside>
    <section class="panel">
      <audio id="audio" controls></audio>
      <div id="wrap"><canvas id="canvas"></canvas><div id="empty" class="empty">Segmentation appears here when a murmur is detected.</div></div>
      <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>S1</span><span><i class="sw" style="background:var(--sys)"></i>Systole</span><span><i class="sw" style="background:var(--s2)"></i>S2</span><span><i class="sw" style="background:var(--dia)"></i>Diastole</span></div>
      <table id="table"></table>
    </section>
  </section>
</main>
<script>
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d'), audio=$('audio');
const colors={S1:'#2563eb',Systole:'#f59e0b',S2:'#16a34a',Diastole:'#db2777'};
let result=null, audioBuffer=null, audioCtx=null, peaks=null;
function pct(x){return (100*x).toFixed(1)+'%'} function num(x,n=3){return Number(x).toFixed(n)}
function setPill(id,text,kind){const el=$(id); el.textContent=text; el.className='pill '+(kind||'')}
function kv(id,rows){$(id).innerHTML=rows.map(r=>`<span>${r[0]}</span><b>${r[1]}</b>`).join('')}
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
async function decodeAudio(){
  if(!result) return; audio.src='/api/audio?id='+encodeURIComponent(result.run_id);
  const arr=await (await fetch(audio.src)).arrayBuffer(); if(!audioCtx) audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  audioBuffer=await audioCtx.decodeAudioData(arr.slice(0)); makePeaks(); draw();
}
function makePeaks(){
  if(!audioBuffer)return; const raw=audioBuffer.getChannelData(0), W=Math.max(500,canvas.clientWidth), step=Math.max(1,Math.floor(raw.length/W)); peaks=new Float32Array(W); let mx=1e-9;
  for(let i=0;i<W;i++){let p=0; for(let j=i*step;j<Math.min(raw.length,(i+1)*step);j++){p=Math.max(p,Math.abs(raw[j]));} peaks[i]=p; mx=Math.max(mx,p);}
  for(let i=0;i<W;i++) peaks[i]/=mx;
}
function draw(){
  const empty=$('empty'); if(!result||!audioBuffer){canvas.style.display='none'; empty.style.display='flex'; return;}
  canvas.style.display='block'; empty.style.display='none';
  const dpr=window.devicePixelRatio||1, W=canvas.clientWidth, H=250; canvas.width=W*dpr; canvas.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const dur=audioBuffer.duration, segs=result.segments||[];
  for(const s of segs){const x0=s.start/dur*W, x1=s.end/dur*W, c=colors[s.name]||'#999'; ctx.fillStyle=hex(c,.27); ctx.fillRect(x0,0,Math.max(1,x1-x0),H); ctx.fillStyle=c; ctx.fillRect(x0,0,Math.max(1,x1-x0),5);}
  ctx.fillStyle='rgba(226,232,240,.9)'; const mid=H/2, bw=W/(peaks?peaks.length:W); if(peaks){for(let i=0;i<peaks.length;i++){const h=peaks[i]*(H*.42); ctx.fillRect(i*bw,mid-h,Math.max(1,bw),h*2);}}
  ctx.font='11px system-ui'; ctx.textAlign='center'; for(const s of segs){const x0=s.start/dur*W, x1=s.end/dur*W, w=x1-x0; if(w>28){ctx.fillStyle=colors[s.name]||'#fff'; ctx.fillText(s.name,x0+w/2,H-9);}}
}
function hex(hex,a){hex=hex.replace('#',''); const r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16); return `rgba(${r},${g},${b},${a})`;}
$('run').onclick=async()=>{
  const f=$('file').files[0]; if(!f){$('status').textContent='Choose a file first.'; return;}
  $('run').disabled=true; $('status').textContent='Loading models and analyzing... first run can take a bit.';
  const fd=new FormData(); fd.append('audio',f);
  try{const res=await fetch('/api/analyze',{method:'POST',body:fd}); result=await res.json(); if(result.error) throw new Error(result.error); update(result); await decodeAudio(); $('status').textContent='Done.';}
  catch(e){$('status').textContent='Error: '+e.message;}
  finally{$('run').disabled=false;}
};
window.addEventListener('resize',()=>{makePeaks();draw();});
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

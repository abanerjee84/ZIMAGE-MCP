import argparse, asyncio, base64, json, os, secrets, shutil, socket, subprocess, sys, threading, time, uuid
from pathlib import Path
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from filelock import FileLock

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
OUT=DATA/'outputs'; OUT.mkdir(exist_ok=True)
TOKEN_FILE=DATA/'token.txt'
if not TOKEN_FILE.exists(): TOKEN_FILE.write_text(secrets.token_hex(32))
TOKEN=TOKEN_FILE.read_text().strip()
URL='http://127.0.0.1:8773'
GPU=os.environ.get('ZIMAGE_GPU_UUID', 'GPU-1fbff0da-1530-2c25-e1df-4f00ecd1601b')
FILES=[ROOT/'models/z_image_turbo-Q6_K.gguf',ROOT/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',ROOT/'models/ae.safetensors']

def executable():
    hits=list((ROOT/'vendor'/'sd').rglob('sd-cli.exe'))
    return hits[0] if hits else ROOT/'vendor/sd/sd-cli.exe'
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.free,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    for row in p.stdout.splitlines():
        u,n,f,used=[x.strip() for x in row.split(',')]
        if u==GPU:
            if '3060' not in n: raise RuntimeError('Configured GPU is not an RTX 3060')
            return {'uuid':u,'name':n,'free_mib':int(f),'used_mib':int(used)}
    raise RuntimeError('Configured RTX 3060 not found')
def doctor():
    try: g=gpu()
    except Exception as e: g={'error':str(e)}
    missing=[str(p) for p in [*FILES,executable(),executable().with_name('sd-server.exe')] if not p.is_file()]
    return {'gpu':g,'missing_files':missing,'ready':not missing and 'error' not in g,'min_free_mib':10500,'max_total_used_mib':11500,'engine_loaded':bool(proc and proc.poll() is None)}
class Request(BaseModel):
    prompt:str=Field(min_length=1,max_length=12000)
    width:int=Field(default=1024,ge=256,le=1536,multiple_of=16)
    height:int=Field(default=1024,ge=256,le=1536,multiple_of=16)
    steps:int=Field(default=8,ge=1,le=100)
    cfg:float=Field(default=1.0,ge=0,le=20)
    seed:int=Field(default=-1,ge=-1,le=2147483647)
    client_request_id:str|None=Field(default=None,max_length=128)
lock=threading.Lock(); jobs={}; proc=None; engine_url=None; cancel=set(); stopping=False

def save(j):
    path=OUT/(j['job_id']+'.json'); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(j,indent=2),encoding='utf-8'); tmp.replace(path)
def job(id):
    if not (len(id)==32 and all(c in '0123456789abcdef' for c in id)): raise HTTPException(404,'Job not found')
    p=OUT/(id+'.json')
    if not p.exists(): raise HTTPException(404,'Job not found')
    return json.loads(p.read_text(encoding='utf-8'))
def stop_engine():
    global proc, engine_url
    if proc is not None:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        proc=None
    engine_url=None

def ensure_engine():
    global proc, engine_url
    if proc is not None and proc.poll() is None: return
    stop_engine()
    g=gpu()
    if g['free_mib']<10500: raise RuntimeError(f"RTX 3060 has {g['free_mib']} MiB free; at least 10500 MiB is required. Stop other GPU services first.")
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    engine_url=f'http://127.0.0.1:{port}'
    cmd=[str(executable().with_name('sd-server.exe')),'--listen-ip','127.0.0.1','--listen-port',str(port),'--diffusion-model',str(FILES[0]),'--llm',str(FILES[1]),'--vae',str(FILES[2]),'--backend','cuda0','--params-backend','cuda0','--max-vram','10.5','--flash-attn','auto','--vae-tiling']
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=GPU
    with (DATA/'engine.log').open('a',encoding='utf-8') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=executable().parent,creationflags=subprocess.CREATE_NO_WINDOW)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if stopping: stop_engine(); raise InterruptedError('Service stopping')
        if proc.poll() is not None: raise RuntimeError(f'Image engine exited {proc.returncode}; see {DATA / "engine.log"}')
        if gpu()['used_mib']>11500: stop_engine(); raise RuntimeError('GPU usage exceeded 11500 MiB during engine startup')
        try:
            r=httpx.get(engine_url+'/sdcpp/v1/capabilities',timeout=2)
            if r.status_code==200: return
        except httpx.HTTPError: pass
        time.sleep(1)
    stop_engine(); raise RuntimeError('Image engine did not start within 300 seconds')

def run(j):
    id=j['job_id']; path=OUT/(id+'.png')
    with lock:
        if id in cancel:
            j['status']='cancelled'; save(j); return
        j['status']='running'; j['started_at']=time.time(); save(j)
    try:
        d=doctor()
        if d['missing_files']: raise RuntimeError('Missing files: '+', '.join(d['missing_files']))
        ensure_engine()
        q=j['request']; seed=q['seed'] if q['seed']>=0 else secrets.randbelow(2147483647); j['seed']=seed; j['gpu']=gpu()
        payload={'prompt':q['prompt'],'width':q['width'],'height':q['height'],'seed':seed,'steps':q['steps'],'cfg_scale':q['cfg']}
        outcome={}
        def generate():
            try:
                response=httpx.post(engine_url+'/sdapi/v1/txt2img',json=payload,timeout=3600)
                response.raise_for_status(); outcome['data']=response.json()
            except Exception as e: outcome['error']=e
        task=threading.Thread(target=generate,daemon=True); task.start()
        peak=gpu()['used_mib']
        while task.is_alive():
            if id in cancel or stopping:
                stop_engine(); task.join(timeout=10); raise InterruptedError('Cancelled; engine unloaded')
            if proc is None or proc.poll() is not None: raise RuntimeError(f'Image engine exited; see {DATA / "engine.log"}')
            current=gpu()['used_mib']; peak=max(peak,current)
            if current>11500: stop_engine(); raise RuntimeError(f'GPU usage exceeded 11500 MiB ({current} MiB); engine stopped.')
            task.join(timeout=1)
        if 'error' in outcome: raise outcome['error']
        path.write_bytes(base64.b64decode(outcome['data']['images'][0],validate=True))
        j.update(status='succeeded',result={'image_path':str(path),'log_path':str(DATA/'engine.log'),'peak_total_gpu_used_mib':peak,'engine_resident':True})
    except InterruptedError as e: j.update(status='cancelled',error=str(e))
    except Exception as e: j.update(status='failed',error=str(e),log_path=str(DATA/'engine.log'))
    finally:
        j['finished_at']=time.time(); save(j)

def worker():
    import queue
    while True:
        j=work.get()
        if j is None: break
        run(j); work.task_done()
import queue
work=queue.Queue()
@asynccontextmanager
async def lifespan(app):
    for p in OUT.glob('*.json'):
        try:
            j=json.loads(p.read_text(encoding='utf-8'))
            if j['status'] in ('queued','running'): j.update(status='failed',error='Service stopped before job completed'); save(j)
        except Exception: pass
    threading.Thread(target=worker,daemon=True).start()
    try:
        yield
    finally:
        stop_engine()
app=FastAPI(lifespan=lifespan)
def auth(authorization):
    if authorization!='Bearer '+TOKEN: raise HTTPException(401,'Unauthorized')
@app.get('/health')
def health(authorization:str=Header(default='')):
    auth(authorization); return doctor()
@app.post('/jobs')
def submit(q:Request,authorization:str=Header(default='')):
    auth(authorization)
    with lock:
        if q.client_request_id:
            for p in OUT.glob('*.json'):
                try:
                    old=json.loads(p.read_text(encoding='utf-8'))
                    if old['request'].get('client_request_id')==q.client_request_id:
                        if old['request']!=q.model_dump(): raise HTTPException(409,'Request ID already used with different parameters')
                        return old
                except (ValueError,KeyError): pass
        j={'job_id':uuid.uuid4().hex,'status':'queued','request':q.model_dump(),'created_at':time.time()}; save(j); work.put(j); return j
@app.get('/jobs/{id}')
def status(id:str,authorization:str=Header(default='')): auth(authorization); return job(id)
@app.post('/jobs/{id}/cancel')
def cancellation(id:str,authorization:str=Header(default='')):
    auth(authorization); j=job(id)
    if j['status'] in ('queued','running'): cancel.add(id)
    return {'job_id':id,'cancel_requested':j['status'] in ('queued','running')}
@app.post('/shutdown')
def shutdown(authorization:str=Header(default='')):
    global stopping
    auth(authorization); stopping=True; stop_engine(); app.state.server.should_exit=True; return {'status':'stopping'}

def request(method,path,**kw):
    return httpx.request(method,URL+path,headers={'Authorization':'Bearer '+TOKEN},timeout=5,**kw)
def start():
    try:
        if request('GET','/health').status_code==200: return
    except httpx.HTTPError: pass
    with FileLock(str(DATA/'start.lock'),timeout=30):
        try:
            if request('GET','/health').status_code==200: return
        except httpx.HTTPError: pass
        with (DATA/'service.log').open('a') as f:
            subprocess.Popen([sys.executable,str(__file__),'serve'],cwd=ROOT,stdout=f,stderr=f,creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            time.sleep(.2)
            try:
                if request('GET','/health').status_code==200: return
            except httpx.HTTPError: pass
        raise RuntimeError('Service failed to start; see data/service.log')
def call(method,path,**kw):
    start(); r=request(method,path,**kw); r.raise_for_status(); return r.json()

def mcp_main():
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
    def structured(value):
        return CallToolResult(content=[TextContent(type='text',text=json.dumps(value))],structuredContent=value)
    m=FastMCP('local-zimage',instructions='Z-Image-Turbo Q6_K on the RTX 3060. Submit a job and poll until complete. Stop other GPU services first.')
    @m.tool()
    def generate_image(prompt:str,width:int=1024,height:int=1024,steps:int=8,cfg:float=1.0,seed:int=-1,client_request_id:str|None=None)->CallToolResult:
        """Queue an image generation job. Returns a job ID immediately."""
        return structured(call('POST','/jobs',json=Request(prompt=prompt,width=width,height=height,steps=steps,cfg=cfg,seed=seed,client_request_id=client_request_id).model_dump()))
    @m.tool()
    def get_image_job(job_id:str,include_preview:bool=False):
        """Get job status and output path. Set include_preview to attach an image after success."""
        from mcp.types import CallToolResult, TextContent, ImageContent
        result=call('GET','/jobs/'+job_id)
        content=[TextContent(type='text',text=json.dumps(result))]
        if include_preview and result['status']=='succeeded':
            import io
            from PIL import Image
            with Image.open(result['result']['image_path']) as im:
                im.thumbnail((768,768)); buf=io.BytesIO(); im.save(buf,format='PNG')
            content.append(ImageContent(type='image',data=base64.b64encode(buf.getvalue()).decode(),mimeType='image/png'))
        return CallToolResult(content=content,structuredContent=result)
    @m.tool()
    def cancel_image_job(job_id:str)->CallToolResult:
        """Cancel a queued or running job."""
        return structured(call('POST','/jobs/'+job_id+'/cancel'))
    @m.tool()
    def image_service_status()->CallToolResult:
        """Inspect model files and RTX 3060 memory."""
        return structured(call('GET','/health'))
    m.run(transport='stdio')

def main():
    p=argparse.ArgumentParser(prog='zimage'); sub=p.add_subparsers(dest='cmd',required=True)
    for x in ['serve','mcp','start','stop','doctor','setup-models']: sub.add_parser(x)
    g=sub.add_parser('generate'); g.add_argument('--prompt',required=True); g.add_argument('--width',type=int,default=1024); g.add_argument('--height',type=int,default=1024); g.add_argument('--steps',type=int,default=8); g.add_argument('--cfg',type=float,default=1.0); g.add_argument('--seed',type=int,default=-1); g.add_argument('--request-id'); g.add_argument('--wait',action='store_true'); g.add_argument('--out'); g.add_argument('--timeout',type=int,default=3600)
    for x in ['status','cancel','wait','download']:
        a=sub.add_parser(x); a.add_argument('job_id')
        if x=='download': a.add_argument('--out',required=True)
        if x=='wait': a.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    if a.cmd=='serve':
        import uvicorn
        with FileLock(str(DATA/'worker.lock'),timeout=0):
            server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8773,log_level='info')); app.state.server=server; server.run()
        return
    if a.cmd=='mcp': mcp_main(); return
    if a.cmd=='doctor':
        try:
            response=request('GET','/health'); response.raise_for_status(); result=response.json()
        except httpx.HTTPError: result=doctor()
    elif a.cmd=='setup-models':
        from assets import setup
        setup(); result=doctor()
    elif a.cmd=='stop':
        try: result=request('POST','/shutdown').json()
        except httpx.HTTPError: result={'status':'not_running'}
    elif a.cmd=='start': result=call('GET','/health')
    elif a.cmd=='generate':
        q=Request(prompt=a.prompt,width=a.width,height=a.height,steps=a.steps,cfg=a.cfg,seed=a.seed,client_request_id=a.request_id)
        result=call('POST','/jobs',json=q.model_dump())
        if a.wait:
            deadline=time.monotonic()+a.timeout
            while result['status'] in ('queued','running') and time.monotonic()<deadline:
                time.sleep(2); result=call('GET','/jobs/'+result['job_id'])
            if result['status']=='succeeded' and a.out: shutil.copyfile(result['result']['image_path'],a.out)
    elif a.cmd=='cancel': result=call('POST','/jobs/'+a.job_id+'/cancel')
    elif a.cmd=='status': result=call('GET','/jobs/'+a.job_id)
    elif a.cmd=='download':
        result=call('GET','/jobs/'+a.job_id)
        if result['status']!='succeeded': raise RuntimeError('Job has not succeeded')
        shutil.copyfile(result['result']['image_path'],a.out)
    elif a.cmd=='wait':
        deadline=time.monotonic()+a.timeout
        while True:
            result=call('GET','/jobs/'+a.job_id)
            if result['status'] not in ('queued','running') or time.monotonic()>=deadline: break
            time.sleep(2)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if result.get('status') in ('failed','cancelled','queued','running') and (a.cmd=='wait' or a.cmd=='generate' and a.wait): sys.exit(1)
if __name__=='__main__': main()

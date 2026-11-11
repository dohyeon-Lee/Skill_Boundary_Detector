"""Exercise the real DOM in Firefox; all API responses are local fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATIC = Path(__file__).resolve().parents[1] / "static"

FIXTURE = r"""
const fixtureState = {
  token:"fixture-token", config:{project_root:"/workspace/project",identity:"/key",interval:30}, error:"",
  updated_at:"2026-11-10T01:00:00Z",
  pods:[
    {id:"pod-one",name:"First Pod",status:"RUNNING",present:true,hourly_cost:.59,gpu_name:"A40",gpu_count:1,
     created_at:"2026-11-08T00:00:00Z",
     elapsed_seconds:5400,
     jobs:[{id:"10",name:"FTeval",state:"done",result_key:"pod-one:a",gpus:"0",started:"2026-11-10T00:00:00Z",finished:"2026-11-10T01:01:30Z",elapsed_seconds:3690},
       {id:"11",name:"Live eval",state:"running",gpus:"0",started:"2026-11-10T00:01:00Z",progress:{
         completed_episodes:76,total_episodes:200,percent:38,models:[
           {label:"Live checkpoint 30k",completed_episodes:51,total_episodes:100,percent:51,
            completed_tasks:4,total_tasks:10,successes:27,observed_episodes:51,success_pct:52.94,
            tasks:[{name:"task0",completed_episodes:10,total_episodes:10,success_pct:60},
                   {name:"task9",completed_episodes:0,total_episodes:10,success_pct:null}]}]}}]},
    {id:"pod-two",name:"Second Pod",status:"EXITED",present:true,gpu_name:"A40",gpu_count:1,
     created_at:"2026-11-09T00:00:00Z", collected_at:"2030-01-01T00:00:00Z",
     jobs:[{id:"10",name:"FTeval",state:"done"}]}],
  results:[
    {key:"pod-one:a",remote_id:"a",pod_id:"pod-one",name:"checkpoint 30k",phase:"final",mtime:3,
     output:"/eval/outputs/first",chart_url:"/must-not-request.png",job_ids:["10"],models:[
       {label:"First model",overall_pct:50,task_count:2,tasks:[{name:"task0",success_pct:100},{name:"task1",success_pct:0}]}]},
    {key:"pod-two:b",remote_id:"b",pod_id:"pod-two",name:"checkpoint 20k",phase:"final",mtime:2,
     job_ids:["10"],models:[{label:"Second model",overall_pct:75,task_count:1,tasks:[{name:"task0",success_pct:75}]}]},
    {key:"old:c",remote_id:"c",pod_id:"old",pod_name:"Archived Pod",name:"Archived eval",phase:"unverified",mtime:1,
     job_ids:[],models:[{label:"Archived model",overall_pct:null,task_count:1,tasks:[{name:"task0",success_pct:null}]}]}]
};
fixtureState.pods[0].jobs.push({id:"12",name:"DPbound",state:"done",result_key:"pod-one:boundary",started:"2026-11-10T00:00:00Z"});
fixtureState.pods[0].jobs.push({id:"13",name:"Predictor training",state:"running",training_progress:{
  completed_steps:79100,total_steps:100000,percent:79.1,updated_at:1794315000,runs:[
    {label:"predictor_mode2",completed_steps:79100,total_steps:100000,percent:79.1,latest_checkpoint:70000}]}});
fixtureState.results.push({key:"pod-one:boundary",remote_id:"boundary",pod_id:"pod-one",name:"Boundary comparison",
  phase:"final",kind:"skill_boundary_detection",job_ids:["12"],models:[],thresholds:{without_gripper:{
    groups:{dataset:{threshold:0.2,episodes:3921,points:77782}},calibration:{evaluation_episodes:200}}}});
const nativeFetch=window.fetch.bind(window), requests=[];
window.fetch=async (path,options={})=>{
  requests.push({path,options});
  if(path==="/runpod/api/state")return {ok:true,json:async()=>JSON.parse(JSON.stringify(fixtureState))};
  if(path==="/runpod/api/stop")return {ok:true,json:async()=>({message:"Stopped fixture"})};
  throw new Error("Unexpected API request: "+path);
};
window.confirm=()=>true;
window.setInterval=()=>0;
"""

CHECKS = r"""
(async()=>{
  const checks=JSON.parse(sessionStorage.getItem("checks")||"[]");
  const assert=(value,message)=>{if(!value)throw new Error(message);checks.push(message);};
  const tick=()=>new Promise(resolve=>setTimeout(resolve,30));
  const wait=async fn=>{for(let i=0;i<100;i++){if(fn())return;await tick();}throw new Error("UI did not become ready");};
  const detail=key=>[...document.querySelectorAll("details[data-disclosure-key]")].find(node=>node.dataset.disclosureKey===key);
  const firstJob=()=>detail("pod:pod-one:job:10");
  const firstModel=()=>firstJob().querySelector(".model-result");
  try {
    await wait(()=>document.querySelectorAll(".pod-card").length===3);
    if(!sessionStorage.getItem("reloaded")){
      assert(!document.getElementById("results"),"No separate results panel");
      assert(!document.querySelector("#pods img"),"No evaluation PNG image");
      assert(!requests.some(r=>r.path.includes("png")),"No PNG request");
      assert(firstJob().querySelector(".job-duration").textContent==="소요 1시간 1분 30초","Completed job duration is readable beside its name");
      assert(firstJob().textContent.includes("종료"),"Completed job includes its end timestamp");
      assert(detail("pod:pod-one").querySelector(".pod-duration").textContent==="전체 경과 1시간 30분 0초","Pod total duration is separate from job duration");
      const boundary=detail("pod:pod-one:job:12");
      assert(boundary.textContent.includes("threshold 0.2")&&boundary.textContent.includes("3921"),"Boundary threshold and calibration size appear under the job");
      assert(boundary.textContent.includes("보고서 대상 200"),"Boundary report subset stays separate from calibration");
      assert(!boundary.querySelector(".overall-row"),"Boundary comparison has no fabricated success rate");
      const live=detail("pod:pod-one:job:11");
      assert(live.querySelector(".job-progress-badge").textContent==="진행 38.0%","Live progress is visible beside the job name");
      assert(live.querySelector(".progress-count").textContent.includes("76 / 200"),"Progress shows completed and total episodes");
      assert(live.querySelector(".model-progress .overall-row .success-value").textContent==="52.9%","Intermediate success rate is separate from completion percentage");
      assert([...live.querySelectorAll(".success-value")].at(-1).textContent==="—","Pending tasks are unknown rather than failures");
      fixtureState.pods[0].jobs[1].progress.percent=40;
      fixtureState.pods[0].jobs[1].progress.completed_episodes=80;
      await load();await tick();
      assert(detail("pod:pod-one:job:11").querySelector(".job-progress-badge").textContent==="진행 40.0%","Polling updates live evaluation progress");
      const training=detail("pod:pod-one:job:13");
      assert(training.querySelector(".job-training-badge").textContent==="학습 79.1%","Training progress is visible beside the job name");
      assert(training.querySelector(".progress-count").textContent.includes("79,100 / 100,000 step"),"Training shows current and target steps");
      assert(training.querySelector(".training-checkpoint").textContent.includes("70,000 step"),"Saved checkpoint is separate from live step");
      assert(!training.querySelector(".overall-row"),"Training does not fabricate an evaluation success rate");
      fixtureState.pods[0].jobs.find(job=>job.id==="13").training_progress.percent=80;
      await load();await tick();
      assert(detail("pod:pod-one:job:13").querySelector(".job-training-badge").textContent==="학습 80.0%","Polling updates training progress");
      assert(document.querySelector(".pod-card").dataset.disclosureKey==="pod:pod-one","Recent job use sorts above newer Pod creation and collection time");
      fixtureState.pods[1].jobs[0].started="2026-11-10T02:00:00Z";
      await load();await tick();
      assert(document.querySelector(".pod-card").dataset.disclosureKey==="pod:pod-two","Restarted older Pod moves to the top when a new job starts");
      delete fixtureState.pods[1].jobs[0].started;
      await load();await tick();
      assert(firstJob().querySelector(".eval-result .overall-row .success-value").textContent==="50.0%","Overall is inside its own job");
      assert(firstModel().querySelectorAll(".task-row").length===3,"Overall and task rates share the same list");
      assert(detail("pod:pod-two:job:10").querySelector(".overall-row .success-value").textContent==="75.0%","Same job IDs on different Pods do not mix results");
      assert(detail("pod:old").querySelector(".job-card .overall-row .success-value").textContent==="—","Archived result survives absent Pod/job records");
      assert(detail("pod:pod-two").querySelector("button").disabled,"Stopped Pod cannot be stopped again");
      firstJob().querySelector("summary").click();
      await tick();
      assert(firstJob().open,"Job result opens through native summary toggle");
      firstModel().querySelector("summary").click();
      detail("pod:pod-one:costs").querySelector("summary").click();
      await tick();
      fixtureState.results[0].models[0].overall_pct=80;
      await load();await tick();
      assert(firstJob().open&&!firstModel().open,"Polling preserves nested open and closed states");
      assert(!detail("pod:pod-one:costs").open,"Polling preserves collapsed cost section");
      assert(firstModel().querySelector(".overall-row .success-value").textContent==="80.0%","Numeric result updates even when Pod data is unchanged");
      document.getElementById("search").value="checkpoint 30k";
      document.getElementById("search").dispatchEvent(new Event("input"));
      assert(document.querySelectorAll(".pod-card").length===1,"Search finds evaluations inside jobs");
      document.getElementById("search").value="";
      document.getElementById("search").dispatchEvent(new Event("input"));
      await tick();
      assert(firstJob().open&&!firstModel().open,"Search preserves disclosure preferences");
      sessionStorage.setItem("checks",JSON.stringify(checks));
      sessionStorage.setItem("reloaded","yes");
      location.reload();return;
    }
    assert(firstJob().open&&!firstModel().open,"Page reload restores disclosure preferences");
    assert(firstJob().querySelector(".job-duration").textContent==="소요 1시간 1분 30초","Completed duration survives page reload");
    document.getElementById("collapse-all").click();await tick();
    assert([...document.querySelectorAll("main details")].every(node=>!node.open),"Collapse all includes Pods, jobs, results and page sections");
    document.getElementById("expand-all").click();await tick();
    assert([...document.querySelectorAll("main details")].every(node=>node.open),"Expand all restores every section");
    detail("pod:pod-one").querySelector("button").click();
    await wait(()=>requests.some(r=>r.path==="/runpod/api/stop"));await tick();
    const request=requests.find(r=>r.path==="/runpod/api/stop");
    assert(JSON.parse(request.options.body).pod_id==="pod-one","Stop action stays bound to the selected Pod");
    assert(request.options.headers["X-Sync-Token"]==="fixture-token","Stop action retains its authentication token");
    assert(detail("pod:pod-one").open,"Stop button does not collapse the Pod");
    assert(document.documentElement.scrollWidth<=document.documentElement.clientWidth+1,"Layout has no horizontal overflow");
    await nativeFetch("/report",{method:"POST",body:JSON.stringify({checks,width:innerWidth})});
  } catch(error){await nativeFetch("/report",{method:"POST",body:JSON.stringify({checks,error:String(error)})});}
})();
"""


@unittest.skipUnless(shutil.which("firefox"), "Firefox is required for native disclosure tests")
class RunPodUITests(unittest.TestCase):
    def test_nested_results_and_disclosure_state_in_browser(self):
        reported = threading.Event()
        reports = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    html = (STATIC / "index.html").read_text()
                    html = html.replace("</head>", "<script>" + FIXTURE + "</script></head>")
                    html = html.replace("</body>", '<script src="/checks.js" defer></script></body>')
                    body, mime = html.encode(), "text/html"
                elif self.path == "/checks.js":
                    body, mime = CHECKS.encode(), "text/javascript"
                elif self.path in ("/runpod/app.js", "/runpod/style.css"):
                    body = (STATIC / self.path.rsplit("/", 1)[-1]).read_bytes()
                    mime = "text/javascript" if self.path.endswith(".js") else "text/css"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                reports.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                reported.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="runpod-ui-") as temp:
                profile = Path(temp) / "profile"
                profile.mkdir()
                (profile / "user.js").write_text('user_pref("browser.shell.checkDefaultBrowser", false);\n'
                    'user_pref("datareporting.policy.dataSubmissionEnabled", false);\n'
                    'user_pref("toolkit.telemetry.enabled", false);\n')
                with (Path(temp) / "browser.log").open("w+") as log:
                    process = subprocess.Popen([shutil.which("firefox"), "--headless", "--no-remote",
                        "--profile", str(profile), "--width", "640", "--height", "1000",
                        f"http://127.0.0.1:{server.server_port}/"], stdout=log, stderr=log,
                        env={**os.environ, "MOZ_HEADLESS": "1"})
                    try:
                        received = reported.wait(40)
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    log.seek(0)
                    self.assertTrue(received, log.read()[-3000:])
                self.assertNotIn("error", reports[0], reports[0])
                self.assertGreaterEqual(len(reports[0]["checks"]), 18)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

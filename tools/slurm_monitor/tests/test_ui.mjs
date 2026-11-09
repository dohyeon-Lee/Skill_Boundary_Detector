// Exercise the real Slurm UI in both its standalone and unified namespaces.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
  constructor(){this.children=[];this.listeners={};this.value="";this.checked=false;this.textContent="";
    this.classList={add(){},remove(){}};this.selectors=new Map();}
  append(...children){this.children.push(...children);}
  setAttribute(name,value){this[name]=value;}
  replaceChildren(...children){this.children=children;}
  insertBefore(child,before){for(const list of lists){const old=list.children.indexOf(child);if(old>=0)list.children.splice(old,1);}
    const index=this.children.indexOf(before);this.children.splice(index<0?this.children.length:index,0,child);}
  addEventListener(name,handler){this.listeners[name]=handler;}
  querySelector(selector){if(!this.selectors.has(selector))this.selectors.set(selector,new Element());return this.selectors.get(selector);}
  cloneNode(){return new Element();}
}
let lists=[];
const html=fs.readFileSync(new URL("../static/index.html",import.meta.url),"utf8");
const source=fs.readFileSync(new URL("../static/app.js",import.meta.url),"utf8");
const preferences=new Map();
for(const prefix of ["","/slurm"]){
  const ids=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const document={getElementById:id=>ids.get(id)||null,createElement:()=>new Element()};
  ids.get("job-template").content={firstElementChild:new Element()};
  ids.get("settings-form").elements={namedItem:()=>new Element()};
  lists=["running","pending","ended","training-running","training-ended"].map(id=>ids.get(id));
  const allocation={key:"123@start",job_id:"123",job_name:"allocation",state:"RUNNING",group:"running",terminal:false,
    display_name:"GPU allocation",delivery:{status:"none"},can_cancel:true,outputs:[],prompt:"saved allocation prompt",armed:false};
  const trainer={...allocation,key:"training:one",kind:"training",group:"training-running",can_cancel:false,
    display_name:"model",outputs:["/repo/model"],allocation_state:"RUNNING",prompt:""};
  const backend={jobs:[allocation,trainer],connection:{configured:true},config:{interval:30,user:"test"},token:"monitor-token",
    gpu:{updated_at:"2026-11-09T01:00:00Z",interval:15,errors:[],devices:[{node:"node200",index:6,name:"NVIDIA B200",
      used_gib:80.5,free_gib:98.5,total_gib:179,job_ids:["123"],updated_at:"2026-11-09T01:00:00Z",stale:false}]}};
  const requests=[];
  const context=vm.createContext({document,location:{pathname:prefix+"/"},confirm:()=>true,console,
    localStorage:{getItem:key=>preferences.get(key)||null,setItem:(key,value)=>preferences.set(key,value)},
    setTimeout:()=>0,clearTimeout(){},setInterval:()=>0,
    fetch:async(path,options={})=>{
      requests.push(path);
      assert.ok(path.startsWith(prefix+"/api/"),path);
      if(options.method==="POST"){
        assert.equal(options.headers["X-Monitor-Token"],backend.token);
        const body=JSON.parse(options.body);
        if(path.endsWith("/job"))Object.assign(backend.jobs.find(job=>job.key===body.key),body.values);
        if(path.endsWith("/cancel"))assert.equal(body.key,allocation.key);
      }
      return {ok:true,json:async()=>path.endsWith("/state")?JSON.parse(JSON.stringify(backend)):{ok:true}};
    }});
  vm.runInContext(source,context);
  await new Promise(resolve=>setTimeout(resolve,10));
  assert.equal(ids.get("count-running").textContent,1);
  assert.equal(ids.get("label-training-running").textContent,1);
  const gpuCard=ids.get("training-running").children[0].querySelector(".job-gpus").children[0];
  assert.match(gpuCard.children[0].textContent,/node200 · GPU 6/);
  assert.match(gpuCard.children[1].textContent,/사용 80.5 GiB.*남음 98.5 GiB.*전체 179.0 GiB/);
  assert.match(ids.get("running").children[0].querySelector(".job-gpus").children[0].children[1].textContent,/사용 80.5 GiB/);
  assert.match(gpuCard.children[3].textContent,/GPU 전체 사용량 · 공유 학습 합산/);
  backend.gpu.devices[0].stale=true;backend.gpu.errors=["node200: timeout"];
  await vm.runInContext("load()",context);
  assert.match(ids.get("training-running").children[0].querySelector(".job-gpus").children[0].children[3].textContent,/조회 지연 · 마지막 값/);
  const section=ids.get("section-training-running");
  if(prefix)assert.equal(section.open,false,"collapsed preference must survive a new page load");
  section.open=false;section.listeners.toggle();
  assert.equal(JSON.parse(preferences.get("sbd.slurm.sections.v1"))["training-running"],false);
  await vm.runInContext("load()",context);
  assert.equal(section.open,false,"polling must not reopen a collapsed section");
  assert.equal(ids.get("label-training-running").textContent,1,"count remains visible when collapsed");
  const card=ids.get("training-running").children[0];
  assert.equal(card.querySelector(".cancel").hidden,true);
  card.querySelector(".prompt").value="model completion instruction";
  card.querySelector(".armed").checked=true;
  card.querySelector(".prompt").listeners.input();
  await card.querySelector(".save").listeners.click({currentTarget:card.querySelector(".save")});
  assert.equal(trainer.prompt,"model completion instruction");
  assert.equal(allocation.prompt,"saved allocation prompt");
  trainer.state="WAITING_FOR_MEMORY";trainer.reason="newest trainer paused";
  await vm.runInContext("load()",context);
  assert.equal(card.querySelector(".state").textContent,"VRAM 대기");
  assert.equal(card.querySelector(".send").hidden,true,"resource wait must not offer failure resubmission");
  trainer.state="RUNNING";
  const allocCard=ids.get("running").children[0];
  await allocCard.querySelector(".cancel").listeners.click();
  assert.ok(requests.includes(prefix+"/api/cancel"));
  trainer.terminal=true;trainer.state="COMPLETED";trainer.group="training-ended";
  await vm.runInContext("load()",context);
  assert.equal(ids.get("training-running").children.length,0);
  assert.equal(ids.get("training-ended").children.length,1);
  assert.equal(card.querySelector(".job-gpus").hidden,true,"completed training must not display current shared GPU usage");
  allocation.job_id="456";
  await vm.runInContext("load()",context);
  assert.equal(allocCard.querySelector(".job-gpus").children.length,1);
  assert.equal(allocCard.querySelector(".job-gpus").children[0].textContent,"VRAM 조회 지연 · 다음 확인을 기다리는 중","never display another allocation's VRAM");
}
console.log("Slurm UI passed: isolated routes, trainer prompts, allocation cancellation, completion display and persistent section toggles.");

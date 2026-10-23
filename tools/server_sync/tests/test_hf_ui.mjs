// Run the real UI with a tiny DOM and fake HTTP boundary; never transfer real data.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
  constructor(tag="div") {
    this.tagName=tag;this.children=[];this.listeners={};this.dataset={};this.value="";this.checked=false;
    this.disabled=false;this.open=false;this.textContent="";this.classes=new Set();
    this.classList={add:name=>this.classes.add(name),toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name)};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=[...children];}
  addEventListener(name,handler){(this.listeners[name]??=[]).push(handler);}
  async dispatch(name){for(const handler of this.listeners[name]??[])await handler({target:this});}
  setAttribute(name,value){this[name]=value;}
  showModal(){this.open=true;}
  close(){this.open=false;}
  scrollIntoView(){}
  querySelectorAll(selector){return this.children.flatMap(child=>[
    ...(selector==="details[open]"&&child.tagName==="details"&&child.open?[child]:[]),...child.querySelectorAll(selector)]);}
}
const html=fs.readFileSync(new URL("../static/hf_index.html",import.meta.url),"utf8");
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
const document={getElementById:id=>{assert.ok(elements.has(id),"Missing HTML element: "+id);return elements.get(id);},createElement:tag=>new Element(tag)};
const $=id=>document.getElementById(id);
const pairs={"dataset-pull":["hub-datasets","local-datasets"],"dataset-push":["local-datasets","hub-datasets"],
  "models-pull":["hub-models","local-models"],"checkpoints-pull":["hub-checkpoints","local-checkpoints"],"watch":["local-checkpoints","hub-checkpoints"],
  "dataset-delete":["hub-datasets","hub-datasets"],"checkpoints-delete":["hub-checkpoints","hub-checkpoints"]};
const config={token:"session",mode:"runpod",authenticated:true,concurrency:2,
  routes:Object.entries(pairs).map(([id,[source,destination]])=>({id,source,destination,name:id})),
  endpoints:[...new Set(Object.values(pairs).flat())].map(id=>({id,name:id,base:"/"+id}))};
const requests=[];
let jobs=[];
const missingFolders=new Set(),deniedFolders=new Set(),recoveredPaths=new Map();
const fetch=async(path,options={})=>{
  const body=options.body?JSON.parse(options.body):undefined;requests.push({path,body,options});let payload;
  if(path==="/api/config")payload=config;
  else if(path==="/api/jobs")payload={jobs};
  else if(path.startsWith("/api/browse?")){
    const query=new URLSearchParams(path.split("?")[1]),folder=query.get("path"),offset=Number(query.get("offset"));
    if(missingFolders.has(folder))return {ok:false,json:async()=>({error:'Entry Not Found: '+folder+' does not exist on "main"'})};
    if(deniedFolders.has(folder))return {ok:false,json:async()=>({error:"Repository Not Found; check credentials"})};
    const resolved=recoveredPaths.get(folder)??folder,resolvedOffset=resolved!==folder?0:offset;
    payload={path:resolved,offset:resolvedOffset,limit:200,total:resolvedOffset+2,has_more:resolvedOffset===0,
      entries:[{name:"050000",path:resolved?resolved+"/050000":"dataset_filtered",kind:"directory",size:null},
               {name:"sample.txt",path:(resolved||"dataset_filtered")+"/sample.txt",kind:"file",size:12}]};
  }else if(path==="/api/plan"){
    assert.equal(options.headers["X-Sync-Token"],"session");
    payload={collapsed:0,notice:"공개 체크포인트 / 미리보기 확인",items:body.paths.map(path=>({path,source:"/from/"+path,destination:"/to/"+path}))};
    if(body.source===body.destination)payload={collapsed:0,notice:"선택한 Hub 파일 삭제 · 로컬 파일 유지",plan_id:"reviewed-delete",
      items:[{path:"선택한 항목 삭제",repo:"me/data",paths:body.paths,source:"https://huggingface.co/datasets/me/data",destination:"Hub에서 삭제",
        file_count:3,size:1000,sample_files:body.paths.map(path=>path+"/weights.bin")}]};
  }else if(path==="/api/transfers")payload={ids:["queued"]};
  else if(path==="/api/cancel")payload={ok:true};
  else throw Error(path);
  return {ok:true,json:async()=>payload};
};
const context=vm.createContext({document,fetch,URLSearchParams,console,confirm:()=>true,setInterval:()=>0,setTimeout,clearTimeout});
vm.runInContext(fs.readFileSync(new URL("../static/hf_app.js",import.meta.url),"utf8"),context);
await new Promise(resolve=>setTimeout(resolve,20));
const run=code=>vm.runInContext(code,context);
assert.equal(run("state.source"),"hub-datasets");
assert.equal($("auth").textContent,"● 서버 HF 토큰 설정됨");
for(let index=1;index<=3;index++){
  await run(`browse("dataset_filtered/run-${index}")`);
  const checkbox=$("file-list").children[0].children[0];checkbox.checked=true;await checkbox.dispatch("change");
}
assert.equal(run("state.selected.size"),3,"navigation must preserve selections");
assert.equal($("next").disabled,false);
await $("next").dispatch("click");
assert.equal(run("state.offset"),200);assert.equal($("next").disabled,true);
$("dry-run").checked=true;
await $("prepare").dispatch("click");
assert.equal($("confirmation").open,true);
assert.equal(requests.filter(request=>request.path==="/api/transfers").length,0,"plan must not start transfers");
await $("start").dispatch("click");
const transfer=requests.find(request=>request.path==="/api/transfers");
assert.equal(transfer.body.paths.length,3);assert.equal(transfer.body.dry_run,true);
assert.equal(transfer.body.source,"hub-datasets");assert.equal(transfer.body.destination,"local-datasets");
assert.equal($("confirmation").open,false);assert.equal(run("state.selected.size"),0);

await run('changeOperation("watch")');
assert.ok($("browse-panel").classes.has("hidden"));
assert.ok(!$("watch-panel").classes.has("hidden"));
$("watch-root").value="outputs_filtered";$("watch-interval").value="60";$("watch-keep").value="3";
$("watch-protect").value="050000,100000";$("watch-prune").checked=true;
await $("prepare-watch").dispatch("click");
assert.equal($("confirmation").open,true);
assert.match($("notice").textContent,/공개/);
await $("start").dispatch("click");
const watch=requests.filter(request=>request.path==="/api/transfers").at(-1);
assert.equal(watch.body.options.keep,3);assert.equal(watch.body.options.prune_local,true);
assert.equal(watch.body.options.root,"outputs_filtered");assert.equal(watch.body.dry_run,true);
assert.equal(watch.body.options.squash,false);

jobs=[{id:"watch-1",path:"outputs_filtered",route:"watch",source_id:"local-checkpoints",destination_id:"hub-checkpoints",
  source_name:"server",destination_name:"hub",status:"running",progress:99,detail:"uploading",logs:["weights 100%"],dry_run:false,options:watch.body.options}];
await run("pollJobs()");
const card=$("jobs").children[0],bottom=card.children[2];
assert.equal(bottom.children[0].value,"","file progress must not claim whole-job completion");
await bottom.children[1].dispatch("click");
assert.equal(requests.at(-2).path,"/api/cancel");
assert.equal(requests.at(-2).body.id,"watch-1");

jobs=[];
await run('changeOperation("dataset-delete")');
$("dry-run").checked=false;
await run('browse("dataset_filtered/a")');
const deletionCheckbox=$("file-list").children[0].children[0];deletionCheckbox.checked=true;await deletionCheckbox.dispatch("change");
assert.match($("prepare").textContent,/삭제 준비/);
await $("prepare").dispatch("click");
assert.equal($("start").textContent,"Hub에서 삭제");
assert.equal($("start").disabled,true,"real deletion must require the explicit confirmation checkbox");
assert.equal($("confirm-delete").checked,false);
const beforeDelete=requests.filter(request=>request.path==="/api/transfers").length;
await $("start").dispatch("click");
assert.equal(requests.filter(request=>request.path==="/api/transfers").length,beforeDelete);
assert.match($("confirmation-list").children[0].children[1].textContent,/3개 파일/);
$("confirm-delete").checked=true;await $("confirm-delete").dispatch("change");
assert.equal($("start").disabled,false);
await $("start").dispatch("click");
const deletion=requests.filter(request=>request.path==="/api/transfers").at(-1);
assert.equal(deletion.body.plan_id,"reviewed-delete");assert.equal(deletion.body.confirm_delete,true);
assert.equal(deletion.body.paths[0],"dataset_filtered/a/050000");

await run('changeOperation("checkpoints-delete")');
$("dry-run").checked=true;
await run('browse("outputs_filtered/run/checkpoints")');
const previewCheckbox=$("file-list").children[0].children[0];previewCheckbox.checked=true;await previewCheckbox.dispatch("change");
await $("prepare").dispatch("click");
assert.equal($("start").disabled,false,"dry-run deletion does not require destructive confirmation");
assert.ok($("delete-confirmation").classes.has("hidden"));
await $("start").dispatch("click");
const preview=requests.filter(request=>request.path==="/api/transfers").at(-1);
assert.equal(preview.body.dry_run,true);assert.equal(preview.body.confirm_delete,false);
jobs=[{id:"deleted-1",path:"outputs_filtered/run/checkpoints/050000",paths:["outputs_filtered/run/checkpoints/050000"],
  route:"checkpoints-delete",source_id:"hub-checkpoints",destination_id:"hub-checkpoints",source_name:"hub",destination_name:"Hub에서 삭제",
  status:"completed",progress:100,detail:"deleted",logs:[],dry_run:false,options:{}}];
await run('browse("outputs_filtered/run/checkpoints/050000")');
await run("pollJobs()");
assert.equal(run("state.path"),"outputs_filtered/run/checkpoints","deleting the current folder should refresh its parent");

jobs=[];
const removed="outputs_filtered/skillVLA_stage1/VSA";
for(const folder of [removed,"outputs_filtered/skillVLA_stage1","outputs_filtered"])missingFolders.add(folder);
await run('state.selected.add("outputs_filtered/skillVLA_stage1/VSA/old");state.selected.add("outputs_other/keep")');
$("search").value="old-search";
const beforeFallback=requests.length;
await run(`browse(${JSON.stringify(removed)},200,true)`);
const probes=requests.slice(beforeFallback).filter(request=>request.path.startsWith("/api/browse?"));
assert.deepEqual(probes.map(request=>new URLSearchParams(request.path.split("?")[1]).get("path")),
  [removed,"outputs_filtered/skillVLA_stage1","outputs_filtered",""]);
assert.equal(run("state.path"),"");assert.equal(run("state.offset"),0);
assert.equal($("search").value,"");assert.equal($("error").textContent,"");
assert.match($("browser-notice").textContent,/저장소 루트/);
assert.equal(run('state.selected.has("outputs_filtered/skillVLA_stage1/VSA/old")'),false);
assert.equal(run('state.selected.has("outputs_other/keep")'),true);
for(const probe of probes.slice(1)){
  const query=new URLSearchParams(probe.path.split("?")[1]);assert.equal(query.get("offset"),"0");assert.equal(query.get("search"),"");
}
missingFolders.clear();
recoveredPaths.set(removed,"outputs_filtered");
await run(`browse(${JSON.stringify(removed)})`);
assert.equal(run("state.path"),"outputs_filtered","UI must follow a recovered path returned by a newer backend");
assert.equal($("breadcrumbs").children.at(-1).textContent,"outputs_filtered");
recoveredPaths.clear();
deniedFolders.add(removed);const beforeDenied=requests.length;
await run(`browse(${JSON.stringify(removed)})`);
assert.equal(requests.length-beforeDenied,1,"auth/repository failures must not silently navigate to a parent");
assert.match($("error").textContent,/Repository Not Found/);
console.log("HF UI passed: selection, deletion confirmation, multi-level parent recovery, filter/selection cleanup, auth errors.");

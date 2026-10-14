// Exercise the actual app.js selection/navigation/confirmation flow without a browser dependency.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

class Element {
  constructor(tag="div") {this.tagName=tag;this.children=[];this.listeners={};this.dataset={};this.value="";this.checked=false;this.disabled=false;this.open=false;this.textContent="";this.classes=new Set();
    this.classList={add:(name)=>this.classes.add(name),toggle:(name,enabled)=>enabled?this.classes.add(name):this.classes.delete(name)};}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=[...children];}
  addEventListener(name,handler){(this.listeners[name]??=[]).push(handler);}
  async dispatch(name){for(const handler of this.listeners[name]??[])await handler({target:this});}
  setAttribute(name,value){this[name]=value;}
  showModal(){this.open=true;}
  close(){this.open=false;}
  scrollIntoView(){}
  querySelectorAll(selector){const result=[];for(const child of this.children){if(selector==="details[open]"&&child.tagName==="details"&&child.open)result.push(child);result.push(...child.querySelectorAll(selector));}return result;}
}
const ids=new Map();
const requests=[];
const document={getElementById:(id)=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id);},createElement:(tag)=>new Element(tag)};
const config={token:"test-token",mode:"rllab",concurrency:2,streams:8,endpoints:[{id:"a",name:"data1",base:"/test/source",host:""},{id:"b",name:"scratch2",base:"/test/destination",host:"user@host"}]};
const fetch=async(path,options={})=>{
  const body=options.body?JSON.parse(options.body):undefined;
  requests.push({path,body,options});let payload;
  if(path==="/api/config")payload=config;
  else if(path==="/api/jobs")payload={jobs:[]};
  else if(path.startsWith("/api/browse?")){
    const query=new URLSearchParams(path.split("?")[1]),relative=query.get("path");
    payload={path:relative,total:3,offset:0,limit:200,entries:["050000","100000","last"].map(name=>({name,path:relative?relative+"/"+name:name,kind:name==="last"?"symlink":"directory",size:null}))};
  }else if(path==="/api/plan"){
    assert.equal(options.headers["X-Sync-Token"],"test-token");
    payload={collapsed:0,items:body.paths.map(path=>({path,source:"/test/source/"+path,destination:"/test/destination/"+path}))};
  }else if(path==="/api/transfers")payload={ids:body.paths.map((_,index)=>String(index))};
  else throw Error("Unexpected request "+path);
  return {ok:true,json:async()=>payload};
};
const context=vm.createContext({document,fetch,URLSearchParams,console,confirm:()=>true,setInterval:()=>0,setTimeout,clearTimeout});
const app=new URL("../static/app.js",import.meta.url);
vm.runInContext(fs.readFileSync(fileURLToPath(app),"utf8"),context);
await new Promise(resolve=>setTimeout(resolve,10));
for(let variant=1;variant<=6;variant++){
  const folder=`outputs_filtered/skillVLA_stage1/VSA/residual_${variant}/checkpoints`;
  await vm.runInContext(`browse(${JSON.stringify(folder)})`,context);
  const checkbox=document.getElementById("file-list").children[0].children[0];
  checkbox.checked=true;await checkbox.dispatch("change");
  assert.equal(vm.runInContext("state.selected.size",context),variant,"selection must survive folder changes");
  assert.equal(document.getElementById("file-list").children[2].children[0].disabled,true,"last symlink is not selectable");
}
assert.equal(document.getElementById("selected-count").textContent,6);
await document.getElementById("prepare").dispatch("click");
assert.equal(document.getElementById("confirmation").open,true);
assert.equal(requests.filter(request=>request.path==="/api/transfers").length,0,"preview must not copy");
await document.getElementById("start").dispatch("click");
const transfer=requests.find(request=>request.path==="/api/transfers");
assert.equal(transfer.body.paths.length,6);
assert.ok(transfer.body.paths.every(path=>path.endsWith("/050000")));
assert.equal(transfer.body.source,"a");assert.equal(transfer.body.destination,"b");
assert.equal(vm.runInContext("state.selected.size",context),0);
assert.equal(document.getElementById("confirmation").open,false);
console.log("UI test passed: six 50k checkpoints selected across folders, confirmed, queued together.");

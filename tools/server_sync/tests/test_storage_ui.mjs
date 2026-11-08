import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
  constructor(){this.children=[];this.textContent="";this.listeners={};}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(name,value){this[name]=value;}
  addEventListener(name,handler){this.listeners[name]=handler;}
}
const ids=Object.fromEntries(["storage-host","storage-volumes","storage-refresh"].map(id=>[id,new Element()]));
let calls=0,interval=0;
const context=vm.createContext({
  document:{getElementById:id=>ids[id],createElement:()=>new Element()},
  fetch:async path=>{assert.equal(path,"/api/storage");calls++;return {ok:true,json:async()=>({hostname:"node-test",volumes:[
    {name:"data",path:"/data",total:2**40,used:2**39,free:2**39,used_percent:50},
    {name:"other",path:"/unavailable",error:"unavailable"}]})};},
  setInterval:(_,ms)=>{interval=ms;},
});
vm.runInContext(fs.readFileSync(new URL("../static/storage.js",import.meta.url),"utf8"),context);
await new Promise(resolve=>setTimeout(resolve,10));
assert.equal(ids["storage-host"].textContent,"node-test · 디스크 용량");
assert.equal(ids["storage-volumes"].children.length,2);
assert.equal(ids["storage-volumes"].children[0].children.at(-1).textContent,"남음 512.0 GiB");
assert.equal(interval,60000);
await ids["storage-refresh"].listeners.click();assert.equal(calls,2);
assert.equal(ids["storage-refresh"].disabled,false);
console.log("Storage UI passed: host, capacity, remaining space, per-volume errors and refresh.");

"use strict";
(() => {
  const host=document.getElementById("storage-host"),list=document.getElementById("storage-volumes"),refresh=document.getElementById("storage-refresh");
  if(!host||!list||!refresh)return;
  let pending=false;
  function element(tag,text,className) {
    const item=document.createElement(tag);if(text!==undefined)item.textContent=text;
    if(className)item.className=className;return item;
  }
  function size(bytes) {
    const units=["B","KiB","MiB","GiB","TiB"];let i=0;
    while(bytes>=1024&&i<units.length-1){bytes/=1024;i++;}
    return bytes.toFixed(i?1:0)+" "+units[i];
  }
  async function update() {
    if(pending)return;pending=true;refresh.disabled=true;
    try {
      const response=await fetch("/api/storage");const data=await response.json();
      if(!response.ok)throw Error(data.error||"조회 실패");
      host.textContent=data.hostname+" · 디스크 용량";list.replaceChildren();
      for(const volume of data.volumes) {
        const card=element("div",undefined,"storage-volume");card.append(element("strong",volume.name),element("code",volume.path,"storage-path"));
        if(volume.error){card.append(element("span",volume.error));list.append(card);continue;}
        const progress=element("progress");progress.max=100;progress.value=volume.used_percent;
        progress.setAttribute("aria-label",volume.name+" 사용률 "+volume.used_percent+"%");
        card.append(progress,element("span",`전체 ${size(volume.total)} · 사용 ${size(volume.used)} (${volume.used_percent}%)`),
          element("strong","남음 "+size(volume.free),"storage-free"));list.append(card);
      }
    } catch(error){list.replaceChildren(element("span","용량 조회 실패: "+error.message));}
    finally{pending=false;refresh.disabled=false;}
  }
  refresh.addEventListener("click",update);update();setInterval(update,60000);
})();

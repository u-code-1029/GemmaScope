'use strict';

// A drag is always calculated from its initial selection. Moving over an item
// repeatedly therefore cannot toggle it repeatedly during the same gesture.
const selectionUI={drag:null,frame:0,suppressUntil:0,lastDragAt:-Infinity,releasePointerId:null};
const selectionThreshold=5;

function selectionIsDragging() { return !!selectionUI.drag?.active; }

function selectionSuppressClick(event) {
  const now=performance.now();
  const suppressed=event.type==='dblclick'
    ? now-selectionUI.lastDragAt<700
    : event.detail>0&&now<selectionUI.suppressUntil;
  if(!suppressed)return false;
  event.preventDefault();event.stopImmediatePropagation();return true;
}

function selectionArmSuppression(pointerId) {
  selectionUI.suppressUntil=performance.now()+700;
  selectionUI.lastDragAt=performance.now();
  selectionUI.releasePointerId=pointerId;
}

function selectionContextMatches(drag) {
  return state.view==='dataset'&&state.project.root===drag.project&&
    state.dataRequest===drag.request&&drag.grid.isConnected;
}

function selectionCanRestore(drag) {
  return state.view==='dataset'&&state.project.root===drag.project&&
    state.dataset.filter===drag.filter&&state.dataset.query===drag.query&&state.dataset.offset===drag.offset&&
    state.dataset.items.length===drag.cards.length&&state.dataset.items.every((item,index)=>String(item.id)===drag.cards[index].dataset.imageId);
}

function selectionCancel({restore=true}={}) {
  const drag=selectionUI.drag;if(!drag)return;
  selectionUI.drag=null;
  cancelAnimationFrame(selectionUI.frame);selectionUI.frame=0;
  drag.grid.classList.remove('drag-selecting');drag.marquee?.remove();
  if(drag.grid.hasPointerCapture?.(drag.pointerId))drag.grid.releasePointerCapture(drag.pointerId);
  if(drag.active) {
    selectionArmSuppression(drag.pointerId);
    if(restore&&selectionCanRestore(drag)) {
      state.selected=new Set(drag.initial);libraryUI.focusId=drag.focusId;libraryUI.anchorId=drag.anchorId;
      librarySyncSelection();
    }
  }
}

function selectionScrollParents(grid) {
  const parents=[];
  for(let current=grid;current&&current!==document.scrollingElement;current=current.parentElement) {
    const style=getComputedStyle(current);
    if((/(auto|scroll)/.test(style.overflowY)&&current.scrollHeight>current.clientHeight)||
       (/(auto|scroll)/.test(style.overflowX)&&current.scrollWidth>current.clientWidth))parents.push(current);
  }
  if(document.scrollingElement)parents.push(document.scrollingElement);
  return parents;
}

function selectionVisibleRect(element) {
  const rect=element===document.scrollingElement
    ? {left:0,top:0,right:window.innerWidth,bottom:window.innerHeight}
    : element.getBoundingClientRect();
  return {left:Math.max(0,rect.left),top:Math.max(0,rect.top),
    right:Math.min(window.innerWidth,rect.right),bottom:Math.min(window.innerHeight,rect.bottom)};
}

function selectionEdgeVelocity(position,start,end) {
  if(end<=start)return 0;
  const edge=Math.min(48,(end-start)/4);
  if(position<start+edge)return -Math.min(20,Math.max(0,(start+edge-position)/edge)*20);
  if(position>end-edge)return Math.min(20,Math.max(0,(position-end+edge)/edge)*20);
  return 0;
}

function selectionAutoScroll(drag,elapsed) {
  let movedX=false,movedY=false;
  for(const element of drag.scrollParents) {
    const rect=selectionVisibleRect(element),isWindow=element===document.scrollingElement;
    if(rect.right<=rect.left||rect.bottom<=rect.top)continue;
    const withinX=isWindow||drag.clientX>=rect.left-48&&drag.clientX<=rect.right+48;
    const withinY=isWindow||drag.clientY>=rect.top-48&&drag.clientY<=rect.bottom+48;
    const dx=!movedX&&withinY?selectionEdgeVelocity(drag.clientX,rect.left,rect.right)*elapsed:0;
    const dy=!movedY&&withinX?selectionEdgeVelocity(drag.clientY,rect.top,rect.bottom)*elapsed:0;
    if(dx){const before=element.scrollLeft;element.scrollLeft+=dx;movedX=element.scrollLeft!==before;}
    if(dy){const before=element.scrollTop;element.scrollTop+=dy;movedY=element.scrollTop!==before;}
    if(movedX&&movedY)break;
  }
}

function selectionRenderDrag(drag) {
  const gridRect=drag.grid.getBoundingClientRect();
  const pointerX=drag.clientX-gridRect.left+drag.grid.scrollLeft;
  const pointerY=drag.clientY-gridRect.top+drag.grid.scrollTop;
  const left=Math.min(drag.startX,pointerX),right=Math.max(drag.startX,pointerX);
  const top=Math.min(drag.startY,pointerY),bottom=Math.max(drag.startY,pointerY);
  const next=new Set(drag.initial);let lastHit=null;
  for(const card of drag.cards) {
    const rect=card.getBoundingClientRect();
    const x=rect.left-gridRect.left+drag.grid.scrollLeft,y=rect.top-gridRect.top+drag.grid.scrollTop;
    if(x<right&&x+rect.width>left&&y<bottom&&y+rect.height>top) {
      const id=card.dataset.imageId;lastHit=id;
      if(drag.invert&&drag.initial.has(id))next.delete(id);else next.add(id);
    }
  }
  if(next.size!==state.selected.size||[...next].some(id=>!state.selected.has(id))) {
    state.selected=next;librarySyncSelection();
  }
  drag.lastHit=lastHit;
  let visible={left:Math.max(0,gridRect.left),top:Math.max(0,gridRect.top),
    right:Math.min(window.innerWidth,gridRect.right),bottom:Math.min(window.innerHeight,gridRect.bottom)};
  for(const parent of drag.scrollParents) {
    const rect=selectionVisibleRect(parent);
    visible={left:Math.max(visible.left,rect.left),top:Math.max(visible.top,rect.top),
      right:Math.min(visible.right,rect.right),bottom:Math.min(visible.bottom,rect.bottom)};
  }
  const x=Math.max(visible.left,left+gridRect.left-drag.grid.scrollLeft);
  const y=Math.max(visible.top,top+gridRect.top-drag.grid.scrollTop);
  const r=Math.min(visible.right,right+gridRect.left-drag.grid.scrollLeft);
  const b=Math.min(visible.bottom,bottom+gridRect.top-drag.grid.scrollTop);
  Object.assign(drag.marquee.style,{left:`${x}px`,top:`${y}px`,width:`${Math.max(0,r-x)}px`,height:`${Math.max(0,b-y)}px`});
}

function selectionFrame(time) {
  const drag=selectionUI.drag;if(!drag?.active)return;
  if(!selectionContextMatches(drag)||libraryBlocked()||$('#modal').open){selectionCancel();return;}
  const elapsed=Math.min(2,Math.max(.25,(time-(drag.lastFrame||time-16.7))/16.7));
  drag.lastFrame=time;
  selectionAutoScroll(drag,elapsed);selectionRenderDrag(drag);
  selectionUI.frame=requestAnimationFrame(selectionFrame);
}

function selectionStart(drag) {
  drag.active=true;closeLibraryMenu(false);
  drag.marquee=document.createElement('div');drag.marquee.className='selection-marquee';
  drag.marquee.setAttribute('aria-hidden','true');document.body.append(drag.marquee);
  drag.grid.classList.add('drag-selecting');
  document.getSelection()?.removeAllRanges();
  try{drag.grid.setPointerCapture(drag.pointerId);}catch{/* A released pointer is cancelled by pointerup. */}
  selectionRenderDrag(drag);selectionUI.frame=requestAnimationFrame(selectionFrame);
}

document.addEventListener('pointerdown',event=>{
  // A fresh press must not inherit suppression intended for the previous drag's click.
  if(selectionUI.drag)selectionCancel();
  selectionUI.suppressUntil=0;selectionUI.releasePointerId=null;
  if(event.button!==0||!event.isPrimary||event.pointerType==='touch'||state.view!=='dataset'||
    libraryBlocked()||$('#modal').open)return;
  const grid=event.target.closest('.image-grid');
  if(!grid||grid.classList.contains('results-grid')||!grid.querySelector('[data-image-id]'))return;
  const control=event.target.closest('button,input,select,textarea,a,[contenteditable],[role="checkbox"]');
  if(control&&!control.matches('.thumb-button[data-action="select"]'))return;
  const rect=grid.getBoundingClientRect();
  selectionUI.drag={grid,cards:[...grid.querySelectorAll('.image-card[data-image-id]')],
    pointerId:event.pointerId,active:false,project:state.project.root,request:state.dataRequest,
    filter:state.dataset.filter,query:state.dataset.query,offset:state.dataset.offset,
    startClientX:event.clientX,startClientY:event.clientY,clientX:event.clientX,clientY:event.clientY,
    startX:event.clientX-rect.left+grid.scrollLeft,startY:event.clientY-rect.top+grid.scrollTop,
    initial:new Set(state.selected),focusId:libraryUI.focusId,anchorId:libraryUI.anchorId,
    invert:event.shiftKey||event.ctrlKey||event.metaKey,scrollParents:selectionScrollParents(grid)};
  // This blocks native image dragging and text selection without changing clicks.
  event.preventDefault();
},true);

document.addEventListener('pointermove',event=>{
  const drag=selectionUI.drag;if(!drag||drag.pointerId!==event.pointerId)return;
  if(!selectionContextMatches(drag)||libraryBlocked()){selectionCancel();return;}
  if(!(event.buttons&1)){selectionCancel();return;}
  drag.clientX=event.clientX;drag.clientY=event.clientY;
  if(!drag.active&&Math.hypot(event.clientX-drag.startClientX,event.clientY-drag.startClientY)>=selectionThreshold)selectionStart(drag);
  if(drag.active){event.preventDefault();selectionRenderDrag(drag);}
},true);

document.addEventListener('pointerup',event=>{
  const drag=selectionUI.drag;
  if(!drag||drag.pointerId!==event.pointerId) {
    if(selectionUI.releasePointerId===event.pointerId){selectionArmSuppression(null);}
    return;
  }
  if(drag.active&&selectionContextMatches(drag)&&!libraryBlocked()) {
    drag.clientX=event.clientX;drag.clientY=event.clientY;selectionRenderDrag(drag);
    const id=drag.lastHit||drag.focusId;
    selectionCancel({restore:false});selectionUI.releasePointerId=null;
    if(id){libraryUI.anchorId=id;libraryFocus(id);}
    event.preventDefault();
  } else selectionCancel({restore:drag.active});
},true);

document.addEventListener('pointercancel',event=>{if(selectionUI.drag?.pointerId===event.pointerId)selectionCancel();},true);
document.addEventListener('lostpointercapture',event=>{if(selectionUI.drag?.active&&selectionUI.drag.pointerId===event.pointerId)selectionCancel();},true);
document.addEventListener('dragstart',event=>{if(event.target.closest('.image-grid:not(.results-grid)'))event.preventDefault();});
window.addEventListener('blur',()=>selectionCancel());

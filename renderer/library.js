'use strict';

// Dataset interaction state is scoped to the current project and visible page.
const libraryUI = { focusId:null, anchorId:null, undo:[], busy:false, switching:false, menuOrigin:null, modalOrigin:null };
const macKeyboard = /Mac/.test(navigator.platform);
const commandKey = macKeyboard ? '⌘' : 'Ctrl';
function editingText(target) { return !!target?.closest('input,textarea,select,[contenteditable=""],[contenteditable="true"],[role="textbox"]'); }
function libraryRemoved() { return state.dataset.filter === 'REMOVED'; }
function libraryBlocked() { return !!(active() || libraryUI.busy || libraryUI.switching || state.dataset.loading); }
function libraryButton(id) { return $$('.thumb-button[data-action="select"]').find(button=>button.dataset.id===String(id)); }
function captureFocus(root=document) {
  const current=document.activeElement;
  if(!current || !root.contains(current) || (!current.id&&!Object.keys(current.dataset).length))return null;
  return {id:current.id,data:{...current.dataset},start:current.selectionStart,end:current.selectionEnd};
}
function restoreFocus(saved, root=document) {
  if(!saved)return;
  const current=saved.id?document.getElementById(saved.id):$$('[data-action],[data-filter],[data-scope]',root).find(el=>Object.entries(saved.data).every(([key,value])=>el.dataset[key]===value));
  if(!current || !root.contains(current) || current.disabled)return;
  current.focus({preventScroll:true});
  if(saved.start!=null && ['text','search','url','tel','password'].includes(current.type))current.setSelectionRange(saved.start,saved.end);
}
function libraryReset() {
  if(typeof selectionCancel==='function')selectionCancel({restore:false});
  state.selected.clear(); libraryUI.focusId=null; libraryUI.anchorId=null; closeLibraryMenu(false);
}
function invalidateSearch() {
  state.searchRequest++; state.results=null; state.searchSnapshot=null; state.searchLoading=false; state.search.offset=0;
}
function librarySyncSelection() {
  const visible=state.dataset.items.map(item=>String(item.id));
  if(!visible.includes(libraryUI.focusId))libraryUI.focusId=visible[0]||null;
  $$('.image-card[data-image-id]').forEach(card=>{
    const id=card.dataset.imageId, selected=state.selected.has(id), button=$('.thumb-button',card);
    card.classList.toggle('selected',selected);
    button.setAttribute('aria-pressed',String(selected)); button.tabIndex=id===libraryUI.focusId?0:-1;
    const checkbox=$('.selection-box',card);checkbox.setAttribute('aria-checked',String(selected));
  });
  const controlsBlocked=libraryUI.busy||libraryUI.switching||state.dataset.loading;
  $$('#labeling-only,[data-layout],[data-action="reset-layout"]').forEach(control=>{control.disabled=controlsBlocked;});
  const toolbar=$('#selection-tools');
  if(toolbar){const markup=selectionToolbarContent();if(toolbar.innerHTML!==markup){const focused=captureFocus(toolbar);toolbar.innerHTML=markup;restoreFocus(focused,toolbar);}}
}
function libraryFocus(id, scroll=false) {
  libraryUI.focusId=String(id); librarySyncSelection();
  const button=libraryButton(id);button?.focus({preventScroll:true});
  if(scroll)button?.scrollIntoView({block:'nearest',inline:'nearest'});
}
function librarySelect(id, {shiftKey=false,ctrlKey=false,metaKey=false,toggle=false,replace=false}={}) {
  id=String(id);const ids=state.dataset.items.map(item=>String(item.id));
  if(!ids.includes(id) || state.dataset.loading || libraryUI.busy || libraryUI.switching)return;
  if(shiftKey && ids.includes(libraryUI.anchorId)) {
    const a=ids.indexOf(libraryUI.anchorId),b=ids.indexOf(id);
    if(!ctrlKey&&!metaKey)state.selected.clear();
    ids.slice(Math.min(a,b),Math.max(a,b)+1).forEach(item=>state.selected.add(item));
  } else if(!replace||ctrlKey||metaKey||toggle) {
    state.selected.has(id)?state.selected.delete(id):state.selected.add(id);libraryUI.anchorId=id;
  } else {state.selected=new Set([id]);libraryUI.anchorId=id;}
  libraryFocus(id);
}
function librarySelectAll() {
  if(state.dataset.loading)return;
  state.dataset.items.forEach(item=>state.selected.add(String(item.id)));librarySyncSelection();
}
function selectionToolbarContent() {
  const count=state.selected.size, disabled=libraryBlocked()?'disabled':'';
  const selectAll=`<button class="button selection-end" data-action="select-all" ${state.dataset.loading?'disabled':''}>페이지 전체 선택</button>`;
  if(!count)return `<span class="selection-hint">${libraryRemoved()?'제외한 이미지를 선택해 복원할 수 있습니다.':'클릭으로 추가·해제 · 드래그로 추가 선택 · Shift+드래그로 선택 반전'}</span>${selectAll}`;
  const actions=libraryRemoved()?`<button class="button primary" data-action="restore-images" ${disabled}>${icon('refresh')}데이터셋으로 복원</button>`:
    `<button class="button" data-action="label" data-label="SEM" ${disabled}>1 · SEM</button><button class="button" data-action="label" data-label="TEM" ${disabled}>2 · TEM</button><button class="button" data-action="label" data-label="OTHER" ${disabled}>3 · 기타</button><button class="button" data-action="label" data-label="" ${disabled}>0 · 라벨 해제</button><button class="button" data-action="metadata" ${disabled}>${icon('settings')}정보 · 그룹 편집</button><button class="button danger-ghost" data-action="remove-images" title="데이터셋에서만 제거합니다. 원본 파일은 보존됩니다." ${disabled}>${icon('trash')}제거 <kbd>Del</kbd></button>`;
  return `<strong>${number(count)}장 선택</strong>${actions}${selectAll}<button class="icon-button" data-action="clear-selection" aria-label="선택 해제 · Esc">${icon('close')}</button>`;
}
async function removeSelectedImages() {
  if(libraryRemoved()||!state.selected.size)return;
  if(libraryBlocked())throw new Error('진행 중인 작업이 끝난 뒤 이미지를 제거해주세요.');
  const ids=[...state.selected].map(Number), project=state.project.root;
  const position=Math.max(0,state.dataset.items.findIndex(item=>String(item.id)===libraryUI.focusId));
  libraryUI.busy=true;librarySyncSelection();closeLibraryMenu(false);
  try {
    const result=await request('remove_images',{ids});
    if(project!==state.project.root)return;
    const entry={ids:result.ids||[],project};
    if(entry.ids.length) {libraryUI.undo.push(entry);libraryUI.undo=libraryUI.undo.slice(-20);}
    invalidateSearch();state.selected.clear();await refreshStatus();await loadImages();
    const next=state.dataset.items[Math.min(position,state.dataset.items.length-1)];
    if(next)libraryFocus(next.id);
    toast(`${number(result.removed)}장을 데이터셋에서 제거했습니다. 원본 파일은 그대로입니다.`,false,{label:`실행 취소 · ${commandKey}+Z`,run:()=>undoImageRemoval(entry)});
  } finally {libraryUI.busy=false;librarySyncSelection();}
}
async function restoreImages(ids, {undo=false}={}) {
  if(libraryBlocked())throw new Error('진행 중인 작업이 끝난 뒤 복원해주세요.');
  libraryUI.busy=true;librarySyncSelection();closeLibraryMenu(false);
  try {
    const result=await request('restore_images',{ids:ids.map(Number)});
    $$('.toast:has(.toast-action)').forEach(item=>item.remove());
    const restored=new Set(result.ids||[]);
    libraryUI.undo.forEach(entry=>{entry.ids=entry.ids.filter(id=>!restored.has(id));});
    libraryUI.undo=libraryUI.undo.filter(entry=>entry.ids.length);
    invalidateSearch();state.selected.clear();await refreshStatus();await loadImages();
    toast(`${number(result.restored)}장을 복원했습니다.${undo?' 라벨과 시료 정보도 유지됩니다.':''}`);
    return result;
  } finally {libraryUI.busy=false;librarySyncSelection();}
}
async function undoImageRemoval(entry=libraryUI.undo.at(-1)) {
  if(!entry||entry.project!==state.project.root||!libraryUI.undo.includes(entry))return;
  await restoreImages(entry.ids,{undo:true});
}
function closeLibraryMenu(restore=true) {
  const menu=$('#context-menu');if(!menu || menu.hidden)return;
  menu.hidden=true;
  if(restore&&libraryUI.menuOrigin)libraryFocus(libraryUI.menuOrigin);
}
function showLibraryMenu(id, x, y) {
  if(libraryUI.switching||state.view!=='dataset'||state.dataset.loading||$('#modal').open)return;
  id=String(id);if(!state.selected.has(id))librarySelect(id,{replace:true});
  libraryUI.menuOrigin=id;libraryUI.focusId=id;
  const disabled=libraryBlocked()?'disabled':'', menu=$('#context-menu');
  const item=(label,action,shortcut='',attributes='')=>`<button role="menuitem" class="context-menu-item ${action==='remove-images'?'danger':''}" data-action="${action}" ${attributes}><span>${label}</span>${shortcut?`<kbd>${shortcut}</kbd>`:''}</button>`;
  menu.innerHTML=`<div class="context-menu-heading">${number(state.selected.size)}장 선택${libraryRemoved()?' · 제외한 이미지':''}</div>`+
    (libraryRemoved()?item('데이터셋으로 복원','restore-images','',disabled):
      item(state.selected.size===1?'이미지 정보 · 미리보기':'선택 이미지 정보 편집','metadata','Enter',disabled)+
      (state.selected.size===1?item('원본 폴더에서 보기','reveal-item','',`data-id="${e(id)}"`):'')+
      '<div role="separator" class="context-menu-separator"></div>'+
      ['SEM','TEM','OTHER'].map((label,i)=>item(`${label==='OTHER'?'기타':label} 라벨 지정`,'label',String(i+1),`data-label="${label}" ${disabled}`)).join('')+
      item('라벨 해제','label','0',`data-label="" ${disabled}`)+
      '<div role="separator" class="context-menu-separator"></div>'+item('데이터셋에서 제거','remove-images','Del',disabled));
  menu.hidden=false;menu.style.left='0px';menu.style.top='0px';
  const rect=menu.getBoundingClientRect();
  menu.style.left=`${Math.max(8,Math.min(x,window.innerWidth-rect.width-8))}px`;
  menu.style.top=`${Math.max(8,Math.min(y,window.innerHeight-rect.height-8))}px`;
  $('button:not(:disabled)',menu)?.focus({preventScroll:true});
}
function libraryShortcuts() {
  const rows=[['클릭 / 체크박스','Ctrl 없이 선택 추가·해제'],['드래그 / Shift+드래그','사각형 안의 이미지 추가 선택 / 시작 시점의 선택 반전'],['Shift+클릭','연속 범위 선택'],['방향키 / Shift+방향키','한 장씩 이동·선택 / 범위 선택'],['Space / Enter / 더블클릭','선택 추가·해제 / 이미지 정보·미리보기'],[`${commandKey}+A`,'현재 페이지 전체 선택'],['1 / 2 / 3 / 0','SEM / TEM / 기타 / 라벨 해제'],[macKeyboard?'Delete(⌫) / Fn+Delete':'Delete','선택 이미지를 데이터셋에서 제거'],[`${commandKey}+Z`,'최근 이미지 제거 실행 취소'],['우클릭 / Shift+F10','선택 이미지 메뉴'],[`${commandKey}+F`,'현재 화면의 검색창으로 이동'],['Esc','드래그 취소 / 메뉴·대화상자 닫기 / 선택 해제'],['F1 / ?','단축키 안내']];
  showModal('마우스 · 키보드 사용 안내','입력 중에는 글자 편집 단축키를 그대로 사용합니다.',`<table class="shortcut-table"><tbody>${rows.map(([keys,description])=>`<tr><th><kbd>${e(keys)}</kbd></th><td>${e(description)}</td></tr>`).join('')}</tbody></table><p class="field-help">제거는 현재 프로젝트에서만 적용됩니다. 원본 파일과 기존 학습 모델은 변경되지 않으며, ‘제외한 이미지’에서 앱을 다시 연 뒤에도 복원할 수 있습니다.</p>`,'<button class="button primary" data-action="close-modal">확인</button>');
}
async function libraryKeydown(event) {
  if(typeof selectionIsDragging==='function'&&selectionIsDragging()) {
    if(event.key==='Escape'){event.preventDefault();selectionCancel();}
    return;
  }
  if(libraryUI.switching||event.isComposing||event.keyCode===229)return;
  const key=event.key, modifier=event.ctrlKey||event.metaKey, menu=$('#context-menu');
  if(!menu.hidden) {
    const buttons=$$('button:not(:disabled)',menu), current=buttons.indexOf(document.activeElement);
    if(['ArrowDown','ArrowUp','Home','End'].includes(key)) {
      event.preventDefault();const index=key==='Home'?0:key==='End'?buttons.length-1:(current+(key==='ArrowDown'?1:-1)+buttons.length)%buttons.length;
      buttons[index]?.focus();return;
    }
    if(key==='Escape'||key==='Tab'){closeLibraryMenu();if(key==='Escape')event.preventDefault();return;}
    if(!modifier&&!event.altKey&&!libraryRemoved()) {
      if(key==='Delete'||(macKeyboard&&key==='Backspace')){event.preventDefault();closeLibraryMenu();await removeSelectedImages();return;}
      if(['0','1','2','3'].includes(key)){event.preventDefault();closeLibraryMenu();await applyLabel({'0':null,'1':'SEM','2':'TEM','3':'OTHER'}[key]);return;}
    }
    if(key==='Enter'||key===' '||key==='Spacebar')return;
    return;
  }
  if($('#modal').open) {
    if(key==='Enter'&&!modifier&&!event.altKey&&event.target.matches('input')) {
      const action=$('#metadata-form')?'save-metadata':$('#import-form')?'confirm-import':null;
      if(action){event.preventDefault();const button=$(`[data-action="${action}"]`);if(button&&!button.disabled)button.click();}
    }
    return;
  }
  if(key==='F1'||(key==='?'&&!editingText(event.target))){event.preventDefault();libraryShortcuts();return;}
  if(modifier&&key.toLowerCase()==='f') {
    const search=state.view==='dataset'?$('#dataset-query'):state.view==='search'?$('#semantic-query'):null;
    if(search){event.preventDefault();search.focus();search.select();}return;
  }
  if(key==='Enter'&&event.target.id==='semantic-query') {event.preventDefault();await doSearch(true);return;}
  if(editingText(event.target)||state.view!=='dataset'||event.altKey)return;
  if(modifier&&key.toLowerCase()==='a'){event.preventDefault();librarySelectAll();return;}
  if(modifier&&key.toLowerCase()==='z'&&!event.shiftKey){event.preventDefault();await undoImageRemoval();return;}
  if(key==='ContextMenu'||(key==='F10'&&event.shiftKey)) {
    const button=libraryButton(libraryUI.focusId||state.dataset.items[0]?.id);
    if(button){event.preventDefault();const rect=button.getBoundingClientRect();showLibraryMenu(button.dataset.id,rect.left+20,rect.top+20);}return;
  }
  if(modifier)return;
  if(key==='Escape'){event.preventDefault();state.selected.clear();librarySyncSelection();return;}
  if(key==='Delete'||(macKeyboard&&key==='Backspace')) {if(state.selected.size&&!libraryRemoved()){event.preventDefault();await removeSelectedImages();}return;}
  if(state.selected.size&&!libraryRemoved()&&['0','1','2','3'].includes(key)){event.preventDefault();await applyLabel({'0':null,'1':'SEM','2':'TEM','3':'OTHER'}[key]);return;}
  const currentButton=event.target.closest('.thumb-button[data-action="select"]');
  if(!currentButton)return;
  if(key==='Enter'){event.preventDefault();if(!libraryRemoved())metadataModal([currentButton.dataset.id]);return;}
  if(key===' '){event.preventDefault();librarySelect(currentButton.dataset.id,{toggle:true,shiftKey:event.shiftKey});return;}
  if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(key)) {
    event.preventDefault();const buttons=$$('.thumb-button[data-action="select"]'), index=buttons.indexOf(currentButton);
    const firstTop=buttons[0]?.getBoundingClientRect().top;
    const columns=Math.max(1,buttons.filter(button=>Math.abs(button.getBoundingClientRect().top-firstTop)<3).length);
    const delta={ArrowLeft:-1,ArrowRight:1,ArrowUp:-columns,ArrowDown:columns};
    const next=key==='Home'?0:key==='End'?buttons.length-1:Math.max(0,Math.min(buttons.length-1,index+delta[key]));
    librarySelect(buttons[next].dataset.id,{shiftKey:event.shiftKey,replace:true});libraryFocus(buttons[next].dataset.id,true);
  }
}

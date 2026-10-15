'use strict';

const libraryPreferenceKey='gemmascope.library.preferences.v1';
function readLibraryPreferences() {
  const defaults={columns:'auto',rows:4,imageSize:176,labelingOnly:false};
  try {
    const saved=JSON.parse(localStorage.getItem(libraryPreferenceKey)||'{}');
    if(saved.columns==='auto'||Number.isInteger(saved.columns)&&saved.columns>=1&&saved.columns<=8)defaults.columns=saved.columns;
    if(Number.isInteger(saved.rows)&&saved.rows>=1&&saved.rows<=12)defaults.rows=saved.rows;
    if(Number.isInteger(saved.imageSize)&&saved.imageSize>=96&&saved.imageSize<=280)defaults.imageSize=saved.imageSize;
    if(typeof saved.labelingOnly==='boolean')defaults.labelingOnly=saved.labelingOnly;
  } catch {}
  return defaults;
}
const initialLibraryPreferences=readLibraryPreferences();
function saveLibraryPreferences() {
  try{localStorage.setItem(libraryPreferenceKey,JSON.stringify({...state.dataset.layout,labelingOnly:state.dataset.labelingOnly}));}
  catch{toast('보기 설정을 저장하지 못했습니다. 현재 창에서는 계속 적용됩니다.',true);}
}
function libraryControls() {
  const d=state.dataset,l=d.layout;
  return `<div class="library-controls"><label class="labeling-mode"><input id="labeling-only" type="checkbox" ${d.labelingOnly?'checked':''}>미분류만 계속 라벨링</label>${d.labelingOnly?`<span class="labeling-status">미분류 ${number(state.project.counts?.UNLABELED)}개 남음 · 라벨 지정 후 자동 채움</span>`:''}<details id="layout-controls" class="layout-controls" ${d.layoutOpen?'open':''}><summary>보기 설정 <span id="layout-summary"></span></summary><div class="layout-options"><div class="layout-field"><label for="layout-columns">열 수</label><select id="layout-columns" data-layout="columns">${['auto',1,2,3,4,5,6,7,8].map(value=>`<option value="${value}" ${l.columns===value?'selected':''}>${value==='auto'?'창 너비에 맞춤':`${value}열`}</option>`).join('')}</select></div><div class="layout-field"><label for="layout-rows">페이지당 행 수</label><input id="layout-rows" data-layout="rows" type="number" min="1" max="12" step="1" value="${l.rows}" required><small>1–12행 · 마지막 페이지는 남은 개수만 표시</small></div><div class="layout-field"><label for="layout-image-size">이미지 크기 <output id="layout-image-size-value" for="layout-image-size">${l.imageSize}px</output></label><input id="layout-image-size" data-layout="imageSize" type="range" min="96" max="280" step="8" value="${l.imageSize}"><small>자동 열 모드에서는 크기에 따라 열 수도 바뀝니다.</small></div><p class="layout-hint"><span id="layout-capacity"></span> · 설정은 이 앱에 저장됩니다. 이미지 원본 크기는 바뀌지 않습니다.</p><button class="button small" data-action="reset-layout">기본 보기로</button></div></details></div>`;
}
function applyLibraryLayout() {
  if(state.view!=='dataset')return;
  const d=state.dataset,l=d.layout,grid=$('.dataset-grid');
  if(grid){
    grid.classList.toggle('auto-columns',l.columns==='auto');
    grid.style.setProperty('--library-image-size',`${l.imageSize}px`);
    grid.style.setProperty('--library-columns',String(l.columns==='auto'?4:l.columns));
  }
  const columns=libraryColumnCount();
  const summary=$('#layout-summary'),capacity=$('#layout-capacity'),size=$('#layout-image-size-value');
  if(summary)summary.textContent=`${l.columns==='auto'?'자동 · ':''}${columns}열 × ${l.rows}행`;
  if(capacity)capacity.textContent=`페이지당 최대 ${number(libraryPageCapacity(columns))}개`;
  if(size)size.textContent=`${l.imageSize}px`;
}
function libraryColumnCount() {
  const d=state.dataset;
  if(d.layout.columns!=='auto')return Number(d.layout.columns);
  const panel=state.view==='dataset'?$('.dataset-panel'):null;
  if(!panel)return d.visibleColumns||4;
  // Dataset padding and gap are fixed independently of the search result grid.
  return Math.max(1,Math.floor((panel.clientWidth-44+16)/(d.layout.imageSize+16)));
}
function libraryPageCapacity(columns=libraryColumnCount()) {
  // Keep offsets aligned with the API's 200-item maximum, even on wide displays.
  return Math.min(200,columns*state.dataset.layout.rows);
}
function syncLibraryPageSize() {
  const d=state.dataset,columns=libraryColumnCount(),limit=libraryPageCapacity(columns);
  d.visibleColumns=columns;
  if(d.limit===limit)return false;
  selectionCancel({restore:false});
  d.offset=d.labelingOnly?0:Math.floor(d.offset/limit)*limit;
  d.limit=limit;
  return true;
}
async function setLabelingOnly(enabled) {
  if(libraryUI.busy||libraryUI.switching)return;
  const focused=captureFocus();
  selectionCancel({restore:false});
  state.dataset.labelingOnly=enabled;
  state.dataset.filter=enabled?'UNLABELED':'ALL';state.dataset.offset=0;
  libraryReset();saveLibraryPreferences();await loadImages();
  if(state.view==='dataset'&&document.activeElement===document.body)restoreFocus(focused);
}
async function commitLibraryLayout(input) {
  if(libraryUI.busy||libraryUI.switching)return;
  if(!input.reportValidity())return;
  const key=input.dataset.layout,value=input.value==='auto'?'auto':Number(input.value);
  if(key==='columns'&&!(value==='auto'||Number.isInteger(value)&&value>=1&&value<=8))return;
  if(key==='rows'&&!(Number.isInteger(value)&&value>=1&&value<=12))return;
  if(key==='imageSize'&&!(value>=96&&value<=280))return;
  state.dataset.layout[key]=value;
  selectionCancel({restore:false});saveLibraryPreferences();applyLibraryLayout();
  if(syncLibraryPageSize()){
    const focused=captureFocus();
    await loadImages();
    if(state.view==='dataset'&&document.activeElement===document.body)restoreFocus(focused);
  }
}
function datasetEmpty() {
  const d=state.dataset;
  if(d.labelingOnly) {
    const completed=!state.project.counts?.UNLABELED;
    return empty(completed?'모든 이미지의 라벨링을 마쳤습니다':'조건에 맞는 미분류 이미지가 없습니다',completed?'라벨을 붙인 이미지는 전체 또는 라벨별 탭에서 다시 확인할 수 있습니다.':'다른 검색어를 입력하거나 검색어를 지워보세요.',`<button class="button" data-action="end-labeling">전체 데이터셋 보기</button>`);
  }
  return empty(libraryRemoved()?'제외한 이미지가 없습니다':state.project.total?'조건에 맞는 이미지가 없습니다':'첫 번째 이미지 폴더를 추가해보세요',libraryRemoved()?'제거한 이미지는 이곳에서 복원할 수 있습니다. 원본 파일과 기존 모델은 그대로 유지됩니다.':state.project.total?'다른 라벨을 선택하거나 검색어를 바꿔보세요.':'컴퓨터에 있는 이미지를 불러오세요.<br>원본은 그대로 두고, 라벨과 학습 데이터만 이 프로젝트에 저장합니다.',libraryRemoved()||state.project.total?'<button class="button" data-action="reset-filter">필터 초기화</button>':`<button class="button primary" data-action="import" ${active()?'disabled':''}>${icon('folder')}폴더에서 가져오기</button>`,'PNG · JPG · TIFF · DM3/DM4 · SER/EMI · MRC');
}

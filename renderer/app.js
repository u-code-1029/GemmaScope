'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const e = escapeHTML;
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const number = value => Number(value || 0).toLocaleString('ko-KR');
const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const date = value => value ? new Date(value).toLocaleString('ko-KR', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
const shortPath = value => String(value || '').split(/[\\/]/).filter(Boolean).pop() || '로컬 프로젝트';
const names = {dataset:'데이터셋',training:'학습 스튜디오',search:'검색 · 검증',runs:'실행 기록'};
const terminal = job => !job || ['completed','complete','done','failed','cancelled','canceled','error'].includes(job.status);
const success = job => job && ['completed','complete','done'].includes(job.status);
const state = {
  view:'dataset', project:{total:0,labeled:0,counts:{SEM:0,TEM:0,OTHER:0,UNLABELED:0},roots:[]}, environment:{}, runs:[], indexes:[],
  dataset:{filter:'ALL',query:'',offset:0,limit:40,items:[],total:0,loading:false}, selected:new Set(),
  training:{mode:'classifier',epochs:3,batch_size:1,gradient_accumulation:8,learning_rate:0.00002,threshold:0.95},
  settings:{device:'auto',offline:false}, search:{query:'',run_id:'',threshold:0.95,label:'ALL',sample:'',date_from:'',date_to:'',offset:0,limit:40},
  results:null, searchLoading:false, job:null, jobKind:null, showLogs:false, dismissedJob:null, handledJobs:new Set(), indexDirty:false,
  connected:false, statusBusy:false, dataRequest:0, searchRequest:0, importPath:'', modalIds:[], searchSnapshot:null
};
const bridge = window.desktop;

function toast(message, isError = false) {
  const element = document.createElement('div');
  element.className = `toast${isError ? ' error' : ''}`;
  const text = document.createElement('span'); text.textContent = String(message);
  const close = document.createElement('button'); close.textContent = '×'; close.setAttribute('aria-label','알림 닫기'); close.onclick = () => element.remove();
  element.append(text, close); $('#toasts').append(element);
  while($('#toasts').children.length>3)$('#toasts').firstElementChild.remove();
  setTimeout(() => element.remove(), isError ? 10000 : 4200);
}
function errorMessage(error) { return String(error?.message || error || '작업을 완료하지 못했습니다.'); }
async function request(action, payload = {}) {
  if (!bridge) throw new Error('이 기능은 GemmaScope 데스크톱 앱에서 실행할 수 있습니다.');
  return bridge.request(action, payload);
}
function active() { return state.job && !terminal(state.job); }
function chosenRun() { return state.runs.find(run => String(run.id) === String(state.search.run_id)); }
function validRuns() { return state.runs.filter(run => !['failed','cancelled','canceled','running','queued'].includes(run.status)); }
function labelBadge(label) { return `<span class="label-badge ${String(label || '').toLowerCase()}">${label === 'OTHER' ? 'OTHER' : e(label || '미분류')}</span>`; }
function heading(eyebrow, title, description, actions = '') { return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${description}</p></div>${actions ? `<div class="heading-actions">${actions}</div>` : ''}</div>`; }
function empty(title, description, action = '', caption = '') { return `<div class="empty-state"><div class="empty-art" aria-hidden="true"><span></span><span>${icon('image')}</span></div><h2>${title}</h2><p>${description}</p>${action}${caption ? `<small>${caption}</small>` : ''}</div>`; }
function field(scope,key,label,type='text',attributes='') { const value=state[scope][key]; return `<div class="field"><label class="field-label" for="${scope}-${key}">${label}</label><input id="${scope}-${key}" type="${type}" data-scope="${scope}" data-key="${key}" value="${e(value)}" ${attributes}></div>`; }
function deviceField() { return `<div class="field"><label class="field-label" for="settings-device">연산 장치</label><select id="settings-device" data-scope="settings" data-key="device">${[['auto','자동 선택'],['mps','Apple Silicon · MPS'],['cuda','NVIDIA · CUDA'],['cpu','CPU']].map(([value,label])=>`<option value="${value}" ${state.settings.device===value?'selected':''}>${label}</option>`).join('')}</select></div>`; }
function offlineField() { return `<label class="inline-check"><input type="checkbox" data-scope="settings" data-key="offline" ${state.settings.offline?'checked':''}> 오프라인 실행 · 다운로드된 모델만 사용</label><p class="field-help">첫 실행에는 모델 다운로드를 위한 인터넷 연결이 필요합니다. 이미지 파일은 외부로 전송하지 않습니다.</p>`; }

function updateChrome() {
  $('#project-name').textContent = shortPath(state.project.root);
  $('#project-location').textContent = state.project.root || (bridge ? '기본 로컬 작업 공간' : '데스크톱 앱에서 실행');
  $('#project-location').title = state.project.root || '';
  $('#nav-count').textContent = number(state.project.total);
  const env = state.environment;
  $('#side-environment').textContent = env.ml_ready ? '학습 환경 준비 완료' : bridge ? '실행 환경 설정' : '데스크톱 앱에서 실행';
  $('#device-pill').innerHTML = `<span></span>${e(env.device || (state.connected ? '로컬 환경' : '환경 확인 중'))}`;
  $$('.nav-item').forEach(item => { item.classList.toggle('active',item.dataset.view===state.view); if(item.dataset.view===state.view) item.setAttribute('aria-current','page');else item.removeAttribute('aria-current'); });
  $('#breadcrumb-current').textContent=names[state.view];
}
function render() {
  updateChrome();
  $('#content').innerHTML = ({dataset:renderDataset,training:renderTraining,search:renderSearch,runs:renderRuns}[state.view])();
  renderJob();
}
function statsCard(label,value,note,color,whichIcon='image') { return `<div class="stat-card"><div class="stat-dot ${color}">${icon(whichIcon)}</div><div class="stat-content"><div class="stat-top">${label}</div><div class="stat-number">${number(value)}<small>장</small></div><div class="stat-note">${note}</div></div></div>`; }
function renderDataset() {
  const d=state.dataset, counts=state.project.counts || {};
  const filters=[['ALL','전체',state.project.total],['UNLABELED','미분류',counts.UNLABELED],['SEM','SEM',counts.SEM],['TEM','TEM',counts.TEM],['OTHER','기타',counts.OTHER]];
  return heading('YOUR IMAGE LIBRARY','데이터셋','로컬 이미지를 모으고, 나만의 전자현미경 분류 기준을 만드세요.',`<button class="button ghost" data-action="metadata-csv">${icon('download')}메타데이터 CSV</button><button class="button primary" data-action="import" ${active()?'disabled':''}>${icon('plus')}이미지 폴더 추가</button>`)
    + `<div class="intro-panel"><div class="intro-icon">${icon('spark')}</div><div class="intro-copy"><strong>좋은 분류는, 잘 정리된 데이터에서 시작됩니다.</strong><p>SEM · TEM · 기타로 라벨을 지정하세요. 같은 시료의 이미지는 하나의 그룹으로 묶어주세요.</p></div><div class="intro-steps"><div class="intro-step active"><span>1</span>이미지 준비</div><div class="intro-step"><span>2</span>모델 학습</div><div class="intro-step"><span>3</span>검색 · 검증</div></div></div>`
    + `<div class="stats-grid">${statsCard('전체 이미지',state.project.total,'이 프로젝트에 등록된 이미지','','grid')}${statsCard('SEM',counts.SEM,'주사 전자현미경','purple')}${statsCard('TEM',counts.TEM,'투과 전자현미경','blue')}${statsCard('기타 이미지',counts.OTHER,'검색에서 제외할 이미지','gray')}</div>`
    + `<section class="dataset-panel"><div class="panel-toolbar"><div class="panel-title">이미지 라이브러리 <span class="count">${number(d.total)}</span></div><div class="input-search">${icon('search')}<input aria-label="파일명 또는 경로 검색" id="dataset-query" placeholder="파일명 또는 경로로 검색" value="${e(d.query)}"></div></div><div class="dataset-filters">${filters.map(([value,label,count])=>`<button class="filter-tab ${d.filter===value?'active':''}" data-filter="${value}">${value!=='ALL'?'<span class="dot"></span>':''}${label}<small>${number(count)}</small></button>`).join('')}<span class="filter-end">빠른 라벨 <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd></span></div>${selectionToolbar()}`
    + (d.loading ? '<div class="loading-inline">이미지 라이브러리를 불러오는 중</div>' : d.items.length ? `<div class="image-grid">${d.items.map(imageCard).join('')}</div>` : empty(state.project.total?'조건에 맞는 이미지가 없습니다':'첫 번째 이미지 폴더를 추가해보세요',state.project.total?'다른 라벨을 선택하거나 검색어를 바꿔보세요.':'컴퓨터에 있는 현미경 이미지를 불러오세요.<br>원본은 그대로 두고, 라벨과 학습 데이터만 이 프로젝트에 저장합니다.',state.project.total?'<button class="button" data-action="reset-filter">필터 초기화</button>':`<button class="button primary" data-action="import" ${active()?'disabled':''}>${icon('folder')}폴더에서 가져오기</button>`,'PNG · JPG · TIFF · DM3/DM4 · SER/EMI · MRC'))
    + `<div class="dataset-footer"><span>${icon('shield')}원본 파일을 변경하지 않습니다</span>${pager('dataset',d.offset,d.limit,d.total)}</div></section>`
    + `<p class="dataset-note">${icon('info')}기타 이미지에는 광학현미경, AFM, 그래프 등 헷갈리기 쉬운 예시를 포함하세요. 서로 다른 시료 그룹으로 학습 · 검증 · 평가를 분리합니다.</p>`;
}
function selectionToolbar() {
  if(!state.selected.size) return '';
  return `<div class="selection-toolbar"><strong>${number(state.selected.size)}장 선택</strong><button class="button" data-action="label" data-label="SEM">1 · SEM</button><button class="button" data-action="label" data-label="TEM">2 · TEM</button><button class="button" data-action="label" data-label="OTHER">3 · 기타</button><button class="button" data-action="label" data-label="">라벨 해제</button><button class="button" data-action="metadata">${icon('settings')}정보 · 그룹 편집</button><button class="button selection-end" data-action="select-all">페이지 전체 선택</button><button class="icon-button" data-action="clear-selection" aria-label="선택 해제">${icon('close')}</button></div>`;
}
function imageCard(item) {
  const selected=state.selected.has(String(item.id)), metadata=item.metadata||{};
  return `<article class="image-card ${selected?'selected':''}"><button class="thumb-button" data-action="select" data-id="${e(item.id)}" aria-label="${e(item.name)} 선택" aria-pressed="${selected}">${item.thumbnail?`<img src="${e(item.thumbnail)}" alt="${e(item.name)}" loading="lazy">`:icon('image')}<span class="selection-box">${icon('check')}</span>${item.frames>1?`<span class="frame-badge">프레임 ${Number(item.frame)+1} / ${number(item.frames)}</span>`:''}</button><div class="card-details"><span class="card-title" title="${e(item.path)}">${e(item.name)}</span><div class="card-meta">${labelBadge(item.label)}<button class="card-more" data-action="detail" data-id="${e(item.id)}" title="정보 편집">${e(metadata.sample || (item.width&&item.height ? `${item.width} × ${item.height}` : '상세 정보'))}${icon('info')}</button></div></div></article>`;
}
function pager(scope,offset,limit,total) { return `<div class="pager"><span>${total?`${number(offset+1)}–${number(Math.min(offset+limit,total))} / ${number(total)}`:'0장'}</span><button class="button small" data-action="page" data-scope="${scope}" data-direction="-1" aria-label="이전 페이지" ${offset===0?'disabled':''}>‹</button><button class="button small" data-action="page" data-scope="${scope}" data-direction="1" aria-label="다음 페이지" ${offset+limit>=total?'disabled':''}>›</button></div>`; }
function environmentPanel() {
  const env=state.environment;
  return `<section class="panel"><h2>${icon('chip')}실행 환경</h2><div class="environment-spec"><span>Python</span><strong title="${e(env.python || '')}">${e(env.python_version || (env.python ? shortPath(env.python) : '확인 중'))}</strong></div><div class="environment-spec"><span>연산 장치</span><strong>${e(env.device || '자동 선택')}</strong></div><div class="environment-spec"><span>모델 캐시</span><strong>${env.model_cached?'다운로드됨':'첫 실행 시 다운로드'}</strong></div><div class="environment-spec"><span>학습 라이브러리</span><strong>${env.ml_ready?'<span class="dot">●</span> 준비 완료':'설치 필요'}</strong></div>${!env.ml_ready&&env.compatibility_error?`<div class="notice error">${icon('info')}${e(env.compatibility_error)}</div>`:''}${!env.ml_ready?`<button class="button full" data-action="setup" ${active()?'disabled':''}>${icon('download')}학습 환경 설치</button><p class="field-help">인터넷 연결이 필요합니다. 프로젝트 전용 Python 환경에 필요한 라이브러리를 설치합니다.</p>`:''}</section>`;
}
function renderTraining() {
  const t=state.training,c=state.project.counts||{}, fine=t.mode==='finetune';
  return heading('TRAIN YOUR DETECTOR','학습 스튜디오','우리 연구실의 SEM과 TEM을 구분하는 모델을 학습하세요.')
    + `<div class="form-layout"><div><section class="panel"><h2>${icon('spark')}학습 방법</h2><p class="panel-subtitle">가벼운 분류기부터 시작하고, 필요할 때 이미지 표현 자체를 조정하세요.</p><div class="mode-options"><label class="mode-card ${!fine?'selected':''}"><input type="radio" name="mode" data-scope="training" data-key="mode" value="classifier" ${!fine?'checked':''}><span class="mode-icon">${icon('grid')}</span><strong>임베딩 + 분류기</strong><p>EmbeddingGemma 2는 고정하고,<br>추출한 특징으로 작은 분류기를 학습합니다.</p><span class="tag">권장 시작점 · 적은 메모리</span></label><label class="mode-card ${fine?'selected':''}"><input type="radio" name="mode" data-scope="training" data-key="mode" value="finetune" ${fine?'checked':''}><span class="mode-icon">${icon('spark')}</span><strong>미세조정 · 마지막 2개 블록</strong><p>모델의 마지막 블록과 정규화 · 투영층을<br>학습한 뒤 분류기를 만듭니다.</p><span class="tag">실제 모델 가중치 조정 · 추가 메모리</span></label></div><div class="field-grid">${deviceField()}${field('training','threshold','최소 통과 신뢰도','number','min="0.5" max="1" step="any"')}</div><p class="field-help">${fine?'SEM · TEM 각각 최소 4개, 기타 최소 3개의 시료 그룹이 필요합니다.':'각 클래스에 최소 3개의 시료 그룹이 필요합니다.'} 검증 데이터에서 통과 기준을 조정하고, 별도의 평가 데이터로 성능을 측정합니다.</p>${offlineField()}<div class="form-divider"></div><h2>고급 학습 설정</h2><p class="panel-subtitle">미세조정에 사용하는 설정입니다. 임베딩 + 분류기는 모델 가중치를 변경하지 않습니다.</p><div class="field-grid three">${field('training','epochs','학습 횟수 · Epoch','number',`min="1" max="100" step="1" ${fine?'':'disabled'}`)}${field('training','batch_size','배치 크기','number','min="1" max="64" step="1"')}${field('training','gradient_accumulation','기울기 누적','number',`min="1" max="128" step="1" ${fine?'':'disabled'}`)}${field('training','learning_rate','학습률','number',`min="0.00000001" max="0.01" step="any" ${fine?'':'disabled'}`)}</div></section><div class="train-footer"><span>학습은 이 기기에서 진행됩니다.<br>실행 시간은 이미지 수와 연산 장치에 따라 달라집니다.</span><button class="button primary" data-action="train" ${active()||!state.project.labeled||!state.environment.ml_ready?'disabled':''}>${icon('play')}학습 시작하기</button></div></div><aside class="training-aside"><section class="panel"><h2>학습 데이터</h2><p class="panel-subtitle">라벨이 있는 이미지만 사용합니다.</p>${[['SEM','주사 전자현미경'],['TEM','투과 전자현미경'],['OTHER','기타 이미지']].map(([label,name])=>`<div class="readiness-row"><span>${labelBadge(label)}${name}</span><strong>${number(c[label])}장</strong></div>`).join('')}<div class="notice">${icon('info')}동일 시료의 유사 이미지는 같은 그룹으로 지정하세요. 그룹을 기준으로 데이터를 분리해 평가 누출을 방지합니다.</div><button class="button ghost full" data-view="dataset">데이터셋 확인 ${icon('arrow')}</button></section>${environmentPanel()}<div class="notice warning">${icon('shield')}엄격한 기준은 오탐을 줄이지만, 일부 SEM · TEM을 놓칠 수 있습니다. 자동 판별은 오탐 0을 보장하지 않습니다.</div></aside></div>`;
}
function searchWarnings(result) {
  if(!result || (!result.warning && !result.stale && !result.unindexed))return '';
  const details=[];
  if(result.stale)details.push(`변경된 이미지 ${number(result.stale)}장`);
  if(result.unindexed)details.push(`색인에 없는 이미지 ${number(result.unindexed)}장`);
  return `<div class="notice warning search-warning">${icon('info')}<div>${e(result.warning || '검색 색인을 다시 생성하면 변경된 이미지를 반영할 수 있습니다.')}${details.length?`<br>${details.join(' · ')}`:''}</div></div>`;
}
function renderSearch() {
  const s=state.search,runs=validRuns(),result=state.results;
  return heading('FIND WHAT MATTERS','검색 · 검증','SEM · TEM 판별을 통과한 이미지 안에서만 검색합니다.',`<button class="button" data-action="export-results" ${!result?.total?'disabled':''}>${icon('download')}전체 결과 CSV</button>`)
    + `<section class="panel search-panel"><div class="model-toolbar"><div class="field"><label class="field-label" for="run-select">판별 모델</label><select id="run-select" data-scope="search" data-key="run_id">${!runs.length?'<option value="">완료된 학습이 없습니다</option>':runs.map(run=>`<option value="${e(run.id)}" ${String(s.run_id)===String(run.id)?'selected':''}>${run.mode==='finetune'?'미세조정':'임베딩 + 분류기'} · ${e(date(run.created_at))}</option>`).join('')}</select></div><div class="field threshold-field">${field('search','threshold','최소 통과 신뢰도','number','min="0.5" max="1" step="any"')}</div><button class="button" data-action="index" ${!runs.length||!state.project.total||active()?'disabled':''}>${icon('refresh')}검색 색인 생성</button></div><div class="search-main-row"><div class="input-search">${icon('search')}<input id="semantic-query" data-scope="search" data-key="query" value="${e(s.query)}" placeholder="예: 다공성 표면의 나노입자" aria-label="이미지 내용 검색"></div><button class="button primary" data-action="search" ${!runs.length||state.searchLoading?'disabled':''}>검색 ${icon('arrow')}</button></div><div class="search-filter-grid"><div class="field"><label class="field-label" for="search-label">이미지 종류</label><select id="search-label" data-scope="search" data-key="label"><option value="ALL" ${s.label==='ALL'?'selected':''}>SEM + TEM</option><option value="SEM" ${s.label==='SEM'?'selected':''}>SEM</option><option value="TEM" ${s.label==='TEM'?'selected':''}>TEM</option></select></div>${field('search','sample','시료 이름','text','placeholder="정확한 시료 이름"')}${field('search','date_from','취득일 · 시작','date')}${field('search','date_to','취득일 · 종료','date')}</div><p class="search-hint">${icon('shield')}기타로 지정한 이미지와 신뢰도 기준을 통과하지 못한 이미지는 검색에서 제외됩니다. 검색어를 비우면 통과한 모든 이미지를 확인합니다.</p>${state.indexDirty?`<div class="notice warning">${icon('info')}이미지 또는 판별 모델이 변경되었습니다. 현재 모델로 검색 색인을 생성한 뒤 검색하세요.</div>`:''}<details class="search-runtime"><summary class="field-help">연산 장치 · 다운로드 설정</summary><div class="field-grid">${deviceField()}</div>${offlineField()}</details></section>`
    + searchWarnings(result)
    + (state.searchLoading?'<section class="dataset-panel"><div class="loading-inline">통과한 이미지에서 검색하는 중</div></section>':result?`<div class="search-results-heading"><div><h2>검색 결과 <span class="muted">${number(result.total)}장</span></h2><p>기준 ${percent(state.searchSnapshot?.threshold)} · 색인 ${number(result.indexed)}장 · 기준 미달 또는 제외 ${number(result.excluded)}장${state.searchSnapshot?.query?` · “${e(state.searchSnapshot.query)}”`:''}</p></div><span class="label-badge sem">SEM / TEM ONLY</span></div><section class="dataset-panel">${result.items?.length?`<div class="image-grid results-grid">${result.items.map((item,i)=>`<article class="image-card"><button class="thumb-button" data-action="reveal-result" data-index="${i}" aria-label="${e(item.name)} 원본 위치 열기">${item.thumbnail?`<img src="${e(item.thumbnail)}" alt="${e(item.name)}" loading="lazy">`:icon('image')}${Number(item.frame)>0?`<span class="frame-badge">프레임 ${Number(item.frame)+1}</span>`:''}</button><div class="card-details"><span class="card-title" title="${e(item.path)}">${e(item.name)}</span><div class="card-meta">${labelBadge(item.label)}<span class="score" title="판별 신뢰도">${percent(item.score)}</span></div>${item.metadata?.sample?`<p class="field-help">${e(item.metadata.sample)}</p>`:''}</div></article>`).join('')}</div>`:empty('기준을 통과한 검색 결과가 없습니다','검색어와 필터를 확인하세요. 통과 기준을 낮추면 오탐이 늘어날 수 있습니다.')}<div class="dataset-footer"><span>${icon('shield')}이미지를 누르면 원본 위치가 열립니다</span>${pager('search',s.offset,s.limit,result.total)}</div></section>`:`<section class="dataset-panel">${empty(runs.length?'찾고 싶은 이미지의 모습을 적어보세요':'먼저 판별 모델을 학습해주세요',runs.length?'검색 색인을 한 번 생성한 후, 이미지의 내용이나 시료 정보로 찾아보세요.<br>검색 전에 SEM · TEM 여부와 신뢰도를 먼저 확인합니다.':'라벨이 있는 데이터로 분류 모델을 만들면<br>SEM과 TEM만 통과시키는 로컬 이미지 검색을 시작할 수 있습니다.',runs.length?'':`<button class="button" data-view="training">학습 스튜디오로 이동 ${icon('arrow')}</button>`)}</section>`)
    + `<p class="dataset-note">${icon('info')}텍스트 검색은 기본 EmbeddingGemma 2의 이미지 · 텍스트 임베딩을 사용합니다. 미세조정한 모델은 판별에 사용하며, 모델 또는 데이터 변경 뒤에는 색인을 다시 생성하세요.</p>`;
}
function renderRuns() {
  return heading('EXPERIMENT HISTORY','실행 기록','학습 결과와 독립 평가 지표를 비교하고, 모델을 보관하세요.',`<button class="button" data-action="refresh">${icon('refresh')}새로고침</button>`)
    + (!state.runs.length?`<section class="dataset-panel">${empty('아직 학습 기록이 없습니다','첫 학습이 완료되면 모델과 평가 결과가 여기에 저장됩니다.<br>실행별 모델을 선택해 검색에 사용하거나 내보낼 수 있습니다.',`<button class="button primary" data-view="training">${icon('spark')}학습 시작하기</button>`)}</section>`:`<div class="table-wrap"><table><thead><tr><th>학습 실행</th><th>상태</th><th>정밀도</th><th>재현율</th><th>오탐 / 통과</th><th>평가 이미지</th><th>통과 기준</th><th>모델</th></tr></thead><tbody>${state.runs.map(run=>{const m=run.metrics||{},failed=['failed','cancelled','canceled'].includes(run.status);return `<tr><td><strong>${run.mode==='finetune'?'미세조정 · 마지막 2개 블록':'임베딩 + 분류기'}</strong><small title="${e(run.id)}">${e(date(run.created_at))}</small></td><td><span class="run-status ${failed?'failed':''}">${failed?'미완료':'완료'}</span></td><td>${percent(m.precision)}</td><td>${percent(m.recall)}</td><td>${m.false_positives==null?'—':number(m.false_positives)} / ${m.accepted==null?'—':number(m.accepted)}</td><td>${m.test_count==null?'—':number(m.test_count)}</td><td>${percent(run.threshold)}</td><td><div class="row-actions"><button class="button small" data-action="run-details" data-id="${e(run.id)}">상세</button><button class="button small" data-action="use-run" data-id="${e(run.id)}" ${failed?'disabled':''}>검색</button><button class="button small" data-action="export-model" data-id="${e(run.id)}" ${failed?'disabled':''} aria-label="모델 내보내기">${icon('download')}</button></div></td></tr>`;}).join('')}</tbody></table></div>`)
    + `<p class="dataset-note">${icon('info')}정밀도·재현율은 SEM/TEM 포함 여부 기준입니다. SEM과 TEM 구분 정확도는 상세 평가에서 확인하세요. 학습에 사용하지 않은 평가 그룹에서 계산하며, 통과한 이미지가 없으면 정밀도는 표시하지 않습니다. 작은 평가 표본의 수치는 실제 드라이브 전체의 성능을 보장하지 않습니다.</p>`;
}
function metricCount(value) { return value==null?'—':number(value); }
function confusionTable(metrics,title) {
  if(!metrics?.confusion)return `<p class="field-help">${title} 혼동 행렬이 기록되지 않았습니다.</p>`;
  return `<h3 class="field-label run-detail-heading">${title} · 혼동 행렬</h3><div class="table-wrap"><table><thead><tr><th>실제 라벨 ↓ / 판정 →</th><th>SEM</th><th>TEM</th><th>제외</th></tr></thead><tbody>${['SEM','TEM','OTHER'].map(label=>`<tr><td>${labelBadge(label)}</td>${['SEM','TEM','REJECT'].map(predicted=>`<td>${metricCount(metrics.confusion[label]?.[predicted])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function showRunDetails(run) {
  const test=run.metrics||{},validation=run.validation_metrics||{},failed=['failed','cancelled','canceled'].includes(run.status);
  const rows=[['precision','정밀도 · 포함 여부',true],['recall','재현율 · 포함 여부',true],['subtype_accuracy','SEM / TEM 구분 정확도',true],['test_count','전체 이미지 수',false],['positive_count','실제 SEM + TEM 수',false],['other_count','실제 기타 수',false],['accepted','통과 이미지 수',false],['false_positives','오탐 · 통과한 기타 수',false]];
  let body=`<div class="path-box">실행 ${e(run.id)}<br>${e(run.mode==='finetune'?'미세조정 · 마지막 2개 블록':'임베딩 + 분류기')} · 통과 기준 ${percent(run.threshold)}</div>`;
  if(failed) {
    body+=`<div class="notice error">${icon('info')}${e(run.error || (run.status==='cancelled'?'학습을 중단했습니다.':'학습이 완료되지 않았습니다.'))}</div><h3 class="field-label run-detail-heading">실행 로그</h3><pre class="job-logs">${e((run.logs||[]).map(line=>typeof line==='string'?line:line.message||JSON.stringify(line)).join('\n') || '기록된 로그가 없습니다.')}</pre>`;
  } else {
    body+=`<p class="field-help">정밀도·재현율은 실제 SEM/TEM이 검색 대상으로 포함되는지를 측정합니다. 구분 정확도는 통과한 실제 SEM/TEM 중 SEM과 TEM 라벨을 맞힌 비율입니다.</p><h3 class="field-label run-detail-heading">검증 · 독립 평가 비교</h3><div class="table-wrap"><table><thead><tr><th>지표</th><th>검증 · 기준 선택</th><th>평가 · 미사용 그룹</th></tr></thead><tbody>${rows.map(([key,label,isPercent])=>`<tr><td>${label}</td><td>${isPercent?percent(validation[key]):metricCount(validation[key])}</td><td>${isPercent?percent(test[key]):metricCount(test[key])}</td></tr>`).join('')}</tbody></table></div><h3 class="field-label run-detail-heading">데이터 분리</h3><div class="table-wrap"><table><thead><tr><th>용도</th><th>이미지</th><th>시료 그룹</th><th>SEM / TEM / 기타</th></tr></thead><tbody>${[['train','학습'],['validation','검증'],['test','평가']].map(([key,label])=>{const split=run.splits?.[key];return `<tr><td>${label}</td><td>${metricCount(split?.images)}</td><td>${metricCount(split?.groups)}</td><td>${['SEM','TEM','OTHER'].map(name=>metricCount(split?.classes?.[name])).join(' / ')}</td></tr>`;}).join('')}</tbody></table></div>${confusionTable(test,'독립 평가')}${confusionTable(validation,'검증')}<p class="field-help">— 표시는 계산할 수 없거나 기록되지 않은 값입니다. ${e(run.notice||'')}</p>`;
  }
  showModal('학습 실행 상세',`${e(date(run.created_at))} · ${failed?'미완료 실행':'검증에서 기준을 선택하고 독립 평가로 측정한 결과입니다.'}`,body,'<button class="button" data-action="close-modal">닫기</button>');
}
function renderJob() {
  const job=state.job;
  if(!job || state.dismissedJob===job.id){$('#job-region').innerHTML='';return;}
  const done=terminal(job),failed=['failed','error'].includes(job.status),p=Math.max(0,Math.min(1,Number(job.progress)||0));
  const titles={import:'이미지 가져오기',setup:'학습 환경 설치',train:'로컬 모델 학습',index:'검색 색인 생성'};
  const logLines=(job.logs||[]).slice(-120).map(line=> typeof line==='string'?line:line.message||JSON.stringify(line));
  if(job.result && state.jobKind==='import') {
    const r=job.result;
    logLines.push(`가져오기 결과 · ${number(r.imported)}개 추가 / ${number(r.duplicates)}개 중복 / ${number(r.corrupt)}개 손상 / ${number(r.unsupported)}개 미지원`);
    for(const item of r.errors || [])logLines.push(`${item.path || ''}\n  ${item.reason || ''}`);
  }
  const logs=logLines.join('\n');
  $('#job-region').innerHTML=`<section class="job-panel"><div class="job-main"><div class="job-icon">${icon(failed?'info':done?'check':'chip')}</div><div class="job-text"><strong>${e(titles[state.jobKind] || job.phase || '로컬 작업')} ${done?failed?'· 실패':success(job)?'· 완료':'· 중단':'· 진행 중'}</strong><p title="${e(job.error || job.message || '')}">${e(job.error || job.message || '작업을 준비하고 있습니다.')}</p></div><span class="job-progress-value">${Math.round(p*100)}%</span><div class="job-controls"><button class="button" data-action="toggle-logs">${state.showLogs?'로그 닫기':'로그'}</button>${done?'<button class="button" data-action="dismiss-job">닫기</button>':'<button class="button" data-action="cancel">중단</button>'}</div></div><div class="job-progress"><progress value="${p}" max="1" aria-label="작업 진행률"></progress></div>${state.showLogs?`<pre class="job-logs">${e(logs || job.error || job.message || '출력을 기다리는 중…')}</pre>`:''}</section>`;
  if(state.showLogs){const log=$('.job-logs');log.scrollTop=log.scrollHeight;}
}
function showModal(title, description, body, footer='') {
  $('#modal-content').innerHTML=`<div class="modal-header"><div><h2>${title}</h2><p>${description}</p></div><button class="icon-button" data-action="close-modal" aria-label="창 닫기">${icon('close')}</button></div><div class="modal-body">${body}</div>${footer?`<div class="modal-footer">${footer}</div>`:''}`;
  if(!$('#modal').open) $('#modal').showModal();
}
function closeModal() { $('#modal').close(); }
function metadataModal(ids) {
  state.modalIds=[...ids];
  const item=ids.length===1?state.dataset.items.find(item=>String(item.id)===String(ids[0])):null;
  const m=item?.metadata||{};
  showModal(ids.length===1?'이미지 정보':`${ids.length}장 정보 편집`,ids.length===1?'시료 정보와 그룹을 편집하고 원본 위치를 확인하세요.':'입력한 정보만 선택한 이미지에 적용합니다. 빈 칸은 변경하지 않습니다.',
    `${item?`<div class="detail-preview">${item.thumbnail?`<img src="${e(item.thumbnail)}" alt="${e(item.name)}">`:icon('image')}</div><div class="path-box">${e(item.path)}${item.frames>1?`<br>프레임 ${Number(item.frame)+1} / ${number(item.frames)}`:''}</div>`:''}<form id="metadata-form"><div class="field"><label class="field-label" for="metadata-group">시료 그룹 · 학습 데이터 분리 기준</label><input id="metadata-group" name="group" type="text" value="${e(item?.group||'')}" placeholder="예: specimen-A"><p class="field-help">같은 시료에서 얻은 이미지는 폴더가 달라도 같은 그룹으로 지정하세요.</p></div><div class="form-divider"></div><div class="field-grid">${[['sample','시료 이름','text'],['acquired_at','취득일','date'],['magnification','배율','text'],['voltage_kv','가속 전압 · kV','number']].map(([key,label,type])=>`<div class="field"><label class="field-label" for="meta-${key}">${label}</label><input id="meta-${key}" name="${key}" type="${type}" value="${e(type==='date'?String(m[key]||'').slice(0,10):m[key]||'')}" ${type==='number'?'min="0" step="any"':''}></div>`).join('')}</div><div class="field"><p class="field-label">설명</p><textarea name="description" aria-label="이미지 설명" placeholder="이미지의 특징이나 관찰 내용을 기록하세요.">${e(m.description||'')}</textarea></div></form>`,`${item?`<button class="button" data-action="reveal-item" data-id="${e(item.id)}">${icon('folder')}원본 위치</button>`:''}<button class="button" data-action="close-modal">취소</button><button class="button primary" data-action="save-metadata">변경 저장</button>`);
}

async function refreshStatus(silent = true) {
  if(!bridge || state.statusBusy) return;
  state.statusBusy=true;
  try {
    const data=await request('status');
    state.project=data.project||state.project;state.environment=data.environment||{};state.runs=data.runs||[];state.indexes=data.indexes||[];state.connected=true;
    if(!state.search.run_id || !state.runs.some(run=>String(run.id)===String(state.search.run_id))) { const first=validRuns()[0]; if(first){state.search.run_id=first.id;state.search.threshold=first.threshold??0.95;}else state.search.run_id=''; }
    const banner=$('#connection-banner'); banner.hidden=true;
    updateChrome();
    if(data.job) receiveJob(data.job);
    if(!silent)render();
  } catch(error) {
    state.connected=false;
    const banner=$('#connection-banner');banner.hidden=false;banner.innerHTML=`<div class="status-banner"><span>${e(errorMessage(error))}</span><button class="button small" data-action="refresh">다시 연결</button></div>`;
  } finally {state.statusBusy=false;}
}
async function loadImages() {
  const version=++state.dataRequest;
  state.dataset.loading=true;if(state.view==='dataset')render();
  try {
    const {filter,query,offset,limit}=state.dataset;
    const result=await request('images',{filter,query,offset,limit});
    if(version!==state.dataRequest)return;
    Object.assign(state.dataset,{items:result.items||[],total:result.total||0});
  } catch(error) {if(version===state.dataRequest)toast(errorMessage(error),true);}
  finally {if(version===state.dataRequest){state.dataset.loading=false;if(state.view==='dataset')render();}}
}
async function completeJob(job) {
  const kind=state.jobKind;
  if(success(job)) {
    if(kind==='import') { const r=job.result||{};toast(`${number(r.imported)}장 가져오기 완료${r.skipped?` · ${number(r.skipped)}개 제외 (로그에서 상세 확인)`:''}`);}
    else if(kind==='index') {state.indexDirty=false;toast('검색 색인이 준비되었습니다.');}
    else if(kind==='setup')toast('학습 환경 설치가 완료되었습니다.');
    else if(kind==='train') {toast('모델 학습이 완료되었습니다. 실행 기록에서 평가 결과를 확인하세요.');state.indexDirty=true;}
  } else if(['failed','error'].includes(job.status))toast(job.error||job.message||'작업이 실패했습니다. 로그를 확인하세요.',true);
  await refreshStatus();
  if(kind==='import'){state.indexDirty=true;state.selected.clear();await loadImages();}
  if(kind==='train'&&success(job)){const first=validRuns()[0];if(first){state.search.run_id=first.id;state.search.threshold=first.threshold??0.95;}}
  render();
  if(kind==='index'&&success(job)&&state.view==='search') await doSearch(true);
}
function receiveJob(job) {
  if(!job)return;
  if(state.job?.id===job.id&&terminal(state.job)&&!terminal(job))return;
  if(state.job?.id!==job.id&&active()&&state.handledJobs.has(job.id))return;
  state.job=job;
  if(job.kind)state.jobKind=job.kind;else if(!state.jobKind)state.jobKind=job.action||(['import','setup','train','index'].includes(job.phase)?job.phase:null);
  renderJob();
  if(terminal(job)&&!state.handledJobs.has(job.id)) {state.handledJobs.add(job.id);completeJob(job).catch(error=>toast(errorMessage(error),true));}
}
async function startJob(action,payload={}) {
  if(active())throw new Error('진행 중인 작업을 완료하거나 중단한 뒤 다시 시도하세요.');
  state.jobKind=action;state.dismissedJob=null;
  const result=await request(action,payload);
  if(result.job)receiveJob(result.job);else {await refreshStatus();toast('작업을 완료했습니다.');}
  render();
  return result;
}
async function applyLabel(label) {
  const ids=[...state.selected].map(Number);if(!ids.length)return;
  await request('label',{ids,label:label||null});state.indexDirty=true;
  toast(`${number(ids.length)}장을 ${label==='OTHER'?'기타':label||'미분류'}로 지정했습니다.`);state.selected.clear();
  await refreshStatus();await loadImages();
}
async function doSearch(reset = false) {
  if(!state.search.run_id)throw new Error('먼저 모델을 학습해주세요.');
  const container=$('#content');
  if(!$$('input',container).every(input=>input.checkValidity())){$$('input',container).find(input=>!input.checkValidity())?.reportValidity();return;}
  if(state.search.date_from&&state.search.date_to&&state.search.date_from>state.search.date_to)throw new Error('취득일 종료 날짜는 시작 날짜보다 빠를 수 없습니다.');
  if(reset)state.search.offset=0;
  const version=++state.searchRequest,snapshot={...state.search,...state.settings};
  state.searchLoading=true;render();
  try {
    const result=await request('search',snapshot);
    if(version!==state.searchRequest)return;
    state.results=result;state.searchSnapshot=snapshot;
  } catch(error) {if(version===state.searchRequest){state.results=null;toast(errorMessage(error),true);}}
  finally {if(version===state.searchRequest){state.searchLoading=false;render();}}
}
async function chooseFolder() {if(!bridge)throw new Error('GemmaScope 데스크톱 앱에서 폴더를 선택할 수 있습니다.');return bridge.chooseFolder();}
async function changeView(view) {
  if(!names[view])return;
  state.view=view;render();
  window.scrollTo({top:0,left:0,behavior:'instant'});
  $('.main-shell').scrollTop=0;
  if(view==='dataset'&&bridge)await loadImages();
  if(view==='runs'||view==='training')await refreshStatus(false);
}
async function handleAction(action,button) {
  switch(action) {
    case 'import': {
      if(active())throw new Error('진행 중인 작업을 먼저 완료해주세요.');
      const path=await chooseFolder();if(!path)return;state.importPath=path;
      showModal('이미지 폴더 가져오기','하위 폴더를 포함해 이미지를 등록합니다. 원본은 변경하지 않습니다.',`<div class="path-box">${e(path)}</div><form id="import-form"><div class="field-grid"><div class="field"><label class="field-label" for="frame-start">첫 프레임 · 0부터 시작</label><input id="frame-start" name="frame_start" type="number" min="0" step="1" value="0" required></div><div class="field"><label class="field-label" for="frame-end">마지막 프레임 · 포함</label><input id="frame-end" name="frame_end" type="number" min="0" step="1" value="0" required></div></div><p class="field-help">스택 이미지는 지정한 프레임만 가져옵니다. 기본값 0–0은 첫 프레임입니다. 일반 이미지는 첫 프레임으로 등록합니다.</p></form><div class="notice">${icon('info')}일반 이미지와 DM3/DM4, SER/EMI, MRC를 지원합니다. 지원하지 않는 신호 · 볼륨, 손상된 파일 및 중복은 제외하고 기록합니다.</div>`,`<button class="button" data-action="close-modal">취소</button><button class="button primary" data-action="confirm-import">${icon('plus')}가져오기</button>`);break;
    }
    case 'confirm-import': {
      const form=$('#import-form');if(!form.reportValidity())return;
      const data=new FormData(form),start=Number(data.get('frame_start')),end=Number(data.get('frame_end'));
      if(end<start)throw new Error('마지막 프레임은 첫 프레임보다 작을 수 없습니다.');
      closeModal();await startJob('import',{path:state.importPath,frame_start:start,frame_end:end});break;
    }
    case 'project': {
      if(active())throw new Error('작업이 진행 중일 때는 프로젝트를 변경할 수 없습니다.');
      const path=await chooseFolder();if(!path)return;
      await request('project',{path});state.selected.clear();state.dataset.offset=0;state.dataset.query='';state.dataset.filter='ALL';state.results=null;state.search.run_id='';state.indexDirty=false;state.job=null;state.jobKind=null;state.view='dataset';await refreshStatus();await loadImages();toast('프로젝트를 열었습니다.');break;
    }
    case 'refresh':await refreshStatus(false);if(state.view==='dataset')await loadImages();break;
    case 'environment':showModal('로컬 실행 환경','모델과 이미지 처리는 이 컴퓨터에서 실행됩니다.',`${environmentPanel()}<section class="panel"><h2>프로젝트 경로</h2><p class="path-box">${e(state.project.root||'아직 연결되지 않음')}</p><h2>원본 폴더 연결</h2><p class="field-help">폴더 위치가 바뀌면 새 위치에 연결하세요. 원본 파일을 이동하지 않습니다.</p>${(state.project.roots||[]).map((root,index)=>`<div class="environment-spec"><span>${e(typeof root==='string'?root:root.path||root.root||'')}</span><button class="button small" data-action="relink" data-index="${index}" ${active()?'disabled':''}>다시 연결</button></div>`).join('')}</section>`);break;
    case 'setup':closeModal();await startJob('setup');break;
    case 'select':{const id=String(button.dataset.id);state.selected.has(id)?state.selected.delete(id):state.selected.add(id);render();break;}
    case 'select-all':state.dataset.items.forEach(item=>state.selected.add(String(item.id)));render();break;
    case 'clear-selection':state.selected.clear();render();break;
    case 'label':await applyLabel(button.dataset.label);break;
    case 'detail':metadataModal([String(button.dataset.id)]);break;
    case 'metadata':metadataModal([...state.selected]);break;
    case 'save-metadata': {
      const form=$('#metadata-form');if(!form.reportValidity())return;
      const data=new FormData(form),ids=[...state.modalIds],metadata={};
      for(const key of ['sample','acquired_at','magnification','voltage_kv','description']) {const value=String(data.get(key)||'').trim();if(value||ids.length===1)metadata[key]=value;}
      if(Object.keys(metadata).length)await request('metadata',{ids:ids.map(Number),metadata});
      const group=String(data.get('group')||'').trim();
      if(group) {
        const batches=new Map();
        for(const item of state.dataset.items.filter(item=>ids.includes(String(item.id)))){const key=item.label||'';if(!batches.has(key))batches.set(key,[]);batches.get(key).push(item.id);}
        for(const [label,batchIds] of batches)await request('label',{ids:batchIds,label:label||null,group});
      }
      closeModal();toast(`${number(ids.length)}장 정보를 저장했습니다.`);await refreshStatus();await loadImages();break;
    }
    case 'metadata-csv':
      showModal('메타데이터 CSV 가져오기','이미 등록한 이미지에 시료와 관찰 정보를 일괄 연결합니다.',`<p class="field-label">사용할 열 이름</p><div class="path-box">path, frame, sample, acquired_at, magnification, voltage_kv, description</div><p class="field-help"><strong>path</strong>는 필수입니다. 원본 파일의 절대 경로 또는 CSV 파일 기준 상대 경로를 사용하세요. <strong>frame</strong>은 0부터 시작하며 생략하면 0입니다.</p><p class="field-help">취득일은 YYYY-MM-DD 형식으로 입력하세요. CSV에 포함한 빈 값은 기존 값을 지웁니다. 파일은 UTF-8 형식으로 저장하세요.</p>`,`<button class="button" data-action="close-modal">취소</button><button class="button primary" data-action="confirm-metadata-csv">${icon('folder')}CSV 파일 선택</button>`);break;
    case 'confirm-metadata-csv': {
      if(!bridge?.importMetadata)throw new Error('데스크톱 앱에서 메타데이터 CSV를 가져올 수 있습니다.');
      const result=await bridge.importMetadata();if(!result)return;
      closeModal();toast(`${number(result.updated)}장 메타데이터 갱신${result.skipped?` · ${number(result.skipped)}개 제외`:''}`);if(result.errors?.length)toast(result.errors.slice(0,3).map(item=>typeof item==='string'?item:`${item.line ? `${item.line}행 · ` : ''}${item.reason || '가져오기 오류'}`).join('\n'),true);await loadImages();break;
    }
    case 'reveal-item':{const item=state.dataset.items.find(item=>String(item.id)===String(button.dataset.id));if(item)await bridge.openPath(item.path);break;}
    case 'reveal-result':{const item=state.results?.items?.[Number(button.dataset.index)];if(item)await bridge.openPath(item.path);break;}
    case 'relink': {
      const root=state.project.roots?.[Number(button.dataset.index)],old=typeof root==='string'?root:root?.path||root?.root;
      if(!old)throw new Error('연결할 원본 폴더를 찾지 못했습니다.');
      const path=await chooseFolder();if(!path)return;const result=await request('relink',{old_root:old,new_root:path});closeModal();toast(`${number(result.updated)}개 파일 경로를 갱신했습니다.`);state.indexDirty=true;await refreshStatus();await loadImages();break;
    }
    case 'train': {
      const invalid=$$('input',$('#content')).find(input=>!input.checkValidity());if(invalid){invalid.reportValidity();return;}
      await startJob('train',{...state.training,...state.settings});break;
    }
    case 'index': {
      const threshold=$('#search-threshold');if(threshold&&!threshold.reportValidity())return;
      await startJob('index',{run_id:state.search.run_id,threshold:Number(state.search.threshold),...state.settings});break;
    }
    case 'search':await doSearch(true);break;
    case 'export-results': {
      if(!state.searchSnapshot||!state.results?.total)return;
      const result=await request('search',{...state.searchSnapshot,offset:0,limit:100000,export:true});
      const path=await bridge.exportResults(result.items||[]);if(path)toast(`${number(result.items?.length)}개 결과를 CSV로 저장했습니다.`);break;
    }
    case 'run-details':{const run=state.runs.find(item=>String(item.id)===String(button.dataset.id));if(run)showRunDetails(run);break;}
    case 'export-model':{const path=await chooseFolder();if(!path)return;await request('export_model',{run_id:button.dataset.id,path});toast('모델을 선택한 폴더로 내보냈습니다.');break;}
    case 'use-run':{const run=state.runs.find(item=>String(item.id)===String(button.dataset.id));if(run){state.search.run_id=run.id;state.search.threshold=run.threshold??0.95;state.results=null;state.indexDirty=true;await changeView('search');}break;}
    case 'page':{const scope=button.dataset.scope;state[scope].offset=Math.max(0,state[scope].offset+Number(button.dataset.direction)*state[scope].limit);if(scope==='dataset'){state.selected.clear();await loadImages();}else await doSearch();break;}
    case 'reset-filter':state.dataset.filter='ALL';state.dataset.query='';state.dataset.offset=0;state.selected.clear();await loadImages();break;
    case 'cancel':await request('cancel');toast('중단을 요청했습니다. 현재 처리 중인 단계가 끝나면 멈춥니다.');break;
    case 'toggle-logs':state.showLogs=!state.showLogs;renderJob();break;
    case 'dismiss-job':state.dismissedJob=state.job?.id;renderJob();break;
    case 'close-modal':closeModal();break;
  }
}

document.addEventListener('click',async event=> {
  const target=event.target.closest('[data-action],[data-view],[data-filter]');if(!target||target.disabled)return;
  if(target.dataset.view){await changeView(target.dataset.view);return;}
  if(target.dataset.filter){state.dataset.filter=target.dataset.filter;state.dataset.offset=0;state.selected.clear();await loadImages();return;}
  if(target.dataset.action){target.disabled=true;try{await handleAction(target.dataset.action,target);}catch(error){toast(errorMessage(error),true);}finally{if(target.isConnected)target.disabled=false;}}
});
let queryTimer;
document.addEventListener('input',event=> {
  const input=event.target;
  if(input.id==='dataset-query'){
    state.dataset.query=input.value;state.dataset.offset=0;state.selected.clear();clearTimeout(queryTimer);
    queryTimer=setTimeout(async()=>{const start=input.selectionStart;await loadImages();const next=$('#dataset-query');if(next){next.focus();next.setSelectionRange(start,start);}},350);return;
  }
  const {scope,key}=input.dataset;if(!scope||!key||!state[scope])return;
  if(input.type==='radio'&&!input.checked)return;
  const value=input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value;
  state[scope][key]=value;
});
document.addEventListener('change',event=> {
  const input=event.target,{scope,key}=input.dataset;if(!scope||!key)return;
  if(scope==='training'&&key==='mode')render();
  if(scope==='search'&&key==='run_id') {const run=chosenRun();state.search.threshold=run?.threshold??0.95;state.results=null;state.indexDirty=true;render();}
});
document.addEventListener('keydown',async event=> {
  if(event.key==='Enter'&&event.target.id==='semantic-query'){event.preventDefault();try{await doSearch(true);}catch(error){toast(errorMessage(error),true);}return;}
  if(event.target.matches('input,textarea,select')||$('#modal').open)return;
  if(state.view!=='dataset')return;
  if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='a'){event.preventDefault();state.dataset.items.forEach(item=>state.selected.add(String(item.id)));render();return;}
  if(event.key==='Escape'){state.selected.clear();render();return;}
  if(state.selected.size&&['0','1','2','3'].includes(event.key)){event.preventDefault();try{await applyLabel({'0':null,'1':'SEM','2':'TEM','3':'OTHER'}[event.key]);}catch(error){toast(errorMessage(error),true);}}
});
$('#modal').addEventListener('click',event=>{if(event.target===$('#modal')){const rect=$('#modal').getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)closeModal();}});
document.querySelector('.brand').addEventListener('click',event=>{event.preventDefault();changeView('dataset');});

async function initialize() {
  render();
  if(!bridge){const banner=$('#connection-banner');banner.hidden=false;banner.innerHTML='<div class="status-banner"><span>UI 미리보기입니다. 폴더 가져오기와 로컬 학습은 GemmaScope 데스크톱 앱에서 실행하세요.</span></div>';return;}
  bridge.onEvent(event=>{if(event.type==='job')receiveJob(event.job);else if(event.type==='error')toast(event.message || '로컬 프로세스 연결이 끊어졌습니다.',true);});
  await refreshStatus();await loadImages();
  setInterval(()=>{refreshStatus().catch(error=>toast(errorMessage(error),true));},3000);
}
initialize().catch(error=>toast(errorMessage(error),true));

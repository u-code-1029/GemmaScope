'use strict';
const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const readline = require('node:readline');
const { spawn, spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const { validRequest, resultsCSV, pythonCandidates } = require('./util.cjs');

const root = path.resolve(__dirname, '..');
const dataDir = path.resolve(process.env.GEMMASCOPE_DATA_DIR || (app.isPackaged ? path.join(app.getPath('userData'), 'workspace') : path.join(root, '.gemmascope')));
app.setPath('userData', path.join(dataDir, 'desktop'));
const backendDir = app.isPackaged ? path.join(process.resourcesPath, 'backend') : path.join(root, 'backend');
let window, worker, interpreter, setupProcess, setupJob, shuttingDown = false, workerReady;
let requestID = 0;
const pending = new Map();
const allowedRoots = new Set();
const settingsFile = path.join(dataDir, 'settings.json');
let projectDir = path.join(dataDir, 'project');

function emit(event) { if (window && !window.isDestroyed()) window.webContents.send('worker-event', event); }
function checkSender(event) {
  if (!window || event.sender !== window.webContents || event.senderFrame !== window.webContents.mainFrame) throw new Error('허용되지 않은 요청입니다.');
}
function env() {
  return { ...process.env, PYTHONUNBUFFERED: '1', PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', HF_HUB_DISABLE_TELEMETRY: '1', DO_NOT_TRACK: '1', TOKENIZERS_PARALLELISM: 'false', HF_HOME: path.join(dataDir, 'models'), MPLCONFIGDIR: path.join(dataDir, 'matplotlib') };
}
function findPython(skipVenv = false) {
  for (const candidate of pythonCandidates(root, dataDir, app.isPackaged, process.resourcesPath)) {
    if (skipVenv && candidate.includes(path.join(dataDir, 'venv'))) continue;
    const result = spawnSync(candidate, ['-c', 'import sys; print(sys.executable); sys.exit(0 if sys.version_info >= (3,12) else 1)'], { encoding: 'utf8', timeout: 15000, windowsHide: true });
    if (result.status === 0) return result.stdout.trim();
  }
  throw new Error('Python 3.12 이상을 찾을 수 없습니다. 배포 앱을 다시 설치하거나 GEMMASCOPE_PYTHON에 Python 실행 경로를 지정하세요.');
}
function failPending(message) {
  for (const { reject, timer } of pending.values()) { clearTimeout(timer); reject(new Error(message)); }
  pending.clear();
}
function startWorker() {
  interpreter = findPython();
  const child = spawn(interpreter, ['-u', path.join(backendDir, 'server.py'), '--data-dir', projectDir, '--cache-dir', path.join(dataDir, 'models')], { env: env(), stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
  worker = child;
  let stderr = '';
  child.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-12000); if (!app.isPackaged) process.stderr.write(chunk); });
  readline.createInterface({ input: child.stdout }).on('line', line => {
    let message; try { message = JSON.parse(line); } catch { return; }
    if (message.event) { emit(message.event); return; }
    const waiting = pending.get(message.id);
    if (!waiting) return;
    clearTimeout(waiting.timer); pending.delete(message.id);
    if (message.error) waiting.reject(new Error(message.error)); else waiting.resolve(message.result);
  });
  child.on('error', err => failPending('학습 프로세스를 시작하지 못했습니다: ' + err.message));
  child.on('exit', code => {
    if (worker === child) {
      worker = null;
      failPending('학습 프로세스가 종료되었습니다. 앱을 다시 열어주세요. ' + stderr.slice(-2000));
      if (!shuttingDown && !setupJob) emit({ type: 'error', message: `학습 프로세스 종료 (${code}). ${stderr.slice(-2000)}` });
    }
  });
  return child;
}
function request(action, payload = {}) {
  if (!worker || worker.killed) return Promise.reject(new Error('로컬 프로세스에 연결되지 않았습니다. 앱을 다시 열어주세요.'));
  const id = ++requestID;
  return new Promise((resolve, reject) => {
    const timeout = action === 'export_model' ? 1800000 : ['search', 'relink', 'import_metadata'].includes(action) ? 300000 : 120000;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('응답 시간이 초과되었습니다. 실행 기록을 확인해주세요.')); }, timeout);
    pending.set(id, { resolve, reject, timer });
    worker.stdin.write(JSON.stringify({ id, action, payload }) + '\n', error => {
      if (error) { clearTimeout(timer); pending.delete(id); reject(error); }
    });
  });
}
function updateSetup(values) {
  Object.assign(setupJob, values);
  emit({ type: 'job', job: { ...setupJob, logs: [...setupJob.logs] } });
}
function runSetupCommand(binary, args, phase) {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { env: env(), windowsHide: true });
    setupProcess = child;
    for (const stream of [child.stdout, child.stderr]) readline.createInterface({ input: stream }).on('line', text => {
      if (!text.trim()) return;
      setupJob.logs.push(text.slice(-1000)); setupJob.logs = setupJob.logs.slice(-100);
      updateSetup({ phase, message: text.slice(-500) });
    });
    child.on('error', reject);
    child.on('exit', code => { setupProcess = null; if (code === 0) resolve(); else reject(new Error(setupJob.cancelled ? '환경 설치를 취소했습니다.' : `환경 설치 실패 (${code}). 실행 로그를 확인한 뒤 다시 시도하세요.`)); });
  });
}
async function setup() {
  if (setupJob?.status === 'running') throw new Error('환경 설치가 이미 진행 중입니다.');
  const status = await request('status');
  if (status.job?.status === 'running') throw new Error('진행 중인 작업이 끝난 후 환경을 설치하세요.');
  setupJob = { id: 'setup-' + Date.now(), status: 'running', kind: 'setup', phase: 'setup', progress: 0, message: '독립된 Python 환경을 준비합니다.', logs: [] };
  const response = { job: { ...setupJob } };
  (async () => {
    try {
      const python = findPython(true);
      const venvDir = path.join(dataDir, 'venv');
      await runSetupCommand(python, ['-m', 'venv', venvDir], '환경 생성');
      const venvPython = path.join(venvDir, ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3']));
      updateSetup({ progress: 0.1, message: '학습 패키지를 내려받습니다. 인터넷 연결과 수 GB의 여유 공간이 필요합니다.' });
      if (process.platform === 'win32') {
        const nvidia = spawnSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
        if (nvidia.status === 0 && nvidia.stdout.trim()) {
          updateSetup({ message: 'NVIDIA GPU를 감지했습니다. 공식 CUDA 13.0용 PyTorch를 설치합니다.' });
          await runSetupCommand(venvPython, ['-m', 'pip', 'install', '--disable-pip-version-check', '--cache-dir', path.join(dataDir, 'pip-cache'), 'torch==2.14.1', 'torchvision==0.29.1', '--index-url', 'https://download.pytorch.org/whl/cu130'], 'CUDA 설치');
        }
      }
      await runSetupCommand(venvPython, ['-m', 'pip', 'install', '--disable-pip-version-check', '--cache-dir', path.join(dataDir, 'pip-cache'), '-r', path.join(backendDir, 'requirements-ml.txt')], '패키지 설치');
      const oldWorker = worker; worker = null; oldWorker?.kill();
      failPending('환경 설치 후 연결을 갱신합니다.');
      startWorker();
      const verification = await request('status');
      if (!verification.environment.ml_ready) throw new Error('패키지 설치 후 환경 점검을 통과하지 못했습니다: ' + verification.environment.missing.join(', '));
      updateSetup({ status: 'completed', progress: 1, message: '학습 환경이 준비되었습니다. 모델은 첫 학습 또는 인덱싱 시 내려받습니다.', result: verification.environment });
    } catch (error) {
      updateSetup({ status: setupJob.cancelled ? 'cancelled' : 'failed', error: error.message, message: error.message });
    }
  })();
  return response;
}
function authorizedDirectory(directory) {
  if (typeof directory !== 'string' || !allowedRoots.has(path.resolve(directory))) throw new Error('폴더 선택 버튼으로 경로를 선택해주세요.');
}

app.whenReady().then(async () => {
  await fsp.mkdir(dataDir, { recursive: true });
  try { const saved = JSON.parse(await fsp.readFile(settingsFile, 'utf8')); if (saved.projectDir && fs.existsSync(saved.projectDir)) projectDir = saved.projectDir; } catch {}
  allowedRoots.add(path.resolve(projectDir));
  try { startWorker(); } catch (error) { dialog.showErrorBox('GemmaScope 실행 오류', error.message); app.quit(); return; }
  window = new BrowserWindow({ width: 1440, height: 960, minWidth: 1000, minHeight: 680, title: 'GemmaScope', backgroundColor: '#f6f7fb', show: false, webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true } });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  window.once('ready-to-show', () => window.show());
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{ label: 'GemmaScope', submenu: [{ role: 'about' }, { type: 'separator' }, { role: 'hide' }, { role: 'quit' }] }] : []),
    { label: '편집', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: '보기', submenu: [{ role: 'reload' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }, ...(!app.isPackaged ? [{ role: 'toggleDevTools' }] : [])] }
  ]));
  ipcMain.handle('request', async (event, action, payload = {}) => {
    checkSender(event); validRequest(action, payload);
    if (action === 'setup') return setup();
    if (action === 'cancel' && setupJob?.status === 'running') { setupJob.cancelled = true; setupProcess?.kill(); return { cancelled: true }; }
    if (setupJob?.status === 'running' && action !== 'status') throw new Error('환경 설치가 끝날 때까지 기다려주세요.');
    if (['import', 'project', 'export_model'].includes(action)) authorizedDirectory(payload.path);
    if (action === 'relink') authorizedDirectory(payload.new_root);
    const result = await request(action, payload);
    if (action === 'status' && setupJob) result.job = setupJob;
    if (action === 'project') {
      projectDir = payload.path;
      await fsp.writeFile(settingsFile, JSON.stringify({ projectDir }));
    }
    if (action === 'status' && setupJob && setupJob.status !== 'running') setupJob = null;
    return result;
  });
  ipcMain.handle('choose-folder', async event => {
    checkSender(event);
    const result = await dialog.showOpenDialog(window, { title: '폴더 선택', properties: ['openDirectory', 'createDirectory'] });
    if (result.canceled || !result.filePaths[0]) return null;
    const selected = path.resolve(result.filePaths[0]); allowedRoots.add(selected); return selected;
  });
  ipcMain.handle('export-results', async (event, rows) => {
    checkSender(event);
    if (!Array.isArray(rows) || rows.length > 100000) throw new Error('내보내기 결과가 올바르지 않습니다.');
    const result = await dialog.showSaveDialog(window, { title: '검색 결과 내보내기', defaultPath: 'gemmascope-results.csv', filters: [{ name: 'CSV', extensions: ['csv'] }] });
    if (result.canceled || !result.filePath) return null;
    await fsp.writeFile(result.filePath, resultsCSV(rows), 'utf8'); return result.filePath;
  });
  ipcMain.handle('import-metadata', async event => {
    checkSender(event);
    const selected = await dialog.showOpenDialog(window, { title: '메타데이터 CSV 가져오기', properties: ['openFile'], filters: [{ name: 'CSV', extensions: ['csv'] }] });
    if (selected.canceled || !selected.filePaths[0]) return null;
    return request('import_metadata', { path: selected.filePaths[0] });
  });
  ipcMain.handle('open-path', async (event, file) => {
    checkSender(event);
    if (typeof file !== 'string' || !path.isAbsolute(file) || !fs.existsSync(file)) throw new Error('원본 파일을 찾을 수 없습니다. 드라이브 연결과 경로를 확인하세요.');
    const allowed = await request('known_path', { path: file });
    if (!allowed.known) throw new Error('등록된 원본 파일만 열 수 있습니다.');
    shell.showItemInFolder(file);
  });
  await window.loadFile(path.join(root, 'renderer', 'index.html'));
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { shuttingDown = true; setupProcess?.kill(); worker?.kill(); failPending('앱이 종료되었습니다.'); });

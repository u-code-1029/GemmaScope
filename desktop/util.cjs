'use strict';
const path = require('node:path');
const fs = require('node:fs');

const ACTIONS = new Set(['status', 'images', 'remove_images', 'restore_images', 'label', 'metadata', 'train', 'cancel', 'index', 'search', 'runs', 'import', 'project', 'relink', 'export_model', 'setup']);
function csvCell(value) {
  let text = value == null ? '' : String(value);
  // Spreadsheet formulas must remain inert when the exported CSV is opened.
  if (/^[\s]*[=+@-]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}
function resultsCSV(rows) {
  const headers = ['name', 'path', 'frame', 'label', 'score', 'similarity', 'sample', 'acquired_at', 'magnification', 'voltage_kv', 'description'];
  return '\uFEFF' + [headers.join(','), ...rows.map(row => headers.map(key => csvCell(row[key] ?? row.metadata?.[key])).join(','))].join('\r\n');
}
function within(candidate, root) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}
function validRequest(action, payload) {
  if (!ACTIONS.has(action)) throw new Error('지원하지 않는 작업입니다.');
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('잘못된 요청입니다.');
  if (JSON.stringify(payload).length > 4 * 1024 * 1024) throw new Error('요청이 너무 큽니다.');
}
function pythonCandidates(root, dataDir, packaged, resources) {
  const executable = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3'];
  const bundled = process.platform === 'win32' ? 'python.exe' : 'bin/python3';
  return [
    path.join(dataDir, 'venv', ...executable),
    process.env.GEMMASCOPE_PYTHON,
    packaged ? path.join(resources, 'python', bundled) : path.join(root, 'runtime', 'python', bundled),
    !packaged && path.join(require('node:os').homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'),
    ...(process.platform === 'win32' ? ['python'] : ['python3.12', 'python3.13', 'python3'])
  ].filter(Boolean);
}
module.exports = { ACTIONS, csvCell, resultsCSV, within, validRequest, pythonCandidates };

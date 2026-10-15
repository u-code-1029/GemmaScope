'use strict';
// CI copies uv-managed python-build-standalone, never a runner's system Python.
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function execute(binary, args) {
  const result = spawnSync(binary, ['-X', 'utf8', ...args], { encoding: 'utf8', timeout: 180000, windowsHide: true,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONNOUSERSITE: '1' } });
  if (result.error || result.status !== 0) {
    throw new Error(`${binary}: ${result.error?.message || result.stderr || result.stdout || `exit ${result.status}`}`);
  }
  return result.stdout.trim();
}
function inside(candidate, parent) {
  const relative = path.relative(parent, candidate);
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}
function executable(root) { return path.join(root, process.platform === 'win32' ? 'python.exe' : 'bin/python3'); }

async function main() {
  const fromIndex = process.argv.indexOf('--from');
  const source = fromIndex >= 0 ? process.argv[fromIndex + 1] : process.env.GEMMASCOPE_PYTHON_RUNTIME;
  if (!source) throw new Error('Use npm run runtime:prepare -- --from /path/to/standalone/python. See README.md.');
  const origin = await fs.realpath(path.resolve(source));
  const target = path.resolve('runtime/python');
  if (inside(target, origin) || inside(origin, target)) throw new Error('The source runtime and destination must be separate directories.');
  const info = JSON.parse(execute(executable(origin), ['-I', '-c',
    'import sys,struct,json,platform,ssl,sqlite3,venv,ensurepip; assert sys.version_info >= (3,12); assert struct.calcsize("P") == 8; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,"platform":sys.platform,"machine":platform.machine()}))']));
  if (await fs.realpath(info.base_prefix) !== origin || await fs.realpath(info.prefix) !== origin) {
    throw new Error('Supply the standalone Python installation, not a virtual environment or an alias.');
  }
  if (info.platform !== process.platform) throw new Error('The Python runtime platform must match the packaging host.');
  const expected = process.arch === 'arm64' ? /^(arm64|aarch64)$/i : /^(amd64|x86_64)$/i;
  if (!expected.test(info.machine)) throw new Error(`Python architecture ${info.machine} does not match Node ${process.arch}.`);
  const staging = path.resolve('.runtime', `python stage ${process.pid}`);
  const testVenv = path.resolve('.runtime', `venv 검증 ${process.pid}`);
  const backup = path.resolve('.runtime', `python previous ${process.pid}`);
  const remove = directory => fs.rm(directory, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
  let replaced = false;
  try {
    await fs.mkdir(path.dirname(staging), { recursive: true });
    await remove(staging);
    await fs.cp(origin, staging, { recursive: true, verbatimSymlinks: true, filter: item => {
      const relative = path.relative(origin, item).split(path.sep);
      return !relative.some(part => ['site-packages', '__pycache__', 'test', 'tests', 'pkgconfig'].includes(part));
    } });
    execute(executable(staging), ['-I', '-c',
      'import sys,os,ssl,sqlite3,venv,ensurepip; assert os.path.samefile(sys.prefix,sys.argv[1]); assert os.path.samefile(sys.base_prefix,sys.argv[1])', staging]);
    // Exercise relocation, ensurepip and spaces/Unicode before replacing a valid
    // runtime. Creating this virtual environment performs no network access.
    execute(executable(staging), ['-I', '-m', 'venv', testVenv]);
    const venvPython = path.join(testVenv, ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3']));
    execute(venvPython, ['-I', '-m', 'pip', '--version']);
    await remove(testVenv);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await remove(backup);
    try { await fs.rename(target, backup); replaced = true; } catch (error) { if (error.code !== 'ENOENT') throw error; }
    try { await fs.rename(staging, target); } catch (error) {
      if (replaced) await fs.rename(backup, target);
      throw error;
    }
    await remove(backup);
    console.log('Python runtime prepared and venv creation verified:', target);
  } finally {
    await remove(staging);
    await remove(testVenv);
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });

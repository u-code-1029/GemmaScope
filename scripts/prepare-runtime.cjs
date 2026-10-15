'use strict';
// Copy a clean, relocatable CPython 3.12+ distribution into packaging resources.
// CI obtains python-build-standalone via uv; local development can use --from.
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
async function main() {
  const fromIndex = process.argv.indexOf('--from');
  const source = fromIndex >= 0 ? process.argv[fromIndex + 1] : process.env.GEMMASCOPE_PYTHON_RUNTIME;
  if (!source) throw new Error('Set GEMMASCOPE_PYTHON_RUNTIME or use npm run runtime:prepare -- --from /path/to/standalone/python. See README.md.');
  const origin = path.resolve(source);
  const binary = path.join(origin, process.platform === 'win32' ? 'python.exe' : 'bin/python3');
  const check = spawnSync(binary, ['-c', 'import sys,venv,ensurepip; assert sys.version_info >= (3,12); print(sys.version)'], { encoding: 'utf8' });
  if (check.status !== 0) throw new Error('A standalone Python >=3.12 with venv and ensurepip is required: ' + check.stderr);
  const target = path.resolve('runtime/python');
  await fs.rm(target, { recursive: true, force: true });
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.cp(origin, target, { recursive: true, verbatimSymlinks: true, filter: item => {
    const relative = path.relative(origin, item).split(path.sep);
    return !relative.includes('site-packages') && !relative.includes('__pycache__') && !relative.includes('test') && !relative.includes('tests') && !relative.includes('pkgconfig');
  } });
  const verify = spawnSync(path.join(target, process.platform === 'win32' ? 'python.exe' : 'bin/python3'), ['-c', 'import sys,ssl,sqlite3,venv,ensurepip; print(sys.prefix)'], { encoding: 'utf8' });
  if (verify.status !== 0 || !verify.stdout.includes(target)) throw new Error('Copied runtime failed relocation verification: ' + verify.stderr);
  console.log('Python runtime prepared:', target);
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });

'use strict';
// Validates that the distributed app boots with its bundled Python, without ML packages.
const { _electron: electron } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

async function main() {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'test-results', 'packaged');
  const executable = path.resolve(process.argv[2] || (process.platform === 'win32'
    ? path.join(root, 'release', 'win-unpacked', 'GemmaScope.exe')
    : path.join(root, 'release', `mac-${process.arch}`, 'GemmaScope.app', 'Contents', 'MacOS', 'GemmaScope')));
  await fs.mkdir(output, { recursive: true });
  const appData = path.join(output, 'app data 한글');
  await fs.rm(appData, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
  const env = { ...process.env, GEMMASCOPE_DATA_DIR: appData };
  delete env.GEMMASCOPE_PYTHON;
  delete env.PYTHONPATH;
  delete env.PYTHONHOME;
  delete env.ELECTRON_RUN_AS_NODE;
  const errors = [];
  const app = await electron.launch({ executablePath: executable, args: [], env, timeout: 60000 });
  try {
    const page = await app.firstWindow({ timeout: 60000 });
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForSelector('[data-action="import"]');
    const status = await page.evaluate(() => window.desktop.request('status'));
    const bundle = await app.evaluate(({ app }) => ({ packaged: app.isPackaged, resources: process.resourcesPath }));
    assert.equal(bundle.packaged, true);
    const relativePython = path.relative(path.join(bundle.resources, 'python'), status.environment.python);
    assert.ok(relativePython && relativePython !== '..' && !relativePython.startsWith('..' + path.sep) && !path.isAbsolute(relativePython), status.environment.python);
    assert.equal(status.environment.ml_ready, false);
    assert.equal(status.project.total, 0);
    assert.deepEqual(errors, []);
    const smokeVenv = path.join(output, 'standalone venv 한글');
    await fs.rm(smokeVenv, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
    const created = spawnSync(status.environment.python, ['-I', '-m', 'venv', smokeVenv], { env, encoding: 'utf8', timeout: 180000, windowsHide: true });
    assert.equal(created.status, 0, created.error?.message || created.stderr);
    const venvPython = path.join(smokeVenv, ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3']));
    const pip = spawnSync(venvPython, ['-I', '-m', 'pip', '--version'], { env, encoding: 'utf8', timeout: 30000, windowsHide: true });
    assert.equal(pip.status, 0, pip.error?.message || pip.stderr);
    await fs.rm(smokeVenv, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
    await page.locator('.nav-item[data-view="training"]').click();
    await page.screenshot({ path: path.join(output, 'first-launch.png') });
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, executable, checks: ['packaged startup', 'bundled Python', 'empty project', 'initial setup screen', 'paths with spaces and Korean', 'bundled Python venv and ensurepip', 'no renderer errors'], bundle, environment: status.environment, errors }, null, 2));
    console.log('Packaged app smoke passed:', status.environment.python);
  } catch (error) {
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: false, executable, error: error.stack || error.message, errors }, null, 2));
    const page = app.windows()[0];
    if (page) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    throw error;
  } finally { await app.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

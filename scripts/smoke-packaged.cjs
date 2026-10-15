'use strict';
// Validates that the distributed app boots with its bundled Python, without ML packages.
const { _electron: electron } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'test-results', 'packaged');
  const executable = process.argv[2] || path.join(root, 'release', 'mac-arm64', 'GemmaScope.app', 'Contents', 'MacOS', 'GemmaScope');
  await fs.mkdir(output, { recursive: true });
  await fs.rm(path.join(output, 'app-data'), { recursive: true, force: true });
  const env = { ...process.env, GEMMASCOPE_DATA_DIR: path.join(output, 'app-data') };
  delete env.GEMMASCOPE_PYTHON;
  const errors = [];
  const app = await electron.launch({ executablePath: executable, args: [], env });
  try {
    const page = await app.firstWindow();
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForSelector('[data-action="import"]');
    const status = await page.evaluate(() => window.desktop.request('status'));
    const bundle = await app.evaluate(({ app }) => ({ packaged: app.isPackaged, resources: process.resourcesPath }));
    assert.equal(bundle.packaged, true);
    assert.ok(status.environment.python.startsWith(path.join(bundle.resources, 'python') + path.sep), status.environment.python);
    assert.equal(status.environment.ml_ready, false);
    assert.equal(status.project.total, 0);
    assert.deepEqual(errors, []);
    await page.locator('.nav-item[data-view="training"]').click();
    await page.screenshot({ path: path.join(output, 'first-launch.png') });
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, checks: ['packaged startup', 'bundled Python', 'empty project', 'initial setup screen', 'no renderer errors'], bundle, environment: status.environment, errors }, null, 2));
    console.log('Packaged app smoke passed:', status.environment.python);
  } finally { await app.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

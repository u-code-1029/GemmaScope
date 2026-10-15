'use strict';
// Run with NODE_PATH pointing to a Playwright installation (dev-only).
const { _electron: electron } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const assert = require('node:assert/strict');
async function main() {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'test-results', 'desktop');
  const fixtures = path.join(output, 'fixtures');
  await fs.rm(path.join(output, 'app-data'), { recursive: true, force: true });
  await fs.mkdir(fixtures, { recursive: true });
  const python = path.join(root, '.gemmascope', 'venv', ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3']));
  const generated = spawnSync(python, ['-c', 'from PIL import Image; import sys; from pathlib import Path; p=Path(sys.argv[1]); [Image.new("RGB",(120+i*10,100),(30+i*60,40,60)).save(p/f"TEST-ONLY-{i}.png") for i in range(3)]', fixtures], { encoding: 'utf8' });
  if (generated.status) throw new Error(generated.stderr);
  const errors = [];
  const app = await electron.launch({ executablePath: require('electron'), args: [root], env: { ...process.env, GEMMASCOPE_DATA_DIR: path.join(output, 'app-data'), GEMMASCOPE_PYTHON: python } });
  try {
    const page = await app.firstWindow();
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.waitForSelector('[data-action="import"]');
    await page.waitForFunction(() => !document.querySelector('#device-pill').textContent.includes('확인 중'));
    await page.screenshot({ path: path.join(output, 'dataset-empty.png') });
    await app.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, fixtures);
    await page.locator('[data-action="import"]').first().click();
    await page.locator('[data-action="confirm-import"]').click();
    await page.waitForFunction(() => document.querySelectorAll('.image-card').length === 3, { timeout: 30000 });
    await page.locator('[data-action="select"]').first().click();
    await page.locator('[data-action="label"][data-label="SEM"]').click();
    await page.waitForFunction(() => document.querySelector('.image-card .label-badge')?.textContent.includes('SEM'));
    await page.locator('[data-action="detail"]').first().click();
    await page.locator('#metadata-group').fill('specimen-A');
    await page.locator('#meta-sample').fill('시료 A');
    await page.locator('[data-action="save-metadata"]').click();
    await page.waitForFunction(() => !document.querySelector('#modal').open);
    await page.screenshot({ path: path.join(output, 'dataset-imported.png') });
    for (const tab of ['training', 'search', 'runs']) {
      await page.locator(`.nav-item[data-view="${tab}"]`).click();
      await page.screenshot({ path: path.join(output, tab + '.png') });
    }
    const status = await page.evaluate(() => window.desktop.request('status'));
    assert.equal(status.project.total, 3);
    assert.equal(status.project.counts.SEM, 1);
    assert.deepEqual(errors, []);
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, checks: ['startup', 'native dialog import', 'thumbnail rendering', 'label', 'metadata group edit', 'all four views', 'no renderer errors'], environment: status.environment, errors }, null, 2));
    console.log('Desktop smoke passed; screenshots:', output);
  } catch (error) {
    const page = await app.firstWindow();
    await page.screenshot({ path: path.join(output, 'failure.png') });
    console.error('Renderer errors:', errors);
    console.error(await page.evaluate(() => document.body.innerText));
    throw error;
  } finally { await app.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

'use strict';

// Actual Electron/Python UI checks; no model downloads or inference are needed.
// GEMMASCOPE_TEST_PYTHON may point to the minimal Python environment used by CI.
const { _electron: electron } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'test-results', 'workbench');
const fixtures = path.join(output, 'fixtures');
const dataDir = path.join(output, 'app-data');
const command = process.platform === 'darwin' ? 'Meta' : 'Control';
const checks = [];
const errors = [];

function findPython() {
  const relative = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3'];
  const explicit = process.env.GEMMASCOPE_TEST_PYTHON || process.env.GEMMASCOPE_PYTHON;
  const candidates = explicit ? [explicit] : [path.join(root, '.venv', ...relative), path.join(root, '.gemmascope', 'venv', ...relative), 'python3'];
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['-c', 'import sys; from PIL import Image; sys.exit(0 if sys.version_info >= (3,12) else 1)'], { encoding: 'utf8', timeout: 15000, windowsHide: true });
    if (result.status === 0) return candidate;
  }
  throw new Error('Set GEMMASCOPE_TEST_PYTHON to Python 3.12+ with Pillow installed. This test does not install dependencies.');
}

async function prepareFixtures(python) {
  await fs.mkdir(output, { recursive: true });
  await fs.rm(dataDir, { recursive: true, force: true });
  await fs.rm(fixtures, { recursive: true, force: true });
  await fs.mkdir(fixtures, { recursive: true });
  const code = [
    'from PIL import Image',
    'from pathlib import Path',
    'import sys',
    'folder=Path(sys.argv[1])',
    'for i in range(41):',
    '    image=Image.new("RGB", (96+i%7, 96+i%5), ((i*5)%256, (i*11+20)%256, (i*19+50)%256))',
    '    image.save(folder/f"TEST-ONLY-{i:03d}.png")',
  ].join('\n');
  const result = spawnSync(python, ['-c', code, fixtures], { encoding: 'utf8', timeout: 30000, windowsHide: true });
  if (result.status !== 0) throw new Error(result.stderr || String(result.error || 'Fixture generation failed.'));
}

async function digest(file) {
  return createHash('sha256').update(await fs.readFile(file)).digest('hex');
}

async function main() {
  const python = findPython();
  await prepareFixtures(python);
  const originals = new Map();
  for (const name of await fs.readdir(fixtures)) originals.set(path.join(fixtures, name), await digest(path.join(fixtures, name)));
  const app = await electron.launch({
    executablePath: require('electron'),
    args: [root],
    env: { ...process.env, GEMMASCOPE_DATA_DIR: dataDir, GEMMASCOPE_PYTHON: python },
  });
  let page;
  const measurements = {};
  try {
    page = await app.firstWindow();
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    const card = id => page.locator(`.image-card[data-image-id="${id}"]`);
    const thumb = id => card(id).locator('.thumb-button');
    const visibleIds = () => page.locator('.image-card[data-image-id]').evaluateAll(cards => cards.map(item => item.dataset.imageId));
    const ready = async (count = null) => page.waitForFunction(wanted =>
      state.connected && !active() && !libraryUI.busy && !libraryUI.switching && !state.dataset.loading &&
      !document.querySelector('#content .loading-inline') &&
      (wanted == null || document.querySelectorAll('.image-card[data-image-id]').length === wanted), count, { timeout: 30000 });
    const selected = async ids => {
      const expected = ids.map(String).sort();
      await page.waitForFunction(wanted => {
        const actual = [...document.querySelectorAll('.image-card.selected[data-image-id]')].map(item => item.dataset.imageId).sort();
        return JSON.stringify(actual) === JSON.stringify(wanted);
      }, expected);
    };
    const clear = async () => {
      const button = page.locator('#selection-tools [data-action="clear-selection"]');
      if (await button.count()) await button.click();
      await selected([]);
    };
    const openLayout = async () => {
      if (!(await page.locator('#layout-controls').evaluate(element => element.open))) await page.locator('#layout-controls > summary').click();
      await page.locator('#layout-columns').waitFor({ state: 'visible' });
    };
    const closeLayout = async () => {
      if (await page.locator('#layout-controls').evaluate(element => element.open)) await page.locator('#layout-controls > summary').click();
    };
    const setLayout = async (columns, rows) => {
      await openLayout();
      await page.locator('#layout-columns').selectOption(String(columns));
      await ready();
      await page.locator('#layout-rows').fill(String(rows));
      await page.locator('#layout-rows').press('Tab');
      await ready(columns === 'auto' ? null : columns * rows);
      await page.waitForFunction(({ columns, rows }) => state.dataset.layout.columns === columns && state.dataset.layout.rows === rows,
        { columns, rows });
    };
    const scrollGrid = async () => {
      await page.locator('.dataset-grid').evaluate(element => element.scrollIntoView({ block: 'start', behavior: 'instant' }));
    };
    const rowDragGeometry = async (ids, fromImage = false) => {
      await scrollGrid();
      const first = await card(ids[0]).boundingBox();
      const last = await card(ids.at(-1)).boundingBox();
      assert.ok(first && last, 'Drag targets must have visible bounds');
      return {
        start: { x: fromImage ? first.x + first.width / 2 : first.x - 8, y: fromImage ? first.y + 45 : first.y - 8 },
        end: { x: fromImage ? last.x + last.width / 2 : last.x + last.width - 3, y: first.y + 70 },
      };
    };
    const beginDrag = async (geometry, modifier) => {
      if (modifier) await page.keyboard.down(modifier);
      await page.mouse.move(geometry.start.x, geometry.start.y);
      await page.mouse.down();
      await page.mouse.move(geometry.end.x, geometry.end.y, { steps: 8 });
      await page.locator('.selection-marquee').waitFor({ state: 'visible' });
    };
    const endDrag = async modifier => {
      await page.mouse.up();
      if (modifier) await page.keyboard.up(modifier);
      await page.locator('.selection-marquee').waitFor({ state: 'detached' });
    };
    const dataset = filter => page.evaluate(filter => window.desktop.request('images', { filter, offset: 0, limit: 200 }), filter || 'ALL');
    const assertAutoCapacity = async () => {
      await page.waitForFunction(() => {
        const grid = document.querySelector('.dataset-grid');
        if (!grid || state.dataset.loading || libraryUI.busy) return false;
        const columns = getComputedStyle(grid).gridTemplateColumns.split(/\s+/).length;
        return state.dataset.limit === columns * state.dataset.layout.rows && grid.children.length === Math.min(state.dataset.total, state.dataset.limit);
      });
      return page.locator('.dataset-grid').evaluate(grid => ({
        columns: getComputedStyle(grid).gridTemplateColumns.split(/\s+/).length,
        visible: grid.children.length,
        width: innerWidth,
      }));
    };

    await page.waitForSelector('[data-action="import"]');
    await ready(0);
    await app.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, fixtures);
    await page.locator('[data-action="import"]').first().click();
    await page.locator('[data-action="confirm-import"]').click();
    await page.waitForFunction(() => state.project.total === 41 && !active() && !state.dataset.loading, null, { timeout: 30000 });
    if (await page.locator('[data-action="dismiss-job"]').count()) await page.locator('[data-action="dismiss-job"]').click();
    const allIds = (await dataset()).items.map(item => String(item.id));
    assert.equal(allIds.length, 41);

    await setLayout(2, 2);
    await closeLayout();
    await ready(4);
    let ids = await visibleIds();
    await thumb(ids[0]).click(); await selected([ids[0]]);
    await thumb(ids[1]).click(); await selected(ids.slice(0, 2));
    await thumb(ids[0]).click(); await selected([ids[1]]);
    await card(ids[2]).locator('.selection-box').click(); await selected([ids[1], ids[2]]);
    checks.push('Ordinary clicks add or remove selection without Ctrl; checkbox toggling remains consistent');

    await thumb(ids[1]).dblclick();
    await page.waitForFunction(() => document.querySelector('#modal').open);
    assert.equal(await page.locator('#modal [data-action="reveal-item"]').getAttribute('data-id'), ids[1]);
    assert.equal(await page.locator('#modal .detail-preview').count(), 1);
    await selected([ids[1], ids[2]]);
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('#modal').open);
    await selected([ids[1], ids[2]]);
    assert.equal(await page.evaluate(() => document.activeElement?.dataset.id), ids[1]);
    checks.push('Double-clicking one image in a multi-selection opens that image preview, preserves both selections, and restores focus after closing');

    await clear();
    await thumb(ids[2]).click();
    let geometry = await rowDragGeometry(ids.slice(0, 2));
    await beginDrag(geometry);
    await selected(ids.slice(0, 3));
    await page.screenshot({ path: path.join(output, 'drag-add.png') });
    await endDrag();
    await selected(ids.slice(0, 3));
    assert.equal(await page.locator('#modal').evaluate(element => element.open), false);
    checks.push('Background rectangle dragging adds intersected images while preserving prior selection; pointer release causes no extra click or preview');

    await thumb(ids[0]).focus();
    await page.keyboard.press(`${command}+a`); await selected(ids);
    geometry = await rowDragGeometry([ids[0]]);
    await beginDrag(geometry, 'Shift');
    await selected(ids.slice(1));
    await page.mouse.move(geometry.start.x + 2, geometry.start.y + 2, { steps: 4 });
    await selected(ids);
    await page.mouse.move(geometry.end.x, geometry.end.y, { steps: 4 });
    await selected(ids.slice(1));
    await page.mouse.move(geometry.end.x - 12, geometry.end.y + 8, { steps: 4 });
    await selected(ids.slice(1));
    await page.screenshot({ path: path.join(output, 'shift-drag-remove.png') });
    await endDrag('Shift');
    await selected(ids.slice(1));
    checks.push('After select-all, Shift-drag deselects intersected images; leaving and re-entering the region is calculated from the gesture baseline');

    geometry = await rowDragGeometry([ids[0]]);
    await beginDrag(geometry, 'Shift');
    await selected(ids);
    await page.keyboard.press('Escape');
    await selected(ids.slice(1));
    await endDrag('Shift');
    await selected(ids.slice(1));
    checks.push('Shift-drag can add a previously unselected image, and Escape restores the exact selection from before the gesture');

    await clear();
    await page.evaluate(() => { window.__workbenchNativeDrags = 0; document.addEventListener('dragstart', () => window.__workbenchNativeDrags++); });
    geometry = await rowDragGeometry(ids.slice(0, 2), true);
    await beginDrag(geometry); await selected(ids.slice(0, 2));
    await endDrag(); await selected(ids.slice(0, 2));
    assert.equal(await page.evaluate(() => window.__workbenchNativeDrags), 0);
    assert.equal(await page.locator('#modal').evaluate(element => element.open), false);
    checks.push('Dragging directly from an image performs rectangle selection instead of native image dragging');

    await clear();
    await setLayout(3, 2); await ready(6);
    assert.equal(await page.locator('.dataset-grid').evaluate(grid => getComputedStyle(grid).gridTemplateColumns.split(/\s+/).length), 3);
    const oldHeight = await page.locator('.dataset-grid .thumb-button').first().evaluate(element => element.getBoundingClientRect().height);
    await page.locator('#layout-image-size').focus();
    await page.keyboard.press('Home');
    for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Tab');
    await page.waitForFunction(() => state.dataset.layout.imageSize === 160);
    const newHeight = await page.locator('.dataset-grid .thumb-button').first().evaluate(element => element.getBoundingClientRect().height);
    assert.equal(newHeight, 160);
    assert.notEqual(newHeight, oldHeight);
    await ready(6);
    measurements.imageSize = { oldHeight, newHeight };
    checks.push('Two columns by two rows yields four images; three by two yields six; keyboard range control changes thumbnail height without changing explicit column count');

    await setLayout('auto', 2);
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1000, 900));
    await page.waitForFunction(() => innerWidth === 1000);
    const narrow = await assertAutoCapacity();
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1440, 960));
    await page.waitForFunction(() => innerWidth === 1440);
    const wide = await assertAutoCapacity();
    assert.ok(wide.columns > narrow.columns, 'A wider window should display additional automatic columns');
    assert.equal(narrow.visible, narrow.columns * 2);
    assert.equal(wide.visible, wide.columns * 2);
    measurements.autoLayout = { narrow, wide };
    await page.screenshot({ path: path.join(output, 'layout-options.png') });
    checks.push('Automatic columns and page capacity follow actual rendered column count after native-window resizing');

    await setLayout(2, 6); await closeLayout(); await clear();
    ids = await visibleIds();
    geometry = await rowDragGeometry(ids.slice(0, 2));
    const scrollBefore = await page.evaluate(() => scrollY);
    const initiallyVisible = await page.locator('.image-card[data-image-id]').evaluateAll(cards => cards.filter(card => card.getBoundingClientRect().top < innerHeight).length);
    const bottom = await page.evaluate(() => innerHeight - 6);
    await beginDrag({ start: geometry.start, end: { x: geometry.end.x, y: bottom } });
    await page.waitForFunction(({ scrollBefore, initiallyVisible }) => scrollY > scrollBefore + 80 && state.selected.size > initiallyVisible, { scrollBefore, initiallyVisible });
    const autoScrolled = await page.evaluate(() => ({ scrollY, selected: state.selected.size }));
    await endDrag();
    assert.ok(autoScrolled.selected > initiallyVisible);
    measurements.dragScroll = { scrollBefore, initiallyVisible, ...autoScrolled };
    await page.screenshot({ path: path.join(output, 'drag-auto-scroll.png') });
    checks.push('Holding a drag at the viewport edge scrolls the document and selects images that started below the viewport');

    await clear();
    await setLayout(3, 2);
    await page.locator('#labeling-only').check();
    await ready(6);
    await page.waitForFunction(() => state.dataset.labelingOnly && state.dataset.filter === 'UNLABELED' && state.dataset.offset === 0);
    const preferences = await page.evaluate(() => JSON.parse(localStorage.getItem('gemmascope.library.preferences.v1')));
    assert.deepEqual(preferences, { columns: 3, rows: 2, imageSize: 160, labelingOnly: true });
    await page.reload(); await ready(6);
    assert.equal(await page.locator('#labeling-only').isChecked(), true);
    assert.equal(await page.locator('[data-filter="UNLABELED"]').getAttribute('aria-pressed'), 'true');
    await openLayout();
    assert.equal(await page.locator('#layout-columns').inputValue(), '3');
    assert.equal(await page.locator('#layout-rows').inputValue(), '2');
    assert.equal(await page.locator('#layout-image-size').inputValue(), '160');
    await page.locator('[data-filter="SEM"]').click(); await ready(0);
    assert.equal(await page.locator('#labeling-only').isChecked(), false);
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('gemmascope.library.preferences.v1')).labelingOnly), false);
    await page.locator('[data-filter="ALL"]').click(); await ready(6);
    checks.push('View and continuous-labeling preferences survive reload; choosing another label tab exits continuous mode and persists that choice');

    await setLayout(2, 2); await closeLayout();
    await page.locator('#labeling-only').check(); await ready(4);
    const firstBatch = await visibleIds();
    assert.deepEqual(firstBatch, allIds.slice(0, 4));
    const labeled = new Set();
    const batches = [];
    while (labeled.size < 41) {
      const batch = await visibleIds();
      assert.ok(batch.length > 0, 'Continuous labeling must automatically replenish the current page');
      assert.ok(batch.every(id => !labeled.has(id)), 'Completed images must not reappear in continuous labeling');
      assert.equal(await page.locator('.image-card .label-badge').evaluateAll(badges => badges.every(badge => badge.textContent === '미분류')), true);
      await thumb(batch[0]).focus();
      await page.keyboard.press(`${command}+a`); await selected(batch);
      await page.keyboard.press('1');
      batch.forEach(id => labeled.add(id));
      const remaining = 41 - labeled.size;
      await page.waitForFunction(({ labeled, remaining }) =>
        !libraryUI.busy && !state.dataset.loading && state.project.counts.SEM === labeled && state.project.counts.UNLABELED === remaining &&
        state.dataset.offset === 0 && state.dataset.total === remaining && state.dataset.items.length === Math.min(4, remaining),
      { labeled: labeled.size, remaining }, { timeout: 30000 });
      batches.push(batch);
      if (batches.length === 1) {
        assert.deepEqual(await visibleIds(), allIds.slice(4, 8));
        await page.screenshot({ path: path.join(output, 'continuous-next-batch.png') });
      }
      assert.ok(batches.length <= 11, 'The 41 images should complete in eleven batches of at most four');
    }
    await ready(0);
    assert.match(await page.locator('.empty-state h2').innerText(), /모든 이미지의 라벨링을 마쳤습니다/);
    assert.equal((await dataset('UNLABELED')).total, 0);
    assert.equal((await dataset('SEM')).total, 41);
    assert.equal((await dataset()).total, 41);
    assert.equal(await page.locator('#labeling-only').isChecked(), true);
    assert.equal(batches.length, 11);
    measurements.labeling = { batches: batches.map(batch => batch.length), completed: labeled.size };
    await page.screenshot({ path: path.join(output, 'continuous-complete.png') });
    checks.push('Continuous mode labels all 41 real records in eleven batches without a page-button click, automatically fills the next unlabeled images, and shows completion with accurate label counts');

    await page.reload(); await ready(0);
    assert.equal(await page.locator('#labeling-only').isChecked(), true);
    assert.match(await page.locator('.empty-state h2').innerText(), /모든 이미지의 라벨링을 마쳤습니다/);
    await page.locator('[data-action="end-labeling"]').click(); await ready(4);
    assert.equal(await page.locator('#labeling-only').isChecked(), false);
    assert.equal(await page.locator('[data-filter="ALL"]').getAttribute('aria-pressed'), 'true');
    checks.push('Completed labeling state survives reload, and the completion action returns to the complete dataset');

    for (const [file, hash] of originals) assert.equal(await digest(file), hash, `Original image changed: ${file}`);
    assert.deepEqual(errors, [], 'Renderer should not emit errors');
    const status = await page.evaluate(() => window.desktop.request('status'));
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, checks, measurements, modelValidation: 'not performed', images: status.project.total, counts: status.project.counts, originalsVerified: originals.size, python, errors }, null, 2));
    console.log(`Library workbench smoke passed (${checks.length} real UI workflows); screenshots: ${output}`);
  } catch (error) {
    let failureState;
    if (page && !page.isClosed()) {
      await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
      await fs.writeFile(path.join(output, 'failure.txt'), await page.locator('body').innerText().catch(() => '')).catch(() => {});
      failureState = await page.evaluate(() => ({
        layout: state.dataset.layout, layoutOpen: state.dataset.layoutOpen,
        detailsOpen: document.querySelector('#layout-controls')?.open,
        labelingOnly: state.dataset.labelingOnly, filter: state.dataset.filter,
        loading: state.dataset.loading, busy: libraryUI.busy, offset: state.dataset.offset,
        limit: state.dataset.limit, total: state.dataset.total, visibleIds: state.dataset.items.map(item => item.id),
        selection: [...state.selected], selectedDomIds: [...document.querySelectorAll('.image-card.selected')].map(card => card.dataset.imageId),
        focus: document.activeElement?.outerHTML, scrollY, viewport: { width: innerWidth, height: innerHeight },
      })).catch(() => null);
    }
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: false, checks, measurements, failureState, error: error.stack || String(error), errors }, null, 2));
    console.error('Renderer errors:', errors);
    throw error;
  } finally {
    await app.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });

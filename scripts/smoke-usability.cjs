'use strict';

// End-to-end desktop interaction checks. Requires a dev-only Playwright install.
// Set NODE_PATH to that installation and GEMMASCOPE_TEST_PYTHON to Python 3.12+
// with Pillow installed. Windows CI can use .venv/Scripts/python.exe.
const { _electron: electron } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'test-results', 'usability');
const fixtures = path.join(output, 'fixtures');
const dataDir = path.join(output, 'app-data');
const raceProject = path.join(output, 'race-project');
const raceFixtures = path.join(output, 'race-fixtures');
const command = process.platform === 'darwin' ? 'Meta' : 'Control';
const checks = [];
const syntheticChecks = [];
const errors = [];

function findPython() {
  const relative = process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python3'];
  const explicit = process.env.GEMMASCOPE_TEST_PYTHON || process.env.GEMMASCOPE_PYTHON;
  const candidates = explicit ? [explicit] : [path.join(root, '.venv', ...relative), path.join(root, '.gemmascope', 'venv', ...relative), 'python3'];
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['-c', 'import sys; from PIL import Image; sys.exit(0 if sys.version_info >= (3,12) else 1)'], { encoding: 'utf8', timeout: 15000, windowsHide: true });
    if (result.status === 0) return candidate;
  }
  throw new Error('Set GEMMASCOPE_TEST_PYTHON to Python 3.12+ with Pillow installed. No packages or models are downloaded by this test.');
}

async function prepareFixtures(python) {
  await fs.mkdir(output, { recursive: true });
  await fs.rm(dataDir, { recursive: true, force: true });
  await fs.rm(fixtures, { recursive: true, force: true });
  await fs.rm(raceProject, { recursive: true, force: true });
  await fs.rm(raceFixtures, { recursive: true, force: true });
  await fs.mkdir(fixtures, { recursive: true });
  await fs.mkdir(raceProject, { recursive: true });
  await fs.mkdir(raceFixtures, { recursive: true });
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
  await fs.copyFile(path.join(fixtures, 'TEST-ONLY-000.png'), path.join(raceFixtures, 'RACE-TEST-ONLY.png'));
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
  try {
    page = await app.firstWindow();
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });

    const card = id => page.locator(`.image-card[data-image-id="${id}"]`);
    const thumb = id => card(id).locator('.thumb-button');
    const visibleIds = () => page.locator('.image-card[data-image-id]').evaluateAll(cards => cards.map(item => item.dataset.imageId));
    const selected = async ids => {
      const expected = ids.map(String).sort();
      await page.waitForFunction(wanted => {
        const actual = [...document.querySelectorAll('.image-card.selected[data-image-id]')].map(item => item.dataset.imageId).sort();
        return JSON.stringify(actual) === JSON.stringify(wanted);
      }, expected);
    };
    const focused = async id => {
      await page.waitForFunction(wanted => document.activeElement?.matches('.thumb-button') && document.activeElement.dataset.id === wanted, String(id));
    };
    const totalIs = async (total, cards) => {
      await page.waitForFunction(({ total, cards }) =>
        Number(document.querySelector('#nav-count')?.textContent.replace(/,/g, '')) === total &&
        !document.querySelector('#content .loading-inline') &&
        (cards == null || document.querySelectorAll('.image-card[data-image-id]').length === cards), { total, cards }, { timeout: 30000 });
    };
    const dataset = (filter = 'ALL') => page.evaluate(filter => window.desktop.request('images', { filter, offset: 0, limit: 200 }), filter);
    const imageRecord = async (id, filter) => (await dataset(filter)).items.find(item => String(item.id) === String(id));
    const noModal = () => page.waitForFunction(() => !document.querySelector('#modal').open);
    const modalOpen = () => page.waitForFunction(() => document.querySelector('#modal').open);
    const assertOriginals = async () => {
      for (const [file, hash] of originals) assert.equal(await digest(file), hash, `Original file changed or missing: ${file}`);
    };

    await page.waitForSelector('[data-action="import"]');
    await page.waitForFunction(() => !document.querySelector('#device-pill').textContent.includes('확인 중'));
    await app.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, fixtures);
    await page.locator('[data-action="import"]').first().click();
    await page.locator('[data-action="confirm-import"]').click();
    await totalIs(41, 40);
    if (await page.locator('[data-action="dismiss-job"]').count()) await page.locator('[data-action="dismiss-job"]').click();
    checks.push('Native folder import creates 41 distinct synthetic images across two pages');

    const ids = await visibleIds();
    assert.equal(ids.length, 40);
    const toolbarHeight = await page.locator('#selection-tools').evaluate(element => element.getBoundingClientRect().height);
    await thumb(ids[0]).click(); await selected([ids[0]]);
    assert.equal(await page.locator('#selection-tools').evaluate(element => element.getBoundingClientRect().height), toolbarHeight);
    await thumb(ids[1]).click(); await selected([ids[1]]);
    await thumb(ids[3]).click({ modifiers: [command] }); await selected([ids[1], ids[3]]);
    await thumb(ids[1]).click();
    await thumb(ids[3]).click({ modifiers: ['Shift'] }); await selected([ids[1], ids[2], ids[3]]);
    await card(ids[0]).locator('.selection-box').click(); await selected([ids[0], ids[1], ids[2], ids[3]]);
    await card(ids[1]).locator('.selection-box').click(); await selected([ids[0], ids[2], ids[3]]);
    assert.equal(await page.locator('#selection-tools').evaluate(element => element.getBoundingClientRect().height), toolbarHeight);
    await page.screenshot({ path: path.join(output, 'selection.png') });
    checks.push('Single selection replaces, modifier click toggles, Shift selects a range, checkbox toggles independently');

    await thumb(ids[0]).click();
    await page.keyboard.press('ArrowRight'); await selected([ids[1]]); await focused(ids[1]);
    await page.keyboard.press('Space'); await selected([]); await focused(ids[1]);
    await page.keyboard.press('Space'); await selected([ids[1]]);
    await page.keyboard.press('Enter'); await modalOpen();
    assert.equal(await page.locator('#metadata-form').count(), 1);
    await page.keyboard.press('Escape'); await noModal(); await focused(ids[1]);
    checks.push('Arrow navigation, Space toggling, Enter preview, and Escape focus restoration');

    // Prepare meaningful annotations, then verify removal/undo preserves them.
    await page.keyboard.press('Enter'); await modalOpen();
    await page.locator('#metadata-group').fill('usability-specimen');
    await page.locator('#meta-sample').fill('원본 보존 확인 시료');
    await page.locator('[data-action="save-metadata"]').click(); await noModal();
    await totalIs(41, 40);
    await thumb(ids[1]).click(); await page.keyboard.press('1');
    await page.waitForFunction(id => document.querySelector(`.image-card[data-image-id="${id}"] .label-badge`)?.textContent === 'SEM', ids[1]);
    await focused(ids[1]);
    const victim = await imageRecord(ids[1]);
    assert.equal(victim.label, 'SEM');
    assert.equal(victim.group, 'usability-specimen');
    assert.equal(victim.metadata.sample, '원본 보존 확인 시료');
    await thumb(ids[1]).click();
    await page.keyboard.press('Delete');
    await totalIs(40, 40);
    assert.equal(await card(ids[1]).count(), 0);
    assert.equal((await imageRecord(ids[1], 'REMOVED')).id, victim.id);
    await assertOriginals();
    await page.screenshot({ path: path.join(output, 'removed-undo.png') });
    await page.keyboard.press(`${command}+z`);
    await totalIs(41, 40);
    const restored = await imageRecord(ids[1]);
    assert.equal(restored.label, victim.label);
    assert.equal(restored.group, victim.group);
    assert.equal(restored.metadata.sample, victim.metadata.sample);
    assert.equal((await dataset('REMOVED')).total, 0);
    checks.push('Delete removes only the dataset record; command-Z restores its label, group, and metadata; original bytes survive');

    await thumb(ids[0]).click();
    await thumb(ids[1]).click({ modifiers: [command] });
    await thumb(ids[0]).click({ button: 'right' });
    await page.locator('#context-menu').waitFor({ state: 'visible' });
    await selected([ids[0], ids[1]]);
    await page.screenshot({ path: path.join(output, 'context-menu.png') });
    await page.keyboard.press('Escape');
    await page.locator('#context-menu').waitFor({ state: 'hidden' }); await focused(ids[0]);
    await thumb(ids[3]).click({ button: 'right' }); await selected([ids[3]]);
    await page.keyboard.press('End');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.action), 'remove-images');
    await page.keyboard.press('Home');
    assert.equal(await page.evaluate(() => document.activeElement.dataset.action), 'metadata');
    await page.keyboard.press('1');
    await page.locator('#context-menu').waitFor({ state: 'hidden' });
    await page.waitForFunction(id => document.querySelector(`.image-card[data-image-id="${id}"] .label-badge`)?.textContent === 'SEM', ids[3]);
    await focused(ids[3]);
    await page.keyboard.press('Shift+F10');
    await page.locator('#context-menu').waitFor({ state: 'visible' });
    await page.keyboard.press('Delete');
    await totalIs(40, 40);
    await page.locator('[data-filter="REMOVED"]').click();
    await page.waitForFunction(() => document.querySelectorAll('.image-card[data-image-id]').length === 1 && !document.querySelector('#content .loading-inline'));
    await thumb(ids[3]).click();
    await page.locator('#selection-tools [data-action="restore-images"]').click();
    await totalIs(41, 0);
    assert.equal((await dataset('REMOVED')).total, 0);
    checks.push('Context menu preserves existing multi-selection, selects unselected targets, supports Home/End/Escape and Shift-F10, accepts numeric labeling/Delete, and restores from REMOVED');

    await page.locator('[data-filter="ALL"]').click(); await totalIs(41, 40);
    await thumb(ids[0]).click();
    const beforeIME = await imageRecord(ids[0]);
    const imeEvents = await page.evaluate(() => {
      const target = document.activeElement;
      const events = [
        new KeyboardEvent('keydown', { key: 'Delete', isComposing: true, bubbles: true, cancelable: true }),
        new KeyboardEvent('keydown', { key: '1', isComposing: true, bubbles: true, cancelable: true }),
        new KeyboardEvent('keydown', { key: 'Delete', keyCode: 229, bubbles: true, cancelable: true }),
        new KeyboardEvent('keydown', { key: '1', keyCode: 229, bubbles: true, cancelable: true }),
      ];
      events.forEach(event => target.dispatchEvent(event));
      return events.map(event => ({ composing: event.isComposing, keyCode: event.keyCode }));
    });
    assert.equal(imeEvents[0].composing, true);
    assert.equal(imeEvents[2].keyCode, 229);
    const afterIME = await imageRecord(ids[0]);
    assert.ok(afterIME, 'IME composition must not remove the focused image');
    assert.equal(afterIME.label, beforeIME.label, 'IME composition must not change image labels');
    await selected([ids[0]]);
    syntheticChecks.push('Synthetic isComposing and keyCode 229 Delete/numeric keydowns leave stored images and labels unchanged; this does not exercise an OS IME');
    await page.locator('#dataset-query').focus();
    await page.keyboard.press('Delete');
    await selected([ids[0]]);
    assert.equal((await dataset()).total, 41);
    await thumb(ids[0]).focus();
    await page.keyboard.press(`${command}+f`);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'dataset-query');
    await page.keyboard.press('F1'); await modalOpen();
    assert.equal(await page.locator('.shortcut-table').count(), 1);
    await page.screenshot({ path: path.join(output, 'shortcuts.png') });
    await page.keyboard.press('Escape'); await noModal();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'dataset-query');
    checks.push('Delete inside a text field is safe; command-F focuses search; F1 opens keyboard help and Escape restores input focus');

    await page.locator('[data-action="page"][data-scope="dataset"][data-direction="1"]').click();
    await page.waitForFunction(() => document.querySelectorAll('.image-card[data-image-id]').length === 1 && !document.querySelector('#content .loading-inline'));
    const lastId = (await visibleIds())[0];
    await thumb(lastId).click(); await page.keyboard.press('Delete');
    await totalIs(40, 40);
    assert.equal(await page.locator('[data-action="page"][data-scope="dataset"][data-direction="-1"]').isDisabled(), true);
    assert.match(await page.locator('.dataset-footer .pager').innerText(), /1[–-]40\s*\/\s*40/);
    await page.screenshot({ path: path.join(output, 'last-page-clamped.png') });
    await page.locator('.toast-action').last().click();
    await totalIs(41, 40);
    assert.equal((await dataset('REMOVED')).total, 0);
    await assertOriginals();
    checks.push('Removing the only image on the last page clamps to a populated page; toast Undo restores it without touching original files');

    const progressLayout = await page.evaluate(() => {
      const saved = { job: state.job, jobKind: state.jobKind, showLogs: state.showLogs, dismissedJob: state.dismissedJob };
      const originalFocus = captureFocus();
      try {
        state.job = { ...state.job, id: 'synthetic-layout-check', status: 'running', progress: .1, error: null, message: '짧은 진행 메시지', logs: Array.from({ length: 100 }, (_, i) => `Synthetic layout log ${i}`) };
        state.jobKind = 'import'; state.showLogs = false; state.dismissedJob = null; renderJob();
        const shortHeight = document.querySelector('.job-main').getBoundingClientRect().height;
        document.querySelector('#job-region [data-action="toggle-logs"]').focus();
        state.showLogs = true; renderJob();
        const toggleFocused = document.activeElement.dataset.action === 'toggle-logs';
        document.querySelector('.job-logs').scrollTop = 42;
        state.job = { ...state.job, progress: .9999, message: '매우_긴_파일명과_디렉터리_정보_'.repeat(40), logs: [...state.job.logs, 'Synthetic polling update'] };
        renderJob();
        return {
          shortHeight,
          longHeight: document.querySelector('.job-main').getBoundingClientRect().height,
          toggleFocused,
          pollingFocusPreserved: document.activeElement.dataset.action === 'toggle-logs',
          logScroll: document.querySelector('.job-logs').scrollTop,
        };
      } finally {
        Object.assign(state, saved); renderJob(); restoreFocus(originalFocus);
      }
    });
    assert.equal(progressLayout.longHeight, progressLayout.shortHeight);
    assert.equal(progressLayout.toggleFocused, true);
    assert.equal(progressLayout.pollingFocusPreserved, true);
    assert.equal(progressLayout.logScroll, 42);
    syntheticChecks.push('Synthetic job messages keep progress-panel height stable, retain log-button focus across render updates, and preserve a manually scrolled log position');

    const searchInvalidation = await page.evaluate(() => {
      const saved = { view: state.view, search: { ...state.search }, results: state.results, searchSnapshot: state.searchSnapshot, searchLoading: state.searchLoading };
      const originalFocus = captureFocus();
      try {
        state.view = 'search'; state.search.offset = 40;
        state.results = { items: [], total: 1, indexed: 1, excluded: 0 };
        state.searchSnapshot = { ...state.search }; state.searchLoading = false;
        render();
        const initiallyEnabled = !document.querySelector('[data-action="export-results"]').disabled;
        const input = document.querySelector('#semantic-query');
        input.value = 'synthetic changed query'; input.dispatchEvent(new Event('input', { bubbles: true }));
        return {
          initiallyEnabled,
          resultsCleared: state.results === null,
          snapshotCleared: state.searchSnapshot === null,
          offset: state.search.offset,
          exportDisabled: document.querySelector('[data-action="export-results"]').disabled,
        };
      } finally {
        Object.assign(state, saved); render(); restoreFocus(originalFocus);
      }
    });
    assert.equal(searchInvalidation.initiallyEnabled, true);
    assert.equal(searchInvalidation.resultsCleared, true);
    assert.equal(searchInvalidation.snapshotCleared, true);
    assert.equal(searchInvalidation.offset, 0);
    assert.equal(searchInvalidation.exportDisabled, true);
    syntheticChecks.push('Synthetic search results are invalidated by query edits, pagination resets, and CSV export becomes unavailable; no model inference is performed');

    // Populate a second real project with the same numeric image ID. Delay only
    // the renderer's project response, leaving the worker already switched, to
    // reproduce the dangerous interval where old selection/undo state existed.
    const originalProject = (await page.evaluate(() => window.desktop.request('status'))).project.root;
    const switchProject = async folder => {
      await app.evaluate(({ dialog }, selected) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selected] }); }, folder);
      await page.locator('[data-action="project"]').click();
      await page.waitForFunction(folder => document.querySelector('#project-location')?.title === folder && !libraryUI.switching && !libraryUI.busy, folder, { timeout: 30000 });
    };
    await switchProject(raceProject); await totalIs(0, 0);
    await app.evaluate(({ dialog }, selected) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selected] }); }, raceFixtures);
    await page.locator('[data-action="import"]').first().click();
    await page.locator('[data-action="confirm-import"]').click();
    await totalIs(1, 1);
    if (await page.locator('[data-action="dismiss-job"]').count()) await page.locator('[data-action="dismiss-job"]').click();
    const newProjectId = (await visibleIds())[0];
    await thumb(newProjectId).click(); await page.keyboard.press('2');
    await page.waitForFunction(() => document.querySelector('.image-card .label-badge')?.textContent === 'TEM');
    const newProjectBefore = await imageRecord(newProjectId);
    await switchProject(originalProject); await totalIs(41, 40);
    const oldIds = await visibleIds();
    const pendingUndoId = oldIds.at(-1);
    await thumb(pendingUndoId).click(); await page.keyboard.press('Delete');
    await totalIs(40, 40);
    assert.ok((await visibleIds()).includes(newProjectId), 'Both projects must contain the same selected numeric ID');
    await thumb(newProjectId).click(); await selected([newProjectId]);
    await page.evaluate(() => {
      const original = request;
      window.__usabilityRace = { original, gap: false, gapReached: false, attemptedMutations: [], switching: false };
      request = async function(action, payload = {}) {
        const test = window.__usabilityRace;
        if (test.gap && ['label', 'remove_images', 'restore_images'].includes(action)) test.attemptedMutations.push(action);
        const result = await original(action, payload);
        if (action === 'project') {
          test.gap = true; test.gapReached = true; test.switching = libraryUI.switching;
          const target = document.activeElement || document.body;
          for (const attributes of [
            { key: 'Delete' },
            { key: 'z', metaKey: macKeyboard, ctrlKey: !macKeyboard },
            { key: '1' },
          ]) target.dispatchEvent(new KeyboardEvent('keydown', { ...attributes, bubbles: true, cancelable: true }));
          await new Promise(resolve => setTimeout(resolve, 300));
          test.gap = false;
        }
        return result;
      };
    });
    let raceResult;
    try {
      await switchProject(raceProject); await totalIs(1, 1);
      raceResult = await page.evaluate(() => ({ gapReached: window.__usabilityRace.gapReached, switching: window.__usabilityRace.switching, attemptedMutations: window.__usabilityRace.attemptedMutations }));
      assert.equal(raceResult.gapReached, true);
      assert.equal(raceResult.switching, true);
      assert.deepEqual(raceResult.attemptedMutations, []);
      const newProjectAfter = await imageRecord(newProjectId);
      assert.equal(newProjectAfter.label, newProjectBefore.label);
      assert.equal(newProjectAfter.path, newProjectBefore.path);
      assert.equal((await dataset('REMOVED')).total, 0);
    } finally {
      await page.evaluate(() => { request = window.__usabilityRace.original; delete window.__usabilityRace; });
    }
    await switchProject(originalProject); await totalIs(40, 40);
    await page.locator('[data-filter="REMOVED"]').click();
    await page.waitForFunction(() => document.querySelectorAll('.image-card[data-image-id]').length === 1 && !document.querySelector('#content .loading-inline'));
    await thumb(pendingUndoId).click();
    await page.locator('[data-action="restore-images"]').click(); await totalIs(41, 0);
    await page.locator('[data-filter="ALL"]').click(); await totalIs(41, 40);
    await assertOriginals();
    checks.push('Two real persisted projects with colliding numeric image IDs remain independent and original project images can be restored after switching');
    syntheticChecks.push('Synthetic Delete/command-Z/numeric keydowns during a 300ms delayed project-response gap issue no mutations against the newly opened real project');

    assert.deepEqual(errors, [], 'Renderer should not emit errors');
    const status = await page.evaluate(() => window.desktop.request('status'));
    await page.screenshot({ path: path.join(output, 'final.png') });
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, checks, syntheticChecks, modelValidation: 'not performed', images: status.project.total, originalsVerified: originals.size, progressLayout, searchInvalidation, raceResult, python, errors }, null, 2));
    console.log(`Usability smoke passed (${checks.length} real UI workflows, ${syntheticChecks.length} synthetic interaction checks); screenshots: ${output}`);
  } catch (error) {
    if (page && !page.isClosed()) {
      await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
      await fs.writeFile(path.join(output, 'failure.txt'), await page.locator('body').innerText().catch(() => '')).catch(() => {});
    }
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: false, checks, syntheticChecks, error: error.stack || String(error), errors }, null, 2));
    console.error('Renderer errors:', errors);
    throw error;
  } finally {
    await app.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });

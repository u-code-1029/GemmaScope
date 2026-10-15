'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { csvCell, resultsCSV, within, validRequest } = require('../desktop/util.cjs');
test('CSV neutralizes formulas and preserves quoted, Korean and multiline paths', () => {
  assert.equal(csvCell('=HYPERLINK("file")'), '"\'=HYPERLINK(""file"")"');
  assert.equal(csvCell('정상\n"시료"'), '"정상\n""시료"""');
  assert.ok(resultsCSV([{ name: 'SEM', metadata: { sample: '시료 A' } }]).startsWith('\uFEFFname,path'));
  assert.ok(resultsCSV([{ name: 'SEM', metadata: { sample: '시료 A' } }]).includes('시료 A'));
});
test('IPC rejects unknown actions and malformed payloads', () => {
  assert.doesNotThrow(() => validRequest('search', { query: '다공성 입자' }));
  assert.throws(() => validRequest('exec', {}));
  assert.throws(() => validRequest('search', []));
});
test('path membership does not accept sibling prefixes or traversal', () => {
  assert.ok(within('/project/images/a.png', '/project/images'));
  assert.equal(within('/project/images-old/a.png', '/project/images'), false);
  assert.equal(within('/project/images/../secrets/a', '/project/images'), false);
});

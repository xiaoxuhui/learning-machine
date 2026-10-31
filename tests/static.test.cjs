const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '..');
test('offline entry uses only relative local runtime resources', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const resources = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map(m => m[1]);
  for (const resource of resources) {
    assert.ok(resource.startsWith('./'), resource);
    assert.ok(fs.existsSync(path.resolve(root, resource)), resource);
  }
});
test('build output exactly matches each runtime source file', () => {
  for (const file of ['index.html', 'src/model.js', 'src/visuals.js', 'src/app.js', 'src/styles.css', 'src/icon.svg']) {
    assert.equal(fs.readFileSync(path.join(root, 'dist', file), 'utf8'), fs.readFileSync(path.join(root, file), 'utf8'), file);
  }
});
test('all three main actions share one control group', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const group = html.match(/<div class="mode-switch"[\s\S]*?<\/div>/)[0];
  for (const id of ['reset', 'train-mode', 'predict-mode']) assert.ok(group.includes(`id="${id}"`));
  assert.equal((html.match(/id="reset"/g) || []).length, 1);
});
test('version appears consistently and server port avoids existing games', () => {
  const version = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version;
  assert.ok(fs.readFileSync(path.join(root, 'index.html'), 'utf8').includes(`v${version}`));
  assert.ok(fs.readFileSync(path.join(root, 'scripts/serve.mjs'), 'utf8').includes("listen(4319, '127.0.0.1'"));
});

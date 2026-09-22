const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(root,'manifest.json')));
assert.equal(manifest.manifest_version,3);
assert.equal(manifest.name,'LitSync IEEE Collector');
assert.deepEqual(manifest.permissions,['storage','tabs','downloads','alarms']);
const files = [manifest.background.service_worker,manifest.action.default_popup,...manifest.content_scripts.flatMap(c=>c.js),'popup/popup.css','popup/popup.js'];
for(const file of new Set(files)) assert.ok(fs.existsSync(path.join(root,file)),`Missing ${file}`);
for(const folder of ['src','popup']) for(const file of fs.readdirSync(path.join(root,folder)).filter(f=>f.endsWith('.js'))) {
  execFileSync(process.execPath,['--check',path.join(root,folder,file)],{stdio:'pipe'});
}
assert.ok(!manifest.host_permissions.includes('<all_urls>'));
assert.ok(!manifest.permissions.includes('debugger'));
console.log('Manifest references, permission scope and all runtime JavaScript syntax validated.');

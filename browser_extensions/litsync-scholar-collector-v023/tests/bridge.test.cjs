const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const Core = require('../src/core.js');

function harness(origin='http://localhost:8000') {
  const listeners = {};
  const posted = [];
  const sent = [];
  const window = {
    location: {origin},
    addEventListener(type, fn) { listeners[type] = fn; },
    postMessage(message, targetOrigin) { posted.push({message, targetOrigin}); }
  };
  const document = { addEventListener(_type, _fn, _opts) {} };
  const chrome = {runtime:{async sendMessage(message){sent.push(message); return {ok:true};}}};
  const sandbox = {globalThis:null, window, document, chrome, URL, console};
  sandbox.globalThis = sandbox;
  sandbox.LitSyncScholarCore = Core;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/bridge.js'),'utf8'),sandbox,{filename:'bridge.js'});
  return {listeners,posted,sent,window};
}

function contextMessage(overrides={}) {
  return {
    type:'LITSYNC_SCHOLAR_QUERY_CONTEXT',
    schema_version:1,
    research_question:'RQ',
    active_query_version:'balanced',
    query_fingerprint:'a'.repeat(64),
    query:'federated learning',
    ...overrides
  };
}

test('bridge announces a dedicated Scholar-ready message', () => {
  const h = harness();
  assert.ok(h.posted.some(x => x.message.type === 'LITSYNC_SCHOLAR_EXTENSION_READY'));
});

test('bridge accepts valid localhost Scholar context', async () => {
  const h = harness();
  await h.listeners.message({source:h.window, origin:'http://localhost:8000', data:contextMessage()});
  assert.equal(h.sent.length,1);
  assert.equal(h.sent[0].type,'LSSC_SAVE_CONTEXT');
  assert.equal(h.sent[0].context.query,'federated learning');
});

test('bridge rejects malformed context and foreign origin', async () => {
  const h = harness();
  await h.listeners.message({source:h.window, origin:'http://localhost:8000', data:contextMessage({query:''})});
  await h.listeners.message({source:h.window, origin:'https://evil.example', data:contextMessage()});
  assert.equal(h.sent.length,0);
});

test('bridge clears Scholar context when LitSync invalidates queries', async () => {
  const h = harness('http://127.0.0.1:8000');
  await h.listeners.message({source:h.window, origin:'http://127.0.0.1:8000', data:{type:'LITSYNC_QUERY_CONTEXT_CLEAR'}});
  assert.equal(h.sent[0].type,'LSSC_CLEAR_CONTEXT');
});

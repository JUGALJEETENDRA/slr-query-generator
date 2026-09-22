const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const Core = require('../src/core.js');

function makeHarness(options = {}) {
  const store = {};
  const tabs = new Map();
  let nextTabId = 1;
  const runtimeListeners = [];
  const onCreatedListeners = [];
  const onChangedListeners = [];
  const sentMessages = [];
  const downloadRequests = [];
  const fetchUrls = [];
  let nextDownloadId = 1000;
  let fetchText = options.fetchText == null ? '' : String(options.fetchText);
  let fetchError = options.fetchError || null;
  let fileSchemeAccess = options.fileSchemeAccess !== false;

  const chrome = {
    storage: {
      local: {
        async get(keys) {
          if (keys == null) return {...store};
          const arr = Array.isArray(keys) ? keys : [keys];
          const out = {};
          for (const key of arr) if (Object.prototype.hasOwnProperty.call(store, key)) out[key] = store[key];
          return out;
        },
        async set(obj) { Object.assign(store, obj); },
        async remove(keys) {
          for (const key of Array.isArray(keys) ? keys : [keys]) delete store[key];
        }
      }
    },
    runtime: {
      onMessage: { addListener(fn) { runtimeListeners.push(fn); } }
    },
    extension: {
      isAllowedFileSchemeAccess(callback) {
        const allowed = fileSchemeAccess;
        if (typeof callback === 'function') callback(allowed);
        return undefined;
      }
    },
    tabs: {
      async create(opts) {
        const tab = {id: nextTabId++, url: opts.url, active: !!opts.active};
        tabs.set(tab.id, tab);
        return tab;
      },
      async get(id) {
        if (!tabs.has(id)) throw new Error('No tab');
        return tabs.get(id);
      },
      async update(id, patch) {
        const tab = await this.get(id);
        Object.assign(tab, patch);
        return tab;
      },
      async sendMessage(id, message) {
        sentMessages.push({id, message});
        if (options.sendMessageError) throw options.sendMessageError;
        return {ok: true};
      },
      async reload(id) {
        const tab = await this.get(id);
        tab.reloadCount = Number(tab.reloadCount || 0) + 1;
        return undefined;
      }
    },
    downloads: {
      onCreated: { addListener(fn) { onCreatedListeners.push(fn); } },
      onChanged: { addListener(fn) { onChangedListeners.push(fn); } },
      async download(opts) {
        const id = nextDownloadId++;
        downloadRequests.push({id, ...opts});
        return id;
      },
      async search(query) {
        const run = store[Core.STORAGE_KEYS.run] || {};
        let filename = String(options.downloadedFilename || run.downloadedFilename || 'citations.csv');
        if (!/^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(filename)) filename = `C:\\Users\\Harshil\\Downloads\\${filename}`;
        return [{id:query.id, filename, state:'complete', exists:true, fileSize:Math.max(1, fetchText.length)}];
      }
    }
  };

  const sandbox = {
    console,
    chrome,
    setTimeout,
    clearTimeout,
    Date,
    Promise,
    URL,
    async fetch(url) {
      fetchUrls.push(String(url || ''));
      if (fetchError) throw fetchError;
      return {ok: true, status: 200, async text() { return fetchText; }};
    },
    globalThis: null,
    importScripts() {}
  };
  sandbox.globalThis = sandbox;
  sandbox.LitSyncScholarCore = Core;
  vm.createContext(sandbox);
  const code = fs.readFileSync(path.join(__dirname, '../src/background.js'), 'utf8');
  vm.runInContext(code, sandbox, {filename: 'background.js'});

  async function send(message, sender = {}) {
    assert.equal(runtimeListeners.length, 1);
    return new Promise((resolve, reject) => {
      let syncReturned;
      try {
        syncReturned = runtimeListeners[0](message, sender, response => resolve(response));
      } catch (e) { reject(e); }
      assert.equal(syncReturned, true);
    });
  }

  return {
    store, tabs, sentMessages, onCreatedListeners, onChangedListeners, downloadRequests, fetchUrls, send,
    setFetchText(value) { fetchText = String(value); },
    setFetchError(value) { fetchError = value; },
    setFileSchemeAccess(value) { fileSchemeAccess = Boolean(value); }
  };
}

function scholarCsv(count, options = {}) {
  const rows = ['Authors,Title,Publication,Volume,Number,Pages,Year,Publisher'];
  for (let i = 1; i <= count; i += 1) {
    const title = options.quotedNewlineAt === i ? `"Paper ${i} line 1\nline 2"` : `"Paper ${i}"`;
    rows.push(`"Author ${i}",${title},"Journal",1,1,"1-2",2026,"Publisher"`);
  }
  return rows.join('\r\n') + '\r\n';
}


function v3LibraryMap(ids) {
  return Object.fromEntries(ids.map(id => [id, id]));
}

function goodContext() {
  return {
    schema_version: 1,
    research_question: 'How can federated learning improve intrusion detection in IoT networks?',
    active_query_version: 'balanced',
    query_fingerprint: 'a'.repeat(64),
    query: '("federated learning" OR "federated machine learning") AND ("intrusion detection" OR IDS)'
  };
}

test('background stores context and starts a durable Scholar run', async () => {
  const h = makeHarness();
  assert.equal((await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()})).ok, true);
  const started = await h.send({type:'LSSC_START'});
  assert.equal(started.ok, true);
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.RUNNING);
  assert.equal(run.stage, Core.STAGE.SEARCH_LOADING);
  assert.equal(run.tabId, 1);
  assert.equal(run.query, goodContext().query);
  assert.match(run.label, /^litsync_/);
  assert.equal(h.tabs.get(1).url, Core.buildSearchUrl(goodContext().query));
  assert.equal(run.lastUrl, Core.buildSearchUrl(goodContext().query));
  assert.ok(h.sentMessages.some(x => x.id === 1 && x.message.type === 'LSSC_WAKE'));
});

test('owner-tab guard rejects state mutation from another Scholar tab', async () => {
  const h = makeHarness();
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const bad = await h.send({type:'LSSC_PATCH_RUN', patch:{message:'bad'}}, {tab:{id:999}});
  assert.equal(bad.ok, false);
  assert.match(bad.error, /does not own/i);
});

test('pause and resume preserve checkpointed saved IDs', async () => {
  const h = makeHarness();
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  await h.send({type:'LSSC_PATCH_RUN', patch:{savedIds:['1','2','3'], targetTotal:500, stage:Core.STAGE.COLLECTING}}, {tab:{id:1}});
  await h.send({type:'LSSC_PAUSE'});
  assert.equal(h.store[Core.STORAGE_KEYS.run].status, Core.STATUS.PAUSED);
  await h.send({type:'LSSC_RESUME'});
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.RUNNING);
  assert.deepEqual(Array.from(run.savedIds), ['1','2','3']);
  assert.equal(run.stage, Core.STAGE.COLLECTING);
});

test('resume reopens a missing tab without resetting progress', async () => {
  const h = makeHarness();
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  await h.send({type:'LSSC_PATCH_RUN', patch:{savedIds:['1'], targetTotal:10, lastUrl:'https://scholar.google.com/scholar?q=x&start=10'}}, {tab:{id:1}});
  h.tabs.delete(1);
  await h.send({type:'LSSC_PAUSE'});
  await h.send({type:'LSSC_RESUME'});
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.tabId, 2);
  assert.equal(h.tabs.get(2).url, 'https://scholar.google.com/scholar?q=x&start=10');
  assert.deepEqual(Array.from(run.savedIds), ['1']);
});

test('CSV download completion requires exact CSV content verification', async () => {
  const h = makeHarness({fetchText: scholarCsv(23)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length: 23}, (_, i) => String(i+1));
  await h.send({type:'LSSC_PATCH_RUN', patch:{savedIds:ids, targetTotal:23, stage:Core.STAGE.LABEL_LOADING, integrityVersion:3, libraryIdsBySearchId:v3LibraryMap(ids), labelAuditComplete:true, labelVerifiedIds:ids, labelVerifiedCount:23}}, {tab:{id:1}});
  const trig = await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  assert.equal(trig.ok, true);
  assert.equal(h.store[Core.STORAGE_KEYS.run].stage, Core.STAGE.EXPORTING);
  assert.equal(h.onCreatedListeners.length, 1);
  await h.onCreatedListeners[0]({id:77,url:'https://scholar.google.com/citations?format=csv',filename:'citations.csv'});
  assert.equal(h.store[Core.STORAGE_KEYS.run].exportDownloadId, 77);
  await h.onChangedListeners[0]({id:77,state:{current:'complete'}});
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.COMPLETE);
  assert.ok(h.fetchUrls.some(url => url.startsWith('file:///C:/Users/Harshil/Downloads/')), 'native download must be verified from the exact local file bytes');
  assert.equal(h.fetchUrls.some(url => url.startsWith('https://scholar.google.com/citations?format=csv')), false, 'completed native export URL must not be fetched a second time');
  assert.equal(run.stage, Core.STAGE.DONE);
  assert.match(run.message, /23\/23/);
});

test('v0.1.2 resume is backward-compatible with an in-progress v0.1.1 checkpoint', async () => {
  const h = makeHarness();
  const oldRun = Core.createRun(goodContext(), {runId:'old-v011'});
  oldRun.tabId = 1;
  oldRun.stage = Core.STAGE.COLLECTING;
  oldRun.status = Core.STATUS.NEEDS_ATTENTION;
  oldRun.targetTotal = 416;
  oldRun.reportedTotal = 416;
  oldRun.savedIds = Array.from({length:159}, (_, i) => `cid-${i+1}`);
  delete oldRun.consecutiveFailures;
  delete oldRun.cooldownCount;
  delete oldRun.retryStartedAt;
  h.store[Core.STORAGE_KEYS.run] = oldRun;
  h.tabs.set(1, {id:1, url:'https://scholar.google.com/scholar?q=x&start=150', active:true});
  const response = await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok, true);
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.RUNNING);
  assert.equal(run.savedIds.length, 159);
  assert.equal(run.targetTotal, 416);
  assert.equal(run.stage, Core.STAGE.COLLECTING);
});

test('partial CSV download is validated but marked INCOMPLETE and never COMPLETE', async () => {
  const h = makeHarness({fetchText: scholarCsv(358)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length: 358}, (_, i) => `cid-${i+1}`);
  const unresolved = {};
  for (let i = 359; i <= 405; i += 1) unresolved[`cid-${i}`] = {finalized:true, error:'save failed'};
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    targetTotal:405,
    accessibleTotal:405,
    unresolved,
    exportMode:'partial',
    stage:Core.STAGE.LABEL_LOADING,
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:358
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:88,url:'https://scholar.google.com/citations?format=csv',filename:'citations.csv'});
  await h.onChangedListeners[0]({id:88,state:{current:'complete'}});
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.INCOMPLETE);
  assert.equal(run.stage, Core.STAGE.DONE);
  assert.match(run.message, /358\/405/);
  assert.match(run.message, /unresolved=47/);
});

test('Resume from INCOMPLETE export retries only unresolved IDs', async () => {
  const h = makeHarness();
  const run = Core.createRun(goodContext(), {runId:'incomplete-run'});
  run.tabId = 1;
  run.status = Core.STATUS.INCOMPLETE;
  run.stage = Core.STAGE.DONE;
  run.targetTotal = 405;
  run.savedIds = Array.from({length:358}, (_, i) => `cid-${i+1}`);
  run.unresolved = {
    'cid-359': {finalized:true, pageUrl:'https://scholar.google.com/scholar?q=x&start=350'},
    'cid-360': {finalized:true, pageUrl:'https://scholar.google.com/scholar?q=x&start=350'}
  };
  h.store[Core.STORAGE_KEYS.run] = run;
  h.tabs.set(1, {id:1,url:'https://scholar.google.com/scholar?q=x&start=350',active:true});
  const response = await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok, true);
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status, Core.STATUS.RUNNING);
  assert.equal(updated.stage, Core.STAGE.RETRYING_FAILURES);
  assert.equal(updated.unresolved['cid-359'].finalized, false);
  assert.equal(updated.unresolved['cid-360'].finalized, false);
  assert.equal(updated.savedIds.length, 358);
  assert.equal(updated.retryRound, 1);
});

test('v0.1.3 resume preserves the real 358 saved plus 47 unresolved checkpoint shape', async () => {
  const h = makeHarness();
  const run = Core.createRun(goodContext(), {runId:'real-shape'});
  run.tabId = 1;
  run.status = Core.STATUS.NEEDS_ATTENTION;
  run.stage = Core.STAGE.COLLECTING;
  run.reportedTotal = 416;
  run.targetTotal = 416;
  run.savedIds = Array.from({length:358}, (_, i) => `saved-${i+1}`);
  run.discoveredIds = [
    ...run.savedIds,
    ...Array.from({length:47}, (_, i) => `failed-${i+1}`)
  ];
  run.unresolved = Object.fromEntries(Array.from({length:47}, (_, i) => [
    `failed-${i+1}`,
    {finalized:false, transient:true, skippedDuringTraversal:true, pageUrl:'https://scholar.google.com/scholar?q=x&start=400'}
  ]));
  h.store[Core.STORAGE_KEYS.run] = run;
  h.tabs.set(1, {id:1,url:'https://scholar.google.com/scholar?q=x&start=400',active:true});
  await h.send({type:'LSSC_RESUME'});
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status, Core.STATUS.RUNNING);
  assert.equal(updated.stage, Core.STAGE.COLLECTING);
  assert.equal(updated.savedIds.length, 358);
  assert.equal(Core.unresolvedIds(updated).length, 47);
  assert.equal(updated.discoveredIds.length, 405);
});

test('Resume rejects a rendered BibTeX/googleusercontent page and returns to My Library without losing 405 saved IDs', async () => {
  const h = makeHarness({fetchText:'@article{wrong, title={BibTeX}}'});
  const run = Core.createRun(goodContext(), {runId:'export-page-resume'});
  run.tabId = 1;
  run.status = Core.STATUS.NEEDS_ATTENTION;
  run.stage = Core.STAGE.EXPORTING;
  run.targetTotal = 405;
  run.savedIds = Array.from({length:405}, (_,i)=>`cid-${i+1}`);
  run.unresolved = {};
  run.exportStartedAt = Date.now() - 20000;
  h.store[Core.STORAGE_KEYS.run] = run;
  const url='https://scholar.googleusercontent.com/citations?view_op=export_citations&user=x';
  h.tabs.set(1, {id:1,url,active:true});
  const response = await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok, true);
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status, Core.STATUS.RUNNING);
  assert.equal(updated.stage, Core.STAGE.OPENING_LIBRARY);
  assert.equal(updated.savedIds.length, 405);
  assert.equal(updated.targetTotal, 405);
  assert.match(updated.exportCsvVerificationError, /BibTeX/i);
  assert.equal(h.tabs.get(1).url, 'https://scholar.google.com/scholar?scilib=1&hl=en');
});

test('Resume validates an exact rendered CSV page and saves a verified CSV copy', async () => {
  const h = makeHarness({fetchText:scholarCsv(2)});
  const run = Core.createRun(goodContext(), {runId:'rendered-csv-resume'});
  run.tabId = 1;
  run.status = Core.STATUS.NEEDS_ATTENTION;
  run.stage = Core.STAGE.EXPORTING;
  run.targetTotal = 2;
  run.savedIds = ['a','b'];
  run.integrityVersion = 3;
  run.libraryIdsBySearchId = v3LibraryMap(run.savedIds);
  run.labelAuditComplete = true;
  run.labelVerifiedIds = ['a','b'];
  run.labelVerifiedCount = 2;
  run.unresolved = {};
  run.exportStartedAt = Date.now() - 20000;
  h.store[Core.STORAGE_KEYS.run] = run;
  const url='https://scholar.googleusercontent.com/citations?view_op=export_citations&user=x';
  h.tabs.set(1, {id:1,url,active:true});
  const response = await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok, true);
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.stage, Core.STAGE.EXPORTING);
  assert.equal(updated.exportCsvVerified, true);
  assert.equal(updated.exportCsvRowCount, 2);
  assert.equal(h.downloadRequests.length, 1);
  assert.match(h.downloadRequests[0].filename, /scholar_litsync_.*\.csv$/);
});

test('download content, not filename or internal format code, decides whether Scholar export is CSV', async () => {
  const h = makeHarness({fetchText:'@article{wrong, title={BibTeX despite csv filename}}'});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids=['1'];
  await h.send({type:'LSSC_PATCH_RUN', patch:{savedIds:ids, targetTotal:1, stage:Core.STAGE.LABEL_LOADING, integrityVersion:3, libraryIdsBySearchId:v3LibraryMap(ids), labelAuditComplete:true, labelVerifiedIds:ids, labelVerifiedCount:1}}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:91,url:'https://scholar.googleusercontent.com/citations?view_op=export_citations',filename:'citations.csv',mime:'text/csv'});
  assert.equal(h.store[Core.STORAGE_KEYS.run].exportDownloadId, 91);
  await h.onChangedListeners[0]({id:91,state:{current:'complete'}});
  const run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.INCOMPLETE);
  assert.equal(run.exportCsvVerified, false);
  assert.match(run.exportCsvVerificationError, /BibTeX/i);
});


test('legacy v0.1.6 COMPLETE checkpoint is re-audited instead of trusted', async () => {
  const h = makeHarness();
  const run = Core.createRun(goodContext(), {runId:'legacy-complete'});
  run.tabId = 1;
  run.status = Core.STATUS.COMPLETE;
  run.stage = Core.STAGE.DONE;
  run.targetTotal = 405;
  run.savedIds = Array.from({length:405}, (_, i) => `cid-${i+1}`);
  delete run.labelAuditComplete;
  delete run.labelVerifiedIds;
  delete run.labelVerifiedCount;
  h.store[Core.STORAGE_KEYS.run] = run;
  h.tabs.set(1, {id:1,url:'https://scholar.google.com/scholar?scilib=1026&hl=en',active:true});
  const response = await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok, true);
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status, Core.STATUS.RUNNING);
  assert.equal(updated.stage, Core.STAGE.OPENING_LIBRARY);
  assert.equal(updated.savedIds.length, 405);
  assert.equal(updated.labelAuditComplete, false);
  assert.deepEqual(Array.from(updated.labelVerifiedIds), []);
  assert.match(updated.message, /Revalidating the actual Scholar label/i);
});

test('CSV download cannot mark COMPLETE when only 341 of 405 IDs are verified in the label', async () => {
  const h = makeHarness({fetchText: scholarCsv(405)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:405}, (_, i) => `cid-${i+1}`);
  const verified = ids.slice(0,341);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    targetTotal:405,
    stage:Core.STAGE.LABEL_LOADING,
    exportMode:'full',
    labelAuditComplete:true,
    labelVerifiedIds:verified,
    labelVerifiedCount:341
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:99,url:'https://scholar.googleusercontent.com/citations?view_op=export_citations',filename:'citations.csv',mime:'text/csv'});
  await h.onChangedListeners[0]({id:99,state:{current:'complete'}});
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.INCOMPLETE);
  assert.equal(run.stage, Core.STAGE.DONE);
  assert.match(run.message, /verified-in-label=341\/405/);
});

test('Resume from integrity-INCOMPLETE reopens My Library for a fresh label audit', async () => {
  const h = makeHarness();
  const run = Core.createRun(goodContext(), {runId:'integrity-incomplete'});
  run.tabId = 1;
  run.status = Core.STATUS.INCOMPLETE;
  run.stage = Core.STAGE.DONE;
  run.targetTotal = 405;
  run.savedIds = Array.from({length:405}, (_, i) => `cid-${i+1}`);
  run.labelAuditComplete = true;
  run.labelVerifiedIds = run.savedIds.slice(0,341);
  run.labelVerifiedCount = 341;
  h.store[Core.STORAGE_KEYS.run] = run;
  h.tabs.set(1, {id:1,url:'https://scholar.google.com/scholar?scilib=1026&hl=en',active:true});
  await h.send({type:'LSSC_RESUME'});
  const updated = h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status, Core.STATUS.RUNNING);
  assert.equal(updated.stage, Core.STAGE.OPENING_LIBRARY);
  assert.deepEqual(Array.from(updated.labelVerifiedIds), []);
  assert.match(updated.message, /Re-auditing the actual LitSync label/i);
});

test('exact 405-ID label can never COMPLETE from a 341-row downloaded CSV', async () => {
  const h = makeHarness({fetchText: scholarCsv(341)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:405}, (_, i) => `cid-${i+1}`);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    targetTotal:405,
    stage:Core.STAGE.LABEL_LOADING,
    exportMode:'full',
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:405
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:1405,url:'https://scholar.googleusercontent.com/citations?view_op=export_citations',filename:'citations.csv',mime:'text/csv'});
  await h.onChangedListeners[0]({id:1405,state:{current:'complete'}});
  const run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.INCOMPLETE);
  assert.equal(run.exportCsvVerified, false);
  assert.equal(run.exportCsvRowCount, 341);
  assert.equal(run.exportCsvExpectedRows, 405);
  assert.match(run.message, /341.*405/);
});

test('verification fetch failure never produces false COMPLETE', async () => {
  const h = makeHarness({fetchError:new Error('network verification failed')});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids=['1','2'];
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,integrityVersion:3,libraryIdsBySearchId:v3LibraryMap(ids),targetTotal:2,stage:Core.STAGE.LABEL_LOADING,exportMode:'full',
    labelAuditComplete:true,labelVerifiedIds:ids,labelVerifiedCount:2
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:2002,url:'https://scholar.googleusercontent.com/citations?view_op=export_citations',filename:'citations.csv',mime:'text/csv'});
  await h.onChangedListeners[0]({id:2002,state:{current:'complete'}});
  const run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.NEEDS_ATTENTION);
  assert.equal(run.exportCsvVerified, false);
  assert.match(run.message, /exact downloaded file could not be read/i);
});

test('Resume after a bad-row-count export reuses the verified label and retries export without recollecting', async () => {
  const h = makeHarness();
  const run = Core.createRun(goodContext(), {runId:'bad-csv-resume'});
  run.tabId=1;
  run.status=Core.STATUS.INCOMPLETE;
  run.stage=Core.STAGE.DONE;
  run.targetTotal=405;
  run.savedIds=Array.from({length:405},(_,i)=>`cid-${i+1}`);
  run.integrityVersion=3;
  run.libraryIdsBySearchId=v3LibraryMap(run.savedIds);
  run.labelAuditComplete=true;
  run.labelVerifiedIds=[...run.savedIds];
  run.labelVerifiedCount=405;
  run.unresolved={};
  run.exportCsvVerified=false;
  run.exportCsvRowCount=341;
  run.exportCsvExpectedRows=405;
  h.store[Core.STORAGE_KEYS.run]=run;
  h.tabs.set(1,{id:1,url:'https://scholar.google.com/scholar?scilib=1026&hl=en',active:true});
  await h.send({type:'LSSC_RESUME'});
  const updated=h.store[Core.STORAGE_KEYS.run];
  assert.equal(updated.status,Core.STATUS.RUNNING);
  assert.equal(updated.stage,Core.STAGE.OPENING_LIBRARY);
  assert.equal(updated.savedIds.length,405);
  assert.equal(updated.labelVerifiedIds.length,405);
  assert.match(updated.message,/label is intact/i);
});


test('Resume safely reloads the Scholar tab when a new extension build has no live content script', async () => {
  const h = makeHarness({sendMessageError:new Error('Receiving end does not exist')});
  const run = Core.createRun(goodContext(), {runId:'fresh-build-wake'});
  run.tabId=1;
  run.status=Core.STATUS.RUNNING;
  run.stage=Core.STAGE.COLLECTING;
  run.targetTotal=10;
  run.savedIds=['a','b'];
  h.store[Core.STORAGE_KEYS.run]=run;
  h.tabs.set(1,{id:1,url:'https://scholar.google.com/scholar?q=x',active:true});
  const response=await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok,true);
  assert.equal(h.tabs.get(1).reloadCount,1);
  assert.deepEqual(Array.from(h.store[Core.STORAGE_KEYS.run].savedIds),['a','b']);
});

test('Start cannot overwrite an existing checkpointed run; Resume or Reset is required', async () => {
  const h = makeHarness();
  h.store[Core.STORAGE_KEYS.context]=goodContext();
  const run=Core.createRun(goodContext(),{runId:'protect-405'});
  run.tabId=1;
  run.status=Core.STATUS.NEEDS_ATTENTION;
  run.stage=Core.STAGE.EXPORTING;
  run.targetTotal=405;
  run.savedIds=Array.from({length:405},(_,i)=>`cid-${i+1}`);
  h.store[Core.STORAGE_KEYS.run]=run;
  h.tabs.set(1,{id:1,url:'https://scholar.google.com/scholar?scilib=1&hl=en',active:true});
  const response=await h.send({type:'LSSC_START'});
  assert.equal(response.ok,true);
  assert.equal(response.reused,true);
  assert.equal(h.store[Core.STORAGE_KEYS.run].runId,'protect-405');
  assert.equal(h.store[Core.STORAGE_KEYS.run].savedIds.length,405);
});

test('v0.2.2 keeps the pre-download verifier as an optional fast path', async () => {
  const h = makeHarness({fetchText: scholarCsv(60)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:60}, (_, i) => `cid-${i+1}`);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    targetTotal:60,
    stage:Core.STAGE.EXPORTING,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:60,
    exportMode:'full'
  }}, {tab:{id:1}});

  const result = await h.send({
    type:'LSSC_VERIFY_EXPORT_REQUEST',
    request:{
      url:'https://scholar.googleusercontent.com/citations?view_op=export_citations&token=once',
      method:'GET',
      body:''
    }
  }, {tab:{id:1}});

  assert.equal(result.ok, true);
  assert.equal(result.captured, true);
  assert.equal(h.downloadRequests.length, 1, 'only the locally validated CSV should be downloaded');
  assert.match(h.downloadRequests[0].url, /^data:text\/csv/);
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.exportCsvVerified, true);
  assert.equal(run.exportCsvRowCount, 60);
  assert.equal(run.exportCsvExpectedRows, 60);
  assert.equal(run.exportDownloadId, h.downloadRequests[0].id);

  await h.onChangedListeners[0]({id:run.exportDownloadId,state:{current:'complete'}});
  const done = h.store[Core.STORAGE_KEYS.run];
  assert.equal(done.status, Core.STATUS.COMPLETE);
  assert.match(done.message, /60\/60/);
});

test('v0.2.2 pre-download verifier refuses a short CSV and falls back without corrupting the successful label checkpoint', async () => {
  const h = makeHarness({fetchText: scholarCsv(59)});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:60}, (_, i) => `cid-${i+1}`);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    targetTotal:60,
    stage:Core.STAGE.EXPORTING,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:60,
    exportMode:'full'
  }}, {tab:{id:1}});

  const result = await h.send({
    type:'LSSC_VERIFY_EXPORT_REQUEST',
    request:{url:'https://scholar.googleusercontent.com/citations?view_op=export_citations&token=once',method:'GET'}
  }, {tab:{id:1}});

  assert.equal(result.ok, false);
  assert.equal(result.fallback, true);
  assert.equal(h.downloadRequests.length, 0);
  const run = h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.savedIds.length, 60);
  assert.equal(run.labelVerifiedIds.length, 60);
  assert.equal(run.status, Core.STATUS.RUNNING);
});

test('v0.2.2 pre-download verifier fetch failure preserves native export fallback and never claims completion', async () => {
  const h = makeHarness({fetchError:new Error('temporary request failed')});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:5}, (_, i) => `cid-${i+1}`);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    targetTotal:5,
    stage:Core.STAGE.EXPORTING,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:5,
    exportMode:'full'
  }}, {tab:{id:1}});
  const result = await h.send({
    type:'LSSC_VERIFY_EXPORT_REQUEST',
    request:{url:'https://scholar.googleusercontent.com/citations?view_op=export_citations&token=once',method:'GET'}
  }, {tab:{id:1}});
  assert.equal(result.ok, false);
  assert.equal(result.fallback, true);
  assert.equal(h.store[Core.STORAGE_KEYS.run].status, Core.STATUS.RUNNING);
  assert.notEqual(h.store[Core.STORAGE_KEYS.run].status, Core.STATUS.COMPLETE);
});


test('v0.2.2 58-row real-run shape completes by reading the downloaded file, not the one-use Scholar URL', async () => {
  const h = makeHarness({fetchText: scholarCsv(58), downloadedFilename:'C:\\Users\\Harshil\\Downloads\\citations (5).csv'});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids = Array.from({length:58}, (_, i) => `cid-${i+1}`);
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,
    targetTotal:58,
    stage:Core.STAGE.LABEL_LOADING,
    integrityVersion:3,
    libraryIdsBySearchId:v3LibraryMap(ids),
    labelAuditComplete:true,
    labelVerifiedIds:ids,
    labelVerifiedCount:58,
    exportMode:'full'
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  const oneUse='https://scholar.googleusercontent.com/citations?view_op=export_citations&token=already-consumed';
  await h.onCreatedListeners[0]({id:5800,url:oneUse,finalUrl:oneUse,filename:'C:\\Users\\Harshil\\Downloads\\citations (5).csv',mime:'text/csv'});
  await h.onChangedListeners[0]({id:5800,state:{current:'complete'}});
  const run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.COMPLETE);
  assert.equal(run.exportCsvVerified, true);
  assert.equal(run.exportCsvRowCount, 58);
  assert.match(run.message, /58\/58/);
  assert.ok(h.fetchUrls.some(url => /^file:\/\/\/C:\/Users\/Harshil\/Downloads\/citations%20\(5\)\.csv$/.test(url)));
  assert.equal(h.fetchUrls.includes(oneUse), false);
});

test('v0.2.2 file-access denial stops safely and Resume verifies the same existing download after access is enabled', async () => {
  const h = makeHarness({fetchText: scholarCsv(2), fileSchemeAccess:false});
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const ids=['a','b'];
  await h.send({type:'LSSC_PATCH_RUN', patch:{
    savedIds:ids,targetTotal:2,stage:Core.STAGE.LABEL_LOADING,integrityVersion:3,libraryIdsBySearchId:v3LibraryMap(ids),
    labelAuditComplete:true,labelVerifiedIds:ids,labelVerifiedCount:2,exportMode:'full'
  }}, {tab:{id:1}});
  await h.send({type:'LSSC_EXPORT_TRIGGERED'}, {tab:{id:1}});
  await h.onCreatedListeners[0]({id:2200,url:'https://scholar.googleusercontent.com/citations?view_op=export_citations',filename:'C:\\Users\\Harshil\\Downloads\\citations.csv',mime:'text/csv'});
  await h.onChangedListeners[0]({id:2200,state:{current:'complete'}});
  let run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.NEEDS_ATTENTION);
  assert.equal(run.exportDownloadId, 2200);
  assert.match(run.message, /Allow access to file URLs/i);
  assert.equal(run.savedIds.length,2);
  assert.equal(run.labelVerifiedIds.length,2);

  h.setFileSchemeAccess(true);
  const resumed = await h.send({type:'LSSC_RESUME'});
  assert.equal(resumed.ok, true);
  assert.equal(resumed.verifiedExistingDownload, true);
  run=h.store[Core.STORAGE_KEYS.run];
  assert.equal(run.status, Core.STATUS.COMPLETE);
  assert.equal(run.exportCsvVerified, true);
  assert.equal(run.exportCsvRowCount, 2);
  assert.equal(run.savedIds.length, 2);
  assert.equal(run.labelVerifiedIds.length, 2);
});

test('Reset followed by Start creates a fresh zero-count run with a new dedicated label', async () => {
  const h = makeHarness();
  await h.send({type:'LSSC_SAVE_CONTEXT', context: goodContext()});
  await h.send({type:'LSSC_START'});
  const first = h.store[Core.STORAGE_KEYS.run];
  await h.send({type:'LSSC_PATCH_RUN', patch:{savedIds:['old-1','old-2'], targetTotal:2, stage:Core.STAGE.COLLECTING}}, {tab:{id:1}});
  await h.send({type:'LSSC_RESET'});
  assert.equal(h.store[Core.STORAGE_KEYS.run], undefined);
  // Ensure a later millisecond even on very fast test machines.
  await new Promise(resolve => setTimeout(resolve, 2));
  await h.send({type:'LSSC_START'});
  const second = h.store[Core.STORAGE_KEYS.run];
  assert.notEqual(second.runId, first.runId);
  assert.notEqual(second.label, first.label);
  assert.deepEqual(Array.from(second.savedIds), []);
  assert.equal(second.targetTotal, null);
  assert.equal(second.status, Core.STATUS.RUNNING);
});

test('58/58 NEEDS_ATTENTION Resume preserves checkpoint and restarts label opening only', async () => {
  const h=makeHarness();
  const savedIds=Array.from({length:58},(_,i)=>`cid-${i+1}`);
  const run={...Core.createRun(goodContext()),status:Core.STATUS.NEEDS_ATTENTION,
    stage:Core.STAGE.OPENING_LABEL,tabId:1,targetTotal:58,scholarTotal:58,savedIds,
    label:'litsync_20260905_d2ee15f2_141237814',pagesVisited:[0,10,20,30,40,50],unresolved:{},
    libraryIdsBySearchId:v3LibraryMap(savedIds)};
  h.store[Core.STORAGE_KEYS.run]=run;
  h.tabs.set(1,{id:1,url:'https://scholar.google.com/scholar?scilib=1'});
  const before=JSON.stringify(run);
  const response=await h.send({type:'LSSC_RESUME'});
  assert.equal(response.ok,true);
  const after=h.store[Core.STORAGE_KEYS.run];
  for(const key of ['savedIds','libraryIdsBySearchId','pagesVisited','unresolved','targetTotal','label','runId']) {
    assert.deepEqual(after[key],JSON.parse(before)[key]);
  }
  assert.equal(after.stage,Core.STAGE.OPENING_LABEL);
  assert.equal(after.status,Core.STATUS.RUNNING);
  assert.equal(h.downloadRequests.length,0);
});

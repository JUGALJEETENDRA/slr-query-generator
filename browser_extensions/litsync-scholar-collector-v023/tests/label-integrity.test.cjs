const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const Core = require('../src/core.js');

function makeRow(spec) {
  const value = typeof spec === 'string' ? {cid:spec} : spec;
  return {
    getAttribute(name) {
      if (name === 'data-cid') return value.cid || null;
      if (name === 'data-lid') return value.lid || null;
      return null;
    },
    querySelector() { return null; }
  };
}

function makeHarness({run, rows=[], anchors=[], url='https://scholar.google.com/scholar?scilib=1026&hl=en&start=340', nextHref=null, timed=false, renderAfter=0, onWait=null}) {
  let clock = 0;
  const rowNodes = rows.map(makeRow);
  const anchorNodes = anchors.map((spec, index) => ({
    tagName: 'A',
    textContent: spec.text || '',
    href: spec.href || `https://scholar.google.com/scholar?scilib=${999+index}&hl=en`,
    attributes: [{name:'data-label',value:spec.text || ''}],
    className: spec.className || '',
    children: spec.children || [],
    querySelectorAll(){ return spec.children || []; },
    parentElement: spec.parent || null,
    getAttribute(name) {
      if (name === 'href') return this.href;
      if (name === 'title') return spec.title || null;
      if (name === 'aria-label') return spec.aria || null;
      if (name === 'role') return spec.role || null;
      return null;
    },
    scrollIntoView(){},
    click(){}
  }));
  const logs = [];
  const location = {href:url};
  const next = nextHref ? {href:nextHref, textContent:'Next', getAttribute(){return null;}, scrollIntoView(){}, click(){ location.href = nextHref; }} : null;
  const document = {
    readyState: 'complete',
    body: {innerText:''},
    hidden: false,
    querySelector(selector) {
      if ((selector === "a[aria-label='Next'], a[aria-label='Next page']") && next) return next;
      return null;
    },
    querySelectorAll(selector) {
      if (selector === '.gs_r[data-cid]') return rowNodes;
      if (selector === "a[aria-label='Next'], a[aria-label='Next page']") return next ? [next] : [];
      if (selector === 'a' || selector === 'a,button,[role], [data-label], [data-name]') return clock >= renderAfter ? [...anchorNodes, ...(next ? [next] : [])] : [];
      if (selector === '#gs_n a[href]') return next ? [next] : [];
      return [];
    },
    addEventListener() {}
  };
  const chrome = {
    runtime: {
      onMessage: {addListener(){}},
      async sendMessage(message) {
        if (message.type === 'LSSC_GET_SNAPSHOT') return {ok:true, run, senderTabId:1};
        if (message.type === 'LSSC_PATCH_RUN') { Object.assign(run, message.patch); return {ok:true, run}; }
        if (message.type === 'LSSC_LOG') { logs.push(message); return {ok:true}; }
        return {ok:true};
      }
    }
  };
  const sandbox = {
    console, chrome, document, location,
    window:{addEventListener(){}},
    getComputedStyle(){ return {display:'block',visibility:'visible',opacity:'1'}; },
    MouseEvent:class{}, Event:class{},
    setTimeout(fn, ms){
      if (timed && fn.name !== 'drive') {
        clock += ms;
        if (onWait) onWait(run, clock);
        Promise.resolve().then(fn);
      }
      return 0;
    }, clearTimeout(){},
    Promise, Date: timed ? class extends Date { static now(){ return clock; } } : Date, URL,
    globalThis:null,
    LitSyncScholarCore:Core
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/scholar.js'),'utf8'), sandbox, {filename:'scholar.js'});
  return {hooks:sandbox.__LSSC_TEST_HOOKS__, run, location, logs};
}

function baseRun() {
  return {
    status: Core.STATUS.RUNNING,
    stage: Core.STAGE.VERIFYING_LABEL,
    tabId: 1,
    query: 'federated learning',
    label: 'litsync_20260905_6f1c8008',
    targetTotal: 405,
    savedIds: Array.from({length:405}, (_,i)=>`search-${i+1}`),
    unresolved: {},
    labelAuditIds: [],
    labelAuditPages: [],
    labelRepairRound: 0,
    pagesVisited: Array.from({length:41}, (_,i)=>i*10),
    resultLocations: {},
    libraryIdsBySearchId: {},
    integrityVersion: 2,
    exportMode:'full'
  };
}

function completeLibraryMap(run) {
  run.integrityVersion = 3;
  run.libraryIdsBySearchId = Object.fromEntries(run.savedIds.map((cid,i)=>[cid,`lib-${i+1}`]));
}

test('v0.1.9 search-ID vs Library-ID mismatch migrates to data-lid indexing instead of calling all 341 records unexpected', async () => {
  const run = baseRun();
  run.labelAuditIds = Array.from({length:340}, (_,i)=>`lib-${i+1}`);
  const h = makeHarness({run, rows:[{cid:'lib-341'}]});
  await h.hooks.auditRunLabel(run);
  assert.equal(run.status, Core.STATUS.RUNNING);
  assert.equal(run.stage, Core.STAGE.INDEXING_LIBRARY_IDS);
  assert.equal(run.integrityVersion, 3);
  assert.equal(run.labelVerifiedCount, 341);
  assert.match(run.message, /different IDs on search pages and My Library/i);
  assert.equal(new URL(h.location.href).searchParams.get('q'), 'federated learning');
});

test('indexing reads search-row data-lid and maps it to the search data-cid namespace', async () => {
  const run = baseRun();
  run.savedIds = ['search-1','search-2','search-3'];
  run.targetTotal = 3;
  run.stage = Core.STAGE.INDEXING_LIBRARY_IDS;
  run.integrityVersion = 3;
  const h = makeHarness({
    run,
    url:'https://scholar.google.com/scholar?q=federated+learning&hl=en',
    rows:[
      {cid:'search-1',lid:'lib-A'},
      {cid:'search-2',lid:'lib-B'},
      {cid:'search-3',lid:'lib-C'}
    ]
  });
  await h.hooks.indexLibraryIds(run);
  assert.deepEqual({...run.libraryIdsBySearchId}, {'search-1':'lib-A','search-2':'lib-B','search-3':'lib-C'});
  assert.equal(run.stage, Core.STAGE.OPENING_LIBRARY);
  assert.equal(h.location.href, 'https://scholar.google.com/scholar?scilib=1&hl=en');
});

test('real 405 checkpoint vs 341 actual label records triggers 64 targeted search-record repairs after data-lid mapping', async () => {
  const run = baseRun();
  completeLibraryMap(run);
  run.labelAuditIds = Array.from({length:340}, (_,i)=>`lib-${i+1}`);
  const h = makeHarness({run, rows:[{cid:'lib-341'}]});
  await h.hooks.auditRunLabel(run);
  assert.equal(run.labelVerifiedCount, 341);
  assert.equal(run.labelMissingIds.length, 64);
  assert.equal(run.labelUnexpectedIds.length, 0);
  assert.equal(run.stage, Core.STAGE.REPAIRING_LABEL);
  assert.equal(run.labelRepairRound, 1);
  assert.equal(run.labelRepairIds.length, 64);
  assert.equal(run.labelRepairIds[0], 'search-342');
  assert.equal(run.labelRepairIds.at(-1), 'search-405');
  assert.match(run.message, /64 checkpointed Scholar record/);
  assert.ok(h.logs.some(x => x.event === 'label_integrity_mismatch'));
});

test('exact 405 Library IDs in label passes audit even though search IDs are a different namespace', async () => {
  const run = baseRun();
  completeLibraryMap(run);
  run.labelAuditIds = Array.from({length:395}, (_,i)=>`lib-${i+1}`);
  const last = Array.from({length:10}, (_,i)=>({cid:`lib-${396+i}`}));
  const h = makeHarness({run, rows:last, url:'https://scholar.google.com/scholar?scilib=1026&hl=en&start=390'});
  await h.hooks.auditRunLabel(run);
  assert.equal(run.labelAuditComplete, true);
  assert.equal(run.labelVerifiedCount, 405);
  assert.equal(run.labelMissingIds.length, 0);
  assert.equal(run.labelUnexpectedIds.length, 0);
  assert.equal(run.stage, Core.STAGE.LABEL_LOADING);
  assert.equal(Core.labelIntegritySatisfied(run), true);
  assert.match(run.message, /Integrity verified 405\/405/);
  assert.ok(h.logs.some(x => x.event === 'label_integrity_pass'));
});

test('truly foreign Library ID in LitSync label remains a hard integrity stop', async () => {
  const run = baseRun();
  run.savedIds = ['search-1','search-2'];
  run.targetTotal = 2;
  run.libraryIdsBySearchId = {'search-1':'lib-1','search-2':'lib-2'};
  run.integrityVersion = 3;
  run.labelAuditIds = ['lib-1','lib-2'];
  const h = makeHarness({run, rows:[{cid:'foreign-lib'}], url:'https://scholar.google.com/scholar?scilib=1026&hl=en&start=20'});
  await h.hooks.auditRunLabel(run);
  assert.equal(run.status, Core.STATUS.NEEDS_ATTENTION);
  assert.equal(run.labelUnexpectedIds.includes('foreign-lib'), true);
  assert.match(run.message, /unexpected Scholar ID/i);
});

test('legacy repair falls back to checkpointed result-page offsets; newer runs use per-search-ID locations', () => {
  const run = baseRun();
  run.labelRepairIds = ['search-11','search-72'];
  run.pagesVisited = [0,10,20,30,40,50,60,70,80];
  const h = makeHarness({run, rows:[]});
  assert.deepEqual(Array.from(h.hooks.repairOffsets(run)), [0,10,20,30,40,50,60,70,80]);
  run.resultLocations = {'search-11':10, 'search-72':70};
  assert.deepEqual(Array.from(h.hooks.repairOffsets(run)), [10,70]);
});

test('current Scholar data-lid marks an already-saved result even when control text still says Save', () => {
  const run = baseRun();
  const h = makeHarness({run, rows:[]});
  const row = {
    getAttribute(name) { return name === 'data-lid' ? 'library-record-id' : null; },
    querySelector() { return null; }
  };
  assert.equal(h.hooks.rowLibraryId(row), 'library-record-id');
  assert.equal(h.hooks.rowAlreadySaved(row), true);
});


test('v0.2.4 can recover a v0.2.3 run whose long label is truncated in the Scholar sidebar', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[{text:'litsync_20260905_d2ee15f2_1412378...'}]
  });
  const link = h.hooks.findRunLabelLink(run.label);
  assert.ok(link);
  assert.equal(link.textContent, 'litsync_20260905_d2ee15f2_1412378...');
});

test('truncated-label recovery refuses an ambiguous sidebar prefix', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[
      {text:'litsync_20260905_d2ee...'},
      {text:'litsync_20260905_d2ee...'}
    ]
  });
  assert.equal(h.hooks.findRunLabelLink(run.label), null);
});

test('v0.2.5 recovers legacy long label when Scholar exposes a clipped prefix without literal ellipsis', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[{text:'litsync_20260905_d2ee15f2_1412378'}]
  });
  const link = h.hooks.findRunLabelLink(run.label);
  assert.ok(link);
});

test('v0.2.5 recovers legacy long label when Scholar inserts whitespace inside the rendered label', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[{text:'litsync_20260905_d2e\n e15f2_1412378...'}]
  });
  const link = h.hooks.findRunLabelLink(run.label);
  assert.ok(link);
});

test('v0.2.5 recovers legacy long label with zero-width DOM characters', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[{text:'litsync_20260905_d2ee\u200b15f2_1412378…'}]
  });
  const link = h.hooks.findRunLabelLink(run.label);
  assert.ok(link);
});

test('v0.2.5 prefix recovery still refuses two possible LitSync labels', () => {
  const run = baseRun();
  run.label = 'litsync_20260905_d2ee15f2_141237814';
  const h = makeHarness({
    run,
    anchors:[
      {text:'litsync_20260905_d2ee15f2_1412'},
      {text:'litsync_20260905_d2ee15f2_14123'}
    ]
  });
  assert.equal(h.hooks.findRunLabelLink(run.label), null);
});

const REAL_LABEL = 'litsync_20260905_d2ee15f2_141237814';
const REAL_PREFIX = 'litsync_20260905_d2ee15f2_1412378...';
const REAL_HREF = 'https://scholar.google.com/scholar?scilib=1030&hl=en&as_sdt=0,5';
function realRun() {
  const run = baseRun();
  Object.assign(run, {runId:'preserved-run', label:REAL_LABEL, stage:Core.STAGE.OPENING_LABEL,
    targetTotal:58, scholarTotal:58, savedIds:Array.from({length:58},(_,i)=>`search-${i+1}`),
    pagesVisited:[0,10,20,30,40,50], labelAuditComplete:false});
  completeLibraryMap(run);
  return run;
}

test('live Scholar duplicate menu and sidebar links resolve to one scilib destination', () => {
  const h=makeHarness({run:realRun(), anchors:[{text:REAL_PREFIX,href:REAL_HREF},{text:REAL_PREFIX,href:REAL_HREF}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL).href, REAL_HREF);
});

test('duplicate full names at different destinations remain ambiguous', () => {
  const h=makeHarness({run:realRun(),anchors:[{text:REAL_LABEL},{text:REAL_LABEL}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL),null);
});

test('multiple old labels do not displace the exact current label', () => {
  const h=makeHarness({run:realRun(),anchors:[{text:'litsync_20260904_old'},{text:REAL_LABEL,href:REAL_HREF},{text:'litsync_20260905_old'}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL).href,REAL_HREF);
});

test('nested text and full accessible descendant name identify a clipped label', () => {
  const child={tagName:'SPAN',textContent:'',getAttribute(name){return name==='title'?REAL_LABEL:null;}};
  const h=makeHarness({run:realRun(),anchors:[{text:'litsync...',children:[child],href:REAL_HREF}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL).href,REAL_HREF);
});

test('nested elements concatenate their text and whitespace normally', () => {
  const parts=['litsync_20260905_', 'd2ee15f2_', '141237814'];
  const children=parts.map(textContent=>({tagName:'SPAN',textContent,getAttribute(){return null;}}));
  const h=makeHarness({run:realRun(),anchors:[{text:parts.join('\n'),children,href:REAL_HREF}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL).href,REAL_HREF);
});

test('a longer different full name is not accepted as the current label', () => {
  const h=makeHarness({run:realRun(),anchors:[{text:REAL_LABEL+'_other'}]});
  assert.equal(h.hooks.findRunLabelLink(REAL_LABEL),null);
});

test('label URL validation rejects foreign sites, special views and search URLs', () => {
  const h=makeHarness({run:realRun()});
  for(const href of ['https://evil.example/scholar?scilib=1030','javascript:alert(1)',
    'https://scholar.google.com/scholar?scilib=1','https://scholar.google.com/scholar?scilib=5',
    'https://scholar.google.com/scholar?scilib=6',REAL_HREF+'&q=other']) {
    assert.equal(h.hooks.runLabelDestination(href),null);
  }
  assert.equal(h.hooks.runLabelDestination(REAL_HREF+'&start=50&as_ylo=2025').href,
    'https://scholar.google.com/scholar?scilib=1030&hl=en');
});

test('58/58 checkpoint waits for sidebar, persists destination, and opens audit without recollection', async () => {
  const run=realRun();
  const before=JSON.stringify({saved:run.savedIds,map:run.libraryIdsBySearchId,pages:run.pagesVisited,unresolved:run.unresolved});
  const h=makeHarness({run,anchors:[{text:REAL_PREFIX,href:REAL_HREF},{text:REAL_PREFIX,href:REAL_HREF}],timed:true,renderAfter:700});
  await h.hooks.openRunLabel(run);
  assert.equal(run.stage,Core.STAGE.VERIFYING_LABEL);
  assert.equal(run.status,Core.STATUS.RUNNING);
  assert.equal(run.labelReference.scilib,'1030');
  assert.equal(run.labelReference.name,REAL_LABEL);
  assert.equal(h.location.href,'https://scholar.google.com/scholar?scilib=1030&hl=en');
  assert.equal(JSON.stringify({saved:run.savedIds,map:run.libraryIdsBySearchId,pages:run.pagesVisited,unresolved:run.unresolved}),before);
  assert.equal(Core.labelIntegritySatisfied(run),false);
  assert.equal(Core.fullExportIntegritySatisfied(run),false);
});

test('stored destination resumes even when no label text is rendered', async () => {
  const run=realRun();
  run.labelReference={name:REAL_LABEL,href:REAL_HREF,scilib:'1030'};
  const h=makeHarness({run});
  await h.hooks.openRunLabel(run);
  assert.equal(run.stage,Core.STAGE.VERIFYING_LABEL);
  assert.equal(run.savedIds.length,58);
  assert.equal(h.location.href,'https://scholar.google.com/scholar?scilib=1030&hl=en');
});

test('legacy labelUrl is migrated without resetting collection', async () => {
  const run=realRun(); run.labelUrl=REAL_HREF;
  const h=makeHarness({run}); await h.hooks.openRunLabel(run);
  assert.equal(run.labelReference.scilib,'1030');
  assert.equal(run.savedIds.length,58);
});

test('a reference belonging to another run label is not reused', async () => {
  const run=realRun();run.labelReference={name:'old_label',href:REAL_HREF};run.labelUrl=REAL_HREF;
  const h=makeHarness({run,timed:true});await h.hooks.openRunLabel(run);
  assert.equal(run.status,Core.STATUS.NEEDS_ATTENTION);
  assert.equal(run.stage,Core.STAGE.OPENING_LABEL);
});

test('missing current label reports bounded actual candidate diagnostics', async () => {
  const run=realRun();
  const h=makeHarness({run,timed:true,anchors:[{text:'litsync_old',href:REAL_HREF,role:'menuitemradio',title:'old',className:'gs_md_li'}]});
  await h.hooks.openRunLabel(run);
  assert.equal(run.status,Core.STATUS.NEEDS_ATTENTION);
  assert.match(run.message,/Recent diagnostics/);
  const details=h.logs.find(l=>l.event==='needs_attention').details;
  assert.equal(details.candidateCount,1);
  const c=details.candidates[0];
  assert.equal(c.tag,'A');assert.equal(c.rawText,'litsync_old');
  assert.equal(c.href,REAL_HREF);assert.equal(c.role,'menuitemradio');
  assert.equal(c.title,'old');assert.equal(c.className,'gs_md_li');
  assert.equal(c.data['data-label'],'litsync_old');
  assert.equal(run.savedIds.length,58);
});

test('pause during sidebar wait prevents navigation and audit mutation', async () => {
  const run=realRun();
  const h=makeHarness({run,timed:true,renderAfter:500,anchors:[{text:REAL_LABEL,href:REAL_HREF}],onWait(r){r.status=Core.STATUS.PAUSED;}});
  const before=h.location.href;
  await h.hooks.openRunLabel(run);
  assert.equal(h.location.href,before);assert.equal(run.labelReference,undefined);
});

test('pause during a failed sidebar wait remains paused', async () => {
  const run=realRun();
  const h=makeHarness({run,timed:true,onWait(r){r.status=Core.STATUS.PAUSED;}});
  await h.hooks.openRunLabel(run);
  assert.equal(run.status,Core.STATUS.PAUSED);
  assert.equal(h.logs.length,0);
});

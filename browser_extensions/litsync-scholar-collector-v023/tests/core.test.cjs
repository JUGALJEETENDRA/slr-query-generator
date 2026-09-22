const test = require('node:test');
const assert = require('node:assert/strict');
const Core = require('../src/core.js');

test('parses standard Scholar totals', () => {
  assert.equal(Core.parseReportedTotal('About 961,000 results (0.12 sec)'), 961000);
  assert.equal(Core.parseReportedTotal('About 1,716 results'), 1716);
  assert.equal(Core.parseReportedTotal('23 results'), 23);
  assert.equal(Core.parseReportedTotal('1 result'), 1);
});

test('parses Indian grouping used by localized screenshots', () => {
  assert.equal(Core.parseReportedTotal('About 9,61,000 results (0.12 sec)'), 961000);
});

test('returns null when result count is absent', () => {
  assert.equal(Core.parseReportedTotal('Google Scholar'), null);
});

test('hard cap is exactly 1000', () => {
  assert.equal(Core.targetForTotal(961000), 1000);
  assert.equal(Core.targetForTotal(1001), 1000);
  assert.equal(Core.targetForTotal(1000), 1000);
});

test('below-cap totals collect all', () => {
  for (const value of [0, 1, 9, 10, 499, 999]) assert.equal(Core.targetForTotal(value), value);
});

test('unknown total stays bounded by 1000', () => {
  assert.equal(Core.targetForTotal(null), 1000);
});

test('start offset parsing is deterministic', () => {
  assert.equal(Core.extractStartOffset('https://scholar.google.com/scholar?q=x'), 0);
  assert.equal(Core.extractStartOffset('https://scholar.google.com/scholar?start=490&q=x'), 490);
  assert.equal(Core.extractStartOffset('not-a-url'), 0);
});

test('search URL contains exact query and optional start', () => {
  const q = '("federated learning" OR "federated machine learning") AND "intrusion detection"';
  const u = new URL(Core.buildSearchUrl(q, 30));
  assert.equal(u.hostname, 'scholar.google.com');
  assert.equal(u.searchParams.get('q'), q);
  assert.equal(u.searchParams.get('start'), '30');
});

test('blocker detection catches Scholar verification without false positives', () => {
  assert.equal(Core.looksLikeBlocker('https://scholar.google.com/sorry/index', ''), true);
  assert.equal(Core.looksLikeBlocker('https://scholar.google.com/scholar?q=x', 'Our systems have detected unusual traffic from your computer network.'), true);
  assert.equal(Core.looksLikeBlocker('https://scholar.google.com/scholar?q=x', 'About 500 results'), false);
});

test('context validation is strict', () => {
  const good = {schema_version: 1, research_question: 'RQ', active_query_version: 'balanced', query_fingerprint: 'a'.repeat(64), query: 'test'};
  assert.equal(Core.validateContext(good), true);
  assert.equal(Core.validateContext({...good, query: ''}), false);
  assert.equal(Core.validateContext({...good, query_fingerprint: 'abc'}), false);
  assert.equal(Core.validateContext({...good, schema_version: 2}), false);
});

test('label names are deterministic and compact', () => {
  assert.equal(Core.makeLabelName('abcdef123456', new Date('2026-09-05T00:00:00Z')), 'litsync_20260905_abcdef12');
});

test('run starts at zero and cap-safe', () => {
  const context = {schema_version: 1, research_question: 'RQ', active_query_version: 'balanced', query_fingerprint: 'b'.repeat(64), query: 'pan card'};
  const run = Core.createRun(context, {now: '2026-09-05T00:00:00Z', runId: 'test-run'});
  assert.equal(run.runId, 'test-run');
  assert.equal(run.status, Core.STATUS.RUNNING);
  assert.equal(run.stage, Core.STAGE.STARTING);
  assert.deepEqual(run.savedIds, []);
  assert.equal(run.targetTotal, null);
  assert.match(run.label, /^litsync_260905_/);
});

test('unique checkpointing never double-counts Scholar IDs', () => {
  assert.deepEqual(Core.appendUnique(['a', 'b'], 'a'), ['a', 'b']);
  assert.deepEqual(Core.appendUnique(['a', 'b'], 'c'), ['a', 'b', 'c']);
});

test('page identity changes by page or result IDs', () => {
  const a = Core.pageIdentity('https://scholar.google.com/scholar?q=x&start=0', ['1', '2']);
  const b = Core.pageIdentity('https://scholar.google.com/scholar?q=x&start=10', ['1', '2']);
  const c = Core.pageIdentity('https://scholar.google.com/scholar?q=x&start=0', ['1', '3']);
  assert.notEqual(a, b);
  assert.notEqual(a, c);
});

test('progress never exceeds 100%', () => {
  assert.deepEqual(Core.progress({savedIds:['1','2'], targetTotal:4}), {saved:2,target:4,remaining:2,percent:50});
  assert.equal(Core.progress({savedIds:['1','2','3'], targetTotal:2}).percent, 100);
});

test('temporary Scholar operation failures are classified without treating normal pages as blocked', () => {
  assert.equal(Core.classifyScholarOperationFailure("The system can't perform the operation now. Try again later."), 'temporary_operation_failure');
  assert.equal(Core.classifyScholarOperationFailure('About 416 results'), null);
});

test('unresolved failures are recorded and can be filtered by final retry state', () => {
  let unresolved = Core.recordUnresolved({}, 'cid1', {error:'save failed', at:'2026-09-05T07:00:00Z', skippedDuringTraversal:true, finalized:false});
  unresolved = Core.recordUnresolved(unresolved, 'cid1', {error:'save failed again', at:'2026-09-05T07:01:00Z', skippedDuringTraversal:true, finalized:false});
  unresolved.cid2 = {error:'permanent', finalized:true};
  assert.equal(unresolved.cid1.failures, 2);
  assert.deepEqual(Core.unresolvedIds({unresolved}), ['cid1','cid2']);
  assert.deepEqual(Core.unresolvedIds({unresolved}, {includeFinalized:false}), ['cid1']);
});

test('final accessible target reconciles a lower real Scholar result set without exceeding 1000', () => {
  assert.equal(Core.finalAccessibleTarget(405), 405);
  assert.equal(Core.finalAccessibleTarget(1000), 1000);
  assert.equal(Core.finalAccessibleTarget(1400), 1000);
});

test('stale Next on a short last page is confirmed as real end of results', () => {
  assert.equal(Core.confirmedEndOfResults({
    currentStart: 400,
    rowCount: 5,
    latestReportedTotal: 405,
    discoveredCount: 405,
    nextStart: 400
  }), true);
});

test('stale Next in the middle of results is not silently treated as the end', () => {
  assert.equal(Core.confirmedEndOfResults({
    currentStart: 150,
    rowCount: 10,
    latestReportedTotal: 416,
    discoveredCount: 160,
    nextStart: 150
  }), false);
});

test('1000-cap final page is accepted even when Scholar exposes no forward Next', () => {
  assert.equal(Core.confirmedEndOfResults({
    currentStart: 990,
    rowCount: 10,
    latestReportedTotal: 5000,
    discoveredCount: 1000,
    nextStart: null
  }), true);
});

test('throttle cooldown backs off and caps at thirty minutes', () => {
  assert.deepEqual([1,2,3,4,5,6,7,20].map(Core.throttleCooldownMs), [60000,120000,300000,600000,900000,1800000,1800000,1800000]);
});

test('provisional target can grow with later Scholar estimates but never shrinks before real end', () => {
  assert.equal(Core.provisionalTarget(null, 405), 405);
  assert.equal(Core.provisionalTarget(405, 416), 416);
  assert.equal(Core.provisionalTarget(416, 405), 416);
  assert.equal(Core.provisionalTarget(1000, 5000), 1000);
});

test('Page N of M results text is parsed as the latest Scholar total', () => {
  assert.equal(Core.parseReportedTotal('Page 41 of 405 results'), 405);
});

test('set difference and equality helpers support strict label auditing', () => {
  assert.deepEqual(Core.differenceStrings(['a','b','c'], ['b','a']), ['c']);
  assert.equal(Core.sameUniqueSet(['a','b','a'], ['b','a']), true);
  assert.equal(Core.sameUniqueSet(['a','b'], ['a','c']), false);
});

test('legacy label integrity still validates exact ID sets for backward compatibility', () => {
  const good = {
    targetTotal: 3,
    savedIds: ['a','b','c'],
    labelAuditComplete: true,
    labelVerifiedIds: ['c','a','b'],
    unresolved: {}
  };
  assert.equal(Core.labelIntegritySatisfied(good), true);
  assert.equal(Core.labelIntegritySatisfied({...good, labelVerifiedIds:['a','b']}), false);
  assert.equal(Core.labelIntegritySatisfied({...good, unresolved:{c:{finalized:true}}}), false);
  assert.equal(Core.labelIntegritySatisfied({...good, savedIds:['a','b']}), false);
});

test('v3 label integrity compares search data-lid library IDs to My Library data-cid IDs', () => {
  const run = {
    targetTotal: 3,
    savedIds: ['search-1','search-2','search-3'],
    integrityVersion: 3,
    libraryIdsBySearchId: {'search-1':'lib-A','search-2':'lib-B','search-3':'lib-C'},
    labelAuditComplete: true,
    labelVerifiedIds: ['lib-C','lib-A','lib-B'],
    unresolved: {}
  };
  assert.equal(Core.labelIntegritySatisfied(run), true);
  assert.deepEqual(Core.expectedLibraryIds(run), ['lib-A','lib-B','lib-C']);
  assert.equal(Core.labelIntegritySatisfied({...run, labelVerifiedIds:['lib-A','lib-B']}), false);
  assert.equal(Core.labelIntegritySatisfied({...run, libraryIdsBySearchId:{'search-1':'lib-A','search-2':'lib-B'}}), false);
});

test('Scholar CSV inspector validates exact row count and required headers', () => {
  const text = [
    'Authors,Title,Publication,Volume,Number,Pages,Year,Publisher',
    '"A One","Paper One","J",1,1,"1-2",2025,"P"',
    '"A Two","Paper Two","J",1,1,"3-4",2026,"P"'
  ].join('\r\n');
  const result = Core.inspectScholarCsv(text, 2);
  assert.equal(result.ok, true);
  assert.equal(result.rowCount, 2);
  assert.deepEqual(result.header.slice(0, 2), ['Authors', 'Title']);
});

test('Scholar CSV inspector rejects real failure shape 341 rows when target is 405', () => {
  const rows = ['Authors,Title,Publication,Volume,Number,Pages,Year,Publisher'];
  for (let i = 1; i <= 341; i += 1) rows.push(`"Author ${i}","Paper ${i}","J",1,1,"1-2",2026,"P"`);
  const result = Core.inspectScholarCsv(rows.join('\n'), 405);
  assert.equal(result.ok, false);
  assert.equal(result.rowCount, 341);
  assert.match(result.errors.join(' '), /341.*405/);
});

test('Scholar CSV inspector rejects BibTeX even if caller thinks the file is CSV', () => {
  const result = Core.inspectScholarCsv('@article{x, title={Wrong format}}', 1);
  assert.equal(result.ok, false);
  assert.equal(result.bibtexLike, true);
  assert.match(result.errors.join(' '), /BibTeX/i);
});

test('CSV parser counts quoted embedded newlines as one record', () => {
  const text = 'Authors,Title,Publication,Year\r\n"A","Line one\nLine two","J",2026\r\n"B","Normal","J",2026\r\n';
  const result = Core.inspectScholarCsv(text, 2);
  assert.equal(result.ok, true);
  assert.equal(result.rowCount, 2);
});

test('full export completion requires v3 library-ID integrity plus exact verified CSV rows', () => {
  const run = {
    targetTotal: 2,
    savedIds: ['search-a','search-b'],
    integrityVersion: 3,
    libraryIdsBySearchId: {'search-a':'lib-a','search-b':'lib-b'},
    labelAuditComplete: true,
    labelVerifiedIds: ['lib-b','lib-a'],
    unresolved: {},
    exportMode: 'full',
    exportCsvVerified: true,
    exportCsvRowCount: 2
  };
  assert.equal(Core.fullExportIntegritySatisfied(run), true);
  assert.equal(Core.fullExportIntegritySatisfied({...run, exportCsvRowCount: 1}), false);
  assert.equal(Core.fullExportIntegritySatisfied({...run, exportCsvVerified: false}), false);
});

test('zero-result Scholar run is a valid completion without creating a CSV', () => {
  assert.equal(Core.fullExportIntegritySatisfied({targetTotal:0,savedIds:[],unresolved:{},exportMode:'full'}), true);
});

test('fresh runs of the same query get different dedicated Scholar labels and start from zero', () => {
  const context = {schema_version: 1, research_question: 'RQ', active_query_version: 'balanced', query_fingerprint: 'c'.repeat(64), query: 'same query'};
  const first = Core.createRun(context, {now: '2026-09-05T13:10:00.111Z', runId: 'run-one'});
  const second = Core.createRun(context, {now: '2026-09-05T13:10:01.222Z', runId: 'run-two'});
  assert.notEqual(first.label, second.label);
  assert.equal(first.label, 'litsync_260905_cccc_0s7y5r');
  assert.equal(second.label, 'litsync_260905_cccc_0s7z0m');
  assert.deepEqual(first.savedIds, []);
  assert.deepEqual(second.savedIds, []);
  assert.equal(first.targetTotal, null);
  assert.equal(second.targetTotal, null);
});

test('run label remains compact but is unique within the same day', () => {
  const a = Core.makeRunLabelName('abcdef123456', new Date('2026-09-05T00:00:00.001Z'));
  const b = Core.makeRunLabelName('abcdef123456', new Date('2026-09-05T00:00:00.002Z'));
  assert.equal(a, 'litsync_260905_abcd_000001');
  assert.equal(b, 'litsync_260905_abcd_000002');
  assert.notEqual(a, b);
  assert.ok(a.length <= 28);
});

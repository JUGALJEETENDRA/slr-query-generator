const {test}=require('node:test');
const assert=require('node:assert/strict');
const H=require('./helpers.cjs');
test('live IEEE modal hides LayoutWrapper from accessibility while keeping result evidence rendered',()=>{
  const w=H.page(`<div id="LayoutWrapper" aria-hidden="true">${H.fixture('results')}</div>${H.fixture('export')}`).window,D=w.LitSyncIEEE.dom;
  assert.equal(D.button(w.document,'Export'),null); // Covered controls remain non-actionable.
  assert.equal(D.results(w.document).total,1);
  assert.equal(D.verifyDialog(w.document).textContent,'Download');w.close();
});
test('native export cap must agree with available results; no guessed row claims',()=>{
  const w=H.page(H.fixture('results')+H.fixture('export')).window,D=w.LitSyncIEEE.dom;
  w.document.querySelector('xpl-search-results-exporter p').textContent='If no results are selected, up to 1000 results will be included. CSV file';
  assert.throws(()=>D.verifyDialog(w.document),/expected/);w.close();
});
test('script reload in the same document pauses instead of silently taking over',async()=>{
  const b=await H.started(),sender={url:'https://ieeexplore.ieee.org/search/advanced/command',tab:{id:7},frameId:0,documentId:'doc'};
  await b.send({type:'CLAIM',document:'first'},sender);
  const r=await b.send({type:'CLAIM',document:'replacement'},sender);
  assert.equal(r.data,null);assert.equal(b.state().job.active,false);assert.match(b.state().job.reason,/restarted/);
});
test('pre-export Resume from Command Search restores the recorded results URL',async()=>{
  const b=await H.started(),j=b.store.litsync_ieee_native_v1.job;
  j.resultsUrl=`https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=${encodeURIComponent(j.query)}&sortType=desc_p_Publication_Year`;
  await b.send({type:'PAUSE'});await b.send({type:'RESUME'});
  assert.equal(b.tabs.get(7).url,j.resultsUrl);assert.equal(b.state().job.stage,'waiting_for_results');
});
test('interrupted and non-CSV downloads cannot report Complete',async()=>{
  for(const state of ['interrupted','complete']){
    const b=await H.started(),j=b.store.litsync_ieee_native_v1.job,at=Date.now();j.stage='downloading';j.attempt={at,id:'once'};
    const d={id:5,url:'https://ieeexplore.ieee.org/export',filename:'error.html',mime:'text/html',startTime:new Date(at).toISOString(),state};
    b.downloads.push(d);await b.chrome.downloads.onCreated.emit(d);assert.equal(b.state().job.stage,'paused');
  }
});
test('missing-download alarm pauses instead of repeating native Download',async()=>{
  const b=await H.started(),j=b.store.litsync_ieee_native_v1.job;j.stage='downloading';j.attempt={at:Date.now(),id:'once'};
  await b.chrome.alarms.onAlarm.emit({name:`ieee-download-${j.id}`});assert.equal(b.state().job.stage,'paused');assert.match(b.state().job.reason,/unconfirmed/);
});
test('foreign website cannot sync a query or clear an existing run',async()=>{
  const b=await H.started(),before=b.state();
  assert.equal((await b.send({type:'SYNC',data:H.payload},{url:'https://example.com',frameId:0})).ok,false);
  assert.equal((await b.send({type:'CLEAR'},{url:'https://example.com',frameId:0})).ok,false);assert.deepEqual(b.state(),before);
});
test('duplicate download transition is rejected durably',async()=>{
  const b=await H.started(),j=b.store.litsync_ieee_native_v1.job,sender={url:'https://ieeexplore.ieee.org/search/searchresult.jsp',tab:{id:7},frameId:0,documentId:'doc'};
  await b.send({type:'CLAIM',document:'x'},sender);b.store.litsync_ieee_native_v1.job.stage='waiting_for_export_dialog';
  const first=await b.send({type:'STEP',id:j.id,document:'x',stage:'downloading'},sender);
  const second=await b.send({type:'STEP',id:j.id,document:'x',stage:'downloading'},sender);
  assert.equal(first.ok,true);assert.equal(second.ok,false);assert.equal(b.state().job.attempt.id,first.data.attempt.id);
});

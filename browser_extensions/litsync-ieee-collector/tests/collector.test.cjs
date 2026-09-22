const {test}=require('node:test');
const assert=require('node:assert/strict');
const H=require('./helpers.cjs');
test('parses ordinary, comma-separated and singular result headings',()=>{
  const w=H.page('').window,C=w.LitSyncIEEE;
  assert.deepEqual({...C.range('Showing 51-75 of 2,739 results')},{first:51,last:75,total:2739});
  assert.equal(C.range('Showing 1 of 1 result').total,1);
  assert.equal(C.range('Showing 51-25 of 10 results'),null);assert.equal(C.range('loading'),null);w.close();
});
test('results readiness requires matching rendered cards and Export',()=>{
  const w=H.page(H.fixture('results')).window,D=w.LitSyncIEEE.dom;
  assert.equal(D.results(w.document).total,1);
  w.document.querySelector('xpl-results-item').remove();assert.equal(D.results(w.document),null);w.close();
});
test('semantic Export discovery never chooses Download PDFs or pagination',()=>{
  const w=H.page(H.fixture('results')).window;assert.equal(w.LitSyncIEEE.dom.button(w.document,'Export').textContent,'Export');w.close();
});
test('native dialog, no-selection mode and scoped Download discovery',()=>{
  const w=H.page(H.fixture('results')+H.fixture('export')).window,D=w.LitSyncIEEE.dom;
  assert.ok(D.exportDialog(w.document));assert.equal(D.verifyDialog(w.document).textContent,'Download');w.close();
});
test('selected count or checked result strictly blocks native no-selection export',()=>{
  const w=H.page(H.fixture('results')+H.fixture('export')).window,D=w.LitSyncIEEE.dom;
  w.document.querySelector('xpl-results-item input').checked=true;assert.throws(()=>D.verifyDialog(w.document),/selected/);
  w.document.querySelector('xpl-results-item input').checked=false;
  w.document.querySelector('xpl-search-results-exporter p').textContent='You have selected 25 results for download to a CSV file.';
  assert.throws(()=>D.verifyDialog(w.document),/did not confirm/);w.close();
});
test('query submission fills the exact string and dispatches native input events',()=>{
  const w=H.page(H.fixture('command')).window,D=w.LitSyncIEEE.dom,input=D.command(w.document);let inputs=0;
  input.addEventListener('input',()=>inputs++);const q='  ("All Metadata":"A & B")\n';D.fill(input,q);
  assert.equal(input.value,q);assert.equal(inputs,1);w.close();
});
test('URL verification accepts only exact query or the observed IEEE outer wrapper',()=>{
  const w=H.page('').window,C=w.LitSyncIEEE,q='"Document Title":"Example"';
  const u=q=>`https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=${encodeURIComponent(q)}`;
  assert.ok(C.sameQuery(u(q),q));assert.ok(C.sameQuery(u(`(${q})`),q));assert.equal(C.sameQuery(u(q+' OR other'),q),false);w.close();
});
test('login export, CAPTCHA and unexpected modal stop; hidden cookie dialog does not',()=>{
  const w=H.page(H.fixture('results')+H.fixture('export')).window,D=w.LitSyncIEEE.dom;
  w.document.querySelector('xpl-search-results-exporter').innerHTML='<h1>Download Results</h1><input type="password">';
  assert.match(D.interruption(w.document,true),/sign in manually/);
  w.document.body.innerHTML='<p>Unusual traffic</p>';assert.match(D.interruption(w.document),/verification/);
  w.document.body.innerHTML='<div role="dialog">Unexpected notice</div>';assert.match(D.interruption(w.document),/unexpected/);
  w.document.querySelector('div').hidden=true;assert.equal(D.interruption(w.document),null);w.close();
});
for(const version of ['balanced','high_recall']) test(`exact ${version} routing and fingerprint`,async()=>{
  const b=await H.started(version),j=b.state().job;assert.equal(j.query,H.payload.query_versions[version].ieee_xplore);assert.equal(j.version,version);assert.match(j.fingerprint,/^[a-f0-9]{64}$/);
});
test('popup VIEW and query sync cannot start a job',async()=>{
  const b=H.background();await b.send({type:'SYNC',data:H.payload},H.bridgeSender);const before=b.state();await b.send({type:'VIEW'});assert.deepEqual(b.state(),before);assert.equal(b.state().job,null);
});
test('actual popup opening sends only VIEW and no navigation',async()=>{
  const w=H.page(H.source('popup/popup.html')).window,sent=[];
  w.chrome={runtime:{sendMessage:async m=>{sent.push(m.type);return {ok:true,data:{context:null,job:null,logs:[]}};}},storage:{onChanged:{addListener(){}}}};
  w.eval(H.source('popup/popup.js'));await new Promise(r=>setTimeout(r,0));assert.deepEqual(sent,['VIEW']);w.close();
});
test('Start from a content script is rejected',async()=>{
  const b=H.background(),r=await b.send({type:'START',version:'balanced'},H.bridgeSender);assert.equal(r.ok,false);assert.match(r.error,/Explicit popup/);
});
test('concurrent duplicate Start is suppressed without resetting the checkpoint',async()=>{
  const b=H.background();await b.send({type:'SYNC',data:H.payload},H.bridgeSender);
  const r=await Promise.all([b.send({type:'START',version:'balanced'}),b.send({type:'START',version:'balanced'})]);assert.equal(r.filter(x=>x.ok).length,1);assert.equal(b.state().job.stage,'submitting_query');
});
test('reload pauses; Resume reuses matching results without rerunning search; stale tab recovers',async()=>{
  const b=await H.started(),s=b.store.litsync_ieee_native_v1;
  s.job.resultsUrl=`https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=${encodeURIComponent(s.job.query)}`;
  b.tabs.get(7).url=s.job.resultsUrl;await b.chrome.runtime.onStartup.emit();assert.equal(b.state().job.active,false);
  const id=s.job.id;await b.send({type:'RESUME'});assert.equal(b.state().job.stage,'waiting_for_results');assert.equal(b.tabs.size,1);
  b.tabs.delete(7);await b.send({type:'RESUME'});assert.equal(b.state().job.id,id);assert.notEqual(b.state().job.tabId,7);
});
test('Discard removes only its checkpoint, not synced query or browser downloads',async()=>{
  const b=await H.started();await b.send({type:'DISCARD'});assert.equal(b.state().job,null);assert.ok(b.state().context);assert.equal(b.tabs.size,1);
});
test('download source/time correlation rejects unrelated files',()=>{
  const w=H.page('').window,C=w.LitSyncIEEE,at=Date.now();
  assert.ok(C.downloadMatches({url:'blob:https://ieeexplore.ieee.org/abc',startTime:new Date(at).toISOString()},{at}));
  assert.equal(C.downloadMatches({url:'https://example.org/file.csv',startTime:new Date(at).toISOString()},{at}),false);w.close();
});
for(const total of [1,75,2739]) test(`native ${total}-result pipeline: zero selection/navigation clicks, one CSV Download`,async()=>{
  const b=await H.started(),dom=H.page(H.fixture('command')),w=dom.window;let downloads=0,searches=0,selection=0,interval;
  const query=b.state().job.query;
  w.chrome={runtime:{sendMessage:m=>b.send(m,{url:w.location.href,tab:{id:7},frameId:0,documentId:'doc1'})}};
  w.setInterval=fn=>{interval=fn;return 1;};w.setTimeout=(fn)=>setTimeout(fn,0);
  w.document.addEventListener('click',e=>{
    if(e.target.matches('input[type=checkbox],xpl-paginator button'))selection++;
    if(e.target.textContent==='Search'){
      searches++;assert.equal(w.document.querySelector('textarea').value,query);
      const url=`https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=${encodeURIComponent(`(${query})`)}`;dom.reconfigure({url});b.tabs.get(7).url=url;
      w.document.body.innerHTML=H.fixture('results');
      const count=Math.min(total,25);w.document.querySelector('h1').textContent=total===1?'Showing 1 of 1 result':`Showing 1-${count} of ${total.toLocaleString()} results`;
      const card=w.document.querySelector('xpl-results-item');for(let i=1;i<count;i++)card.after(card.cloneNode(true));
    }
    if(e.target.textContent==='Export'){
      w.document.body.innerHTML=`<div id="LayoutWrapper" aria-hidden="true">${w.document.body.innerHTML}</div>`;
      w.document.body.insertAdjacentHTML('beforeend',H.fixture('export').replace('up to 1 results',`up to ${Math.min(total,1000).toLocaleString()} results`));
    }
    if(e.target.textContent==='Download'){
      downloads++;const d={id:9,url:'blob:https://ieeexplore.ieee.org/native',filename:'IEEE.csv',mime:'text/csv',startTime:new Date().toISOString(),state:'complete'};
      b.downloads.push(d);b.chrome.downloads.onCreated.emit(d);
    }
  });
  w.eval(H.source('src/content.js'));w.eval(H.source('src/content.js'));
  await H.until(()=>b.state().job.stage==='complete');await interval();
  assert.equal(selection,0);assert.equal(searches,1);assert.equal(downloads,1);assert.equal(b.state().job.resultsCount,total);assert.equal(b.state().job.downloadState,'complete');w.close();
});
test('full-document continuation claims the previously authorized waiting-results stage',async()=>{
  const b=await H.started(),j=b.state().job,sender={url:H.page('').window.location.href,tab:{id:7},frameId:0,documentId:'old'};
  await b.send({type:'CLAIM'},sender);await b.send({type:'STEP',id:j.id,stage:'waiting_for_results'},sender);
  const next=await b.send({type:'CLAIM'},{...sender,documentId:'new'});assert.equal(next.data.id,j.id);assert.equal(next.data.stage,'waiting_for_results');
  const stale=await b.send({type:'ERROR',id:j.id,reason:'old error'},sender);assert.equal(stale.ok,false);
});
test('ambiguous or missing download never triggers a second download on Resume',async()=>{
  const b=await H.started(),j=b.store.litsync_ieee_native_v1.job;
  j.stage='downloading';j.attempt={at:Date.now(),id:'once'};
  await b.send({type:'RESUME'});assert.equal(b.state().job.stage,'paused');assert.match(b.state().job.reason,/unconfirmed/);
  await b.send({type:'RESUME'});assert.equal(b.state().job.attempt.id,'once');assert.equal(b.state().job.active,false);
});
test('observable download completion updates the checkpoint',async()=>{
  const b=await H.started(),j=b.store.litsync_ieee_native_v1.job,at=Date.now();j.stage='downloading';j.attempt={at,id:'once'};
  const d={id:42,url:'https://ieeexplore.ieee.org/native.csv',filename:'native.csv',startTime:new Date(at).toISOString(),state:'in_progress'};b.downloads.push(d);
  await b.chrome.downloads.onCreated.emit(d);assert.equal(b.state().job.downloadState,'in_progress');d.state='complete';await b.chrome.downloads.onChanged.emit({id:42,state:{current:'complete'}});assert.equal(b.state().job.stage,'complete');
});

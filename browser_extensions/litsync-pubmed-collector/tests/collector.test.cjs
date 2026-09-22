const {test}=require('node:test');
const assert=require('node:assert/strict');
const H=require('./helpers.cjs');
const url=q=>`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(q)}`;
function ready(total=2,query='test',withPanel=true){
  const dom=H.page(H.fixture('search')+(withPanel?H.fixture('save'):''),url(query));
  dom.window.document.querySelector('#id_term').value=query;
  dom.window.document.querySelector('.results-amount').textContent=`${total.toLocaleString()} results`;
  return dom;
}
for(const [input,expected] of [['578 results',578],['1 result',1],['10,001 results',10001],['0 results',0],['Loading',null],['Page 1 of 58',null]])test(`result count: ${input}`,()=>{
  const w=H.page('').window;assert.equal(w.LitSyncPubMed.resultCount(input),expected);w.close();
});
test('popup opening only reads VIEW',async()=>{
  const w=H.page(H.source('popup/popup.html')).window,messages=[];
  w.chrome={runtime:{sendMessage:async m=>{messages.push(m.type);return {ok:true,data:{context:null,job:null,logs:[]}};}},storage:{onChanged:{addListener(){}}}};
  w.eval(H.source('popup/popup.js'));await new Promise(r=>setTimeout(r,0));assert.deepEqual(messages,['VIEW']);w.close();
});
test('sync and VIEW do not create a job',async()=>{
  const b=H.background();await b.send({type:'SYNC',data:H.payload},H.bridgeSender);const before=b.state();await b.send({type:'VIEW'});assert.deepEqual(b.state(),before);assert.equal(b.state().job,null);
});
test('explicit Start can only originate from the extension popup',async()=>{
  const b=H.background();assert.equal((await b.send({type:'START'},H.bridgeSender)).ok,false);
});
for(const version of ['balanced','high_recall'])test(`exact ${version} routing and SHA-256 fingerprint`,async()=>{
  const b=await H.started(version),j=b.state().job;assert.equal(j.query,H.payload.query_versions[version].pubmed);assert.equal(j.version,version);assert.match(j.fingerprint,/^[a-f0-9]{64}$/);
});
test('search field preserves whitespace, tags, parentheses and operators exactly',()=>{
  const w=ready().window,D=w.LitSyncPubMed.dom,input=D.queryInput(w.document),q=' ("a & b"[tiab] OR "c"[MeSH Terms]) ';let events=0;
  input.addEventListener('input',()=>events++);D.fill(input,q);assert.equal(input.value,q);assert.equal(events,1);w.close();
});
test('hidden duplicate input created by PubMed Save is ignored',()=>{
  const w=ready().window;assert.equal(w.document.querySelectorAll('#id_term').length,2);assert.equal(w.LitSyncPubMed.dom.queryInput(w.document).closest('form').id,'search-form');w.close();
});
test('matching URL requires exact query and PubMed search origin',()=>{
  const w=H.page('').window,C=w.LitSyncPubMed;assert.ok(C.sameQuery(url('(a[tiab])'),'(a[tiab])'));
  assert.equal(C.sameQuery(url('a[tiab]'),'(a[tiab])'),false);assert.equal(C.sameQuery('https://example.org/?term=a','a'),false);w.close();
});
test('readiness requires the current query, result count, cards and Save',()=>{
  const w=ready().window,D=w.LitSyncPubMed.dom;assert.equal(D.results(w.document,'test').total,2);assert.equal(D.results(w.document,'wrong'),null);
  w.document.querySelectorAll('article').forEach(e=>e.remove());assert.equal(D.results(w.document,'test'),null);w.close();
});
test('Save discovery is exact and does not choose Send to',()=>{
  const w=ready().window;assert.equal(w.LitSyncPubMed.dom.button(w.document,'Save').id,'save-results-panel-trigger');w.close();
});
test('native Save citations panel discovered by heading',()=>{
  const w=ready().window;assert.equal(w.LitSyncPubMed.dom.saveDialog(w.document).id,'save-action-panel');w.close();
});
for(const [kind,label,value] of [['selection','All results','all-results'],['format','CSV','csv']]){
  test(`${kind}: associated control and exact native option discovery`,()=>{
    const w=ready().window,D=w.LitSyncPubMed.dom,c=D[kind](D.saveDialog(w.document));assert.equal(D.option(c,label).value,value);w.close();
  });
  test(`${kind}: activation is verified in selectedOptions`,()=>{
    const w=ready().window,D=w.LitSyncPubMed.dom,c=D[kind](D.saveDialog(w.document));D.choose(c,label);assert.equal(c.value,value);assert.equal(c.selectedOptions[0].textContent,label);w.close();
  });
}
test('native option activation failure is not treated as success',()=>{
  const w=ready().window,D=w.LitSyncPubMed.dom,c=D.selection(D.saveDialog(w.document));c.addEventListener('change',()=>{c.value='this-page';});assert.throws(()=>D.choose(c,'All results'),/did not retain/);w.close();
});
test('Create file requires verified All results and CSV',()=>{
  const w=ready().window,D=w.LitSyncPubMed.dom,p=D.saveDialog(w.document);assert.throws(()=>D.verifyExport(w.document,'test',2),/All results/);
  D.choose(D.selection(p),'All results');assert.throws(()=>D.verifyExport(w.document,'test',2),/CSV/);
  D.choose(D.format(p),'CSV');assert.equal(D.verifyExport(w.document,'test',2).textContent,'Create file');w.close();
});
test('native export restrictions stop without inventing or bypassing a limit',()=>{
  const w=ready(10001).window,D=w.LitSyncPubMed.dom,p=D.saveDialog(w.document);p.insertAdjacentHTML('beforeend','<p role="alert">Only the first 10,000 results can be saved.</p>');assert.match(D.interruption(w.document,true),/10,000/);w.close();
});
for(const html of ['<p>Access denied</p>','<p>Verify you are human</p>','<input type="password">','<div role="dialog">Unrelated modal</div>'])test(`interrupts: ${html}`,()=>{
  const w=H.page(html).window;assert.match(w.LitSyncPubMed.dom.interruption(w.document),/Action required/);w.close();
});
test('duplicate Start is serialized and suppressed',async()=>{
  const b=H.background();await b.send({type:'SYNC',data:H.payload},H.bridgeSender);const r=await Promise.all([b.send({type:'START',version:'balanced'}),b.send({type:'START',version:'balanced'})]);assert.equal(r.filter(x=>x.ok).length,1);
});
test('reload Resume before Save preserves query and existing results tab',async()=>{
  const b=await H.started(),j=b.store.litsync_pubmed_native_v1.job;j.resultsUrl=url(j.query);b.tabs.get(7).url=j.resultsUrl;
  await b.chrome.runtime.onStartup.emit();assert.equal(b.state().job.active,false);await b.send({type:'RESUME'});assert.equal(b.state().job.stage,'waiting_for_results');assert.equal(b.tabs.size,1);
});
test('stale tab recovery uses saved results URL and job identity',async()=>{
  const b=await H.started(),j=b.store.litsync_pubmed_native_v1.job;j.resultsUrl=url(j.query);const id=j.id;b.tabs.delete(7);await b.send({type:'RESUME'});assert.equal(b.state().job.id,id);assert.equal(b.tabs.get(b.state().job.tabId).url,j.resultsUrl);
});
test('full document navigation continues only the authorized waiting-results job',async()=>{
  const b=await H.started(),j=b.state().job,s={url:'https://pubmed.ncbi.nlm.nih.gov/',tab:{id:7},frameId:0,documentId:'old'};
  await b.send({type:'CLAIM',document:'a'},s);await b.send({type:'STEP',id:j.id,document:'a',stage:'waiting_for_results'},s);
  const r=await b.send({type:'CLAIM',document:'b'},{...s,url:url(j.query),documentId:'new'});assert.equal(r.data.id,j.id);
});
test('Discard removes checkpoint without changing synced context',async()=>{
  const b=await H.started();await b.send({type:'DISCARD'});assert.equal(b.state().job,null);assert.ok(b.state().context);
});
test('durable Create file intent refuses duplicate transition and retry',async()=>{
  const b=await H.started(),j=b.store.litsync_pubmed_native_v1.job,s={url:'https://pubmed.ncbi.nlm.nih.gov/',tab:{id:7},frameId:0,documentId:'doc'};
  await b.send({type:'CLAIM',document:'a'},s);b.store.litsync_pubmed_native_v1.job.stage='creating_file';
  assert.equal((await b.send({type:'STEP',document:'a',id:j.id,stage:'downloading'},s)).ok,true);
  assert.equal((await b.send({type:'STEP',document:'a',id:j.id,stage:'downloading'},s)).ok,false);
  const attempt=b.state().job.attempt.id;await b.send({type:'RESUME'});assert.equal(b.state().job.attempt.id,attempt);assert.equal(b.state().job.stage,'paused');
});
test('download completion requires source, start time and CSV evidence',async()=>{
  const b=await H.started(),j=b.store.litsync_pubmed_native_v1.job,at=Date.now();j.stage='downloading';j.attempt={at,id:'once'};
  const unrelated={id:1,url:'https://example.org/file.csv',filename:'file.csv',startTime:new Date(at).toISOString(),state:'complete'};
  await b.chrome.downloads.onCreated.emit(unrelated);assert.equal(b.state().job.downloadId,undefined);
  const d={...unrelated,id:2,url:'https://pubmed.ncbi.nlm.nih.gov/results-export-search-data/',state:'in_progress'};b.downloads.push(d);
  await b.chrome.downloads.onCreated.emit(d);assert.equal(b.state().job.downloadId,2);assert.equal(b.state().job.downloadState,'in_progress');d.state='complete';await b.chrome.downloads.onChanged.emit({id:2});assert.equal(b.state().job.stage,'complete');
});
for(const resumePanel of [false,true])test(`actual controller: Search/Save/Create once; no scraping or pagination; resumePanel=${resumePanel}`,async()=>{
  const b=await H.started(),query=b.state().job.query,dom=ready(578,query,resumePanel),w=dom.window;
  const clicks={search:0,save:0,create:0,selection:0,pagination:0};let tick;
  if(resumePanel){const j=b.store.litsync_pubmed_native_v1.job;j.stage='waiting_for_results';j.resultsUrl=url(query);}
  else {dom.reconfigure({url:'https://pubmed.ncbi.nlm.nih.gov/'});w.document.querySelector('#id_term').value='';}
  w.chrome={runtime:{sendMessage:m=>b.send(m,{url:w.location.href,tab:{id:7},frameId:0,documentId:'doc'})}};
  w.setInterval=fn=>{tick=fn;return 1;};w.setTimeout=fn=>setTimeout(fn,0);
  w.fetch=()=>{throw Error('Metadata/API fetch forbidden');};
  w.document.addEventListener('click',e=>{
    e.preventDefault();const label=e.target.textContent.trim();
    if(e.target.matches('input[type=checkbox]'))clicks.selection++;
    if(label==='Next')clicks.pagination++;
    if(label==='Search'){clicks.search++;assert.equal(w.document.querySelector('#id_term').value,query);dom.reconfigure({url:url(query)});b.tabs.get(7).url=url(query);}
    if(label==='Save'){clicks.save++;w.document.body.insertAdjacentHTML('beforeend',H.fixture('save'));}
    if(label==='Create file'){
      clicks.create++;assert.equal(w.document.querySelector('#save-action-selection').value,'all-results');assert.equal(w.document.querySelector('#save-action-format').value,'csv');
      const d={id:8,url:'https://pubmed.ncbi.nlm.nih.gov/results-export-search-data/',filename:'pubmed.csv',mime:'text/csv',startTime:new Date().toISOString(),state:'complete'};b.downloads.push(d);b.chrome.downloads.onCreated.emit(d);
    }
  });
  w.eval(H.source('src/content.js'));w.eval(H.source('src/content.js'));await H.until(()=>b.state().job.stage==='complete');await tick();
  assert.deepEqual(clicks,{search:resumePanel?0:1,save:resumePanel?0:1,create:1,selection:0,pagination:0});assert.equal(b.state().job.resultsCount,578);w.close();
});
test('runtime has no metadata fetch, CSV construction or pagination click path',()=>{
  const code=['src/content.js','src/dom.js','src/background.js'].map(H.source).join('\n');assert.doesNotMatch(code,/\bfetch\s*\(|XMLHttpRequest|eutils|esearch|efetch|downloads\.download\s*\(/);
});

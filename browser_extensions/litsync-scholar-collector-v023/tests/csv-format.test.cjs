const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const Core = require('../src/core.js');

function node(text, dataA, options = {}) {
  return {
    textContent: text,
    tagName: 'A',
    hidden: Boolean(options.hidden),
    parentElement: options.parentElement || null,
    attrs: {'data-a': dataA, role:'menuitem', ...(options.attrs || {})},
    classList: {contains(){return false;}},
    getAttribute(name){ return this.attrs[name] ?? null; },
    querySelectorAll(){ return []; }, querySelector(){ return null; },
    scrollIntoView(){}, dispatchEvent(){}, click(){ this.clicked = (this.clicked||0)+1; }
  };
}

function loadWithControls(controls, href='https://scholar.google.com/scholar?scilib=1') {
  const document = {
    body:{innerText:'',textContent:''}, hidden:false,
    querySelectorAll(selector) {
      if (selector.includes('[role="menu"]') || selector.includes('.gs_md_d') || selector.includes('.gs_md_ul') || selector.includes('#gs_res_ab_exp-d')) return [];
      if (selector.includes('a') || selector.includes('[role="menuitem"]')) return controls;
      return [];
    },
    querySelector(){return null;}, addEventListener(){}, getElementById(){return null;}
  };
  const chrome={runtime:{onMessage:{addListener(){}}, async sendMessage(){return {ok:true,run:null};}}};
  const sandbox={console,chrome,document,window:{addEventListener(){}},location:{href},URL,URLSearchParams,getComputedStyle(){return {display:'block',visibility:'visible',opacity:'1'};},MouseEvent:class{},Event:class{},setTimeout(){return 0;},clearTimeout(){},Promise,Date,globalThis:null,LitSyncScholarCore:Core};
  sandbox.globalThis=sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/scholar.js'),'utf8'),sandbox,{filename:'scholar.js'});
  return sandbox.__LSSC_TEST_HOOKS__;
}

test('CSV selector chooses the visible control explicitly labelled CSV, regardless of data-a value', () => {
  const bib=node('BibTeX','4');
  const csv=node('CSV','0');
  const hooks=loadWithControls([bib,csv]);
  assert.equal(hooks.findCsvFormatControl(), csv);
});

test('CSV selector rejects a stale CSV control hidden by an ancestor', () => {
  const hiddenParent={hidden:true,parentElement:null,getAttribute(){return null;}};
  const stale=node('CSV','4',{parentElement:hiddenParent});
  const live=node('CSV','0');
  const hooks=loadWithControls([stale,live]);
  assert.equal(hooks.findCsvFormatControl(), live);
});

test('numeric data-a alone can never turn a BibTeX control into CSV', () => {
  const bib=node('BibTeX','4');
  const hooks=loadWithControls([bib]);
  assert.equal(hooks.findCsvFormatControl(), null);
});

test('googleusercontent citation-render page is recognized as export result page', () => {
  const hooks=loadWithControls([], 'https://scholar.googleusercontent.com/citations?view_op=export_citations&x=1');
  assert.equal(hooks.isGoogleusercontentExportPage(), true);
});


test('v0.2.1 captures the visible final Scholar CSV URL without rewriting its format parameters', () => {
  const csv=node('CSV','0',{attrs:{href:'https://scholar.googleusercontent.com/citations?view_op=export_citations&cit_fmt=9&token=once'}});
  csv.href=csv.attrs.href;
  csv.closest=()=>null;
  const hooks=loadWithControls([csv]);
  const request=hooks.exportRequestFromControl(csv);
  assert.equal(request.method, 'GET');
  assert.equal(request.url, csv.attrs.href, 'must preserve Scholar\'s exact visible CSV request URL');
  assert.match(request.url, /cit_fmt=9/);
});

test('v0.2.1 can serialize the final all-label export form before the one-use Scholar request is consumed', () => {
  const fields=[
    {name:'view_op',value:'export_citations',type:'hidden',disabled:false,tagName:'INPUT'},
    {name:'scope',value:'page',type:'radio',checked:false,disabled:false,tagName:'INPUT'},
    {name:'scope',value:'label',type:'radio',checked:true,disabled:false,tagName:'INPUT'},
    {name:'cit_fmt',value:'csv-native-value',type:'hidden',disabled:false,tagName:'INPUT'}
  ];
  const form={
    action:'https://scholar.google.com/citations?hl=en',
    method:'GET',
    elements:fields,
    getAttribute(name){ return name==='action' ? this.action : name==='method' ? this.method : null; }
  };
  const submit={
    name:'',value:'',href:'',
    getAttribute(){return null;},
    closest(selector){return selector==='form' ? form : null;}
  };
  const hooks=loadWithControls([]);
  const request=hooks.exportRequestFromControl(submit);
  const parsed=new URL(request.url);
  assert.equal(request.method,'GET');
  assert.equal(parsed.searchParams.get('scope'),'label');
  assert.equal(parsed.searchParams.get('cit_fmt'),'csv-native-value');
  assert.equal(parsed.searchParams.get('view_op'),'export_citations');
  assert.equal(parsed.searchParams.get('hl'),'en');
});

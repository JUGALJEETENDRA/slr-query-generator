const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const Core = require('../src/core.js');

function style(node) { return node && node._hidden ? {display:'none', visibility:'hidden', opacity:'0'} : {display:'block', visibility:'visible', opacity:'1'}; }

function makeNode({text='', tag='div', role=null, checked=false, className='', id='', hidden=false} = {}) {
  const node = {
    _text: text,
    className, id, _hidden: hidden,
    tagName: tag.toUpperCase(),
    checked,
    parentElement: null,
    children: [],
    attrs: {...(role ? {'role': role} : {}), ...(id ? {id} : {})},
    classList: { contains(name) { return String(node.className || '').split(/\s+/).includes(name); } },
    clicked: 0,
    scrollIntoView() {},
    dispatchEvent() { return true; },
    getAttribute(name) { return this.attrs[name] ?? null; },
    matches(selector) {
      if (selector.includes("input[type='radio']") && this.tagName === 'INPUT' && this.attrs.type === 'radio') return true;
      if (selector.includes("[role='radio']") && this.attrs.role === 'radio') return true;
      if (selector.includes("[role='dialog']") && this.attrs.role === 'dialog') return true;
      if (selector === 'label' && this.tagName === 'LABEL') return true;
      if (selector === '.gs_in_ra' && this.classList.contains('gs_in_ra')) return true;
      if (selector === '.gs_md_d' && this.classList.contains('gs_md_d')) return true;
      if (selector === '.gs_md_w' && this.classList.contains('gs_md_w')) return true;
      if (selector === '.gs_md_wnw' && this.classList.contains('gs_md_wnw')) return true;
      return false;
    },
    closest(selector) {
      let cur = this;
      while (cur) {
        if (cur.matches && cur.matches(selector)) return cur;
        cur = cur.parentElement;
      }
      return null;
    },
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] || null;
    },
    querySelectorAll(selector) {
      const all = [];
      const visit = n => { for (const c of n.children || []) { all.push(c); visit(c); } };
      visit(this);
      const sels = selector.split(',').map(x => x.trim());
      const matchOne = (n, sel) => {
        if (sel === 'label') return n.tagName === 'LABEL';
        if (sel === 'span') return n.tagName === 'SPAN';
        if (sel === 'div') return n.tagName === 'DIV';
        if (sel === 'h1' || sel === 'h2' || sel === 'h3') return n.tagName === sel.toUpperCase();
        if (sel === 'button') return n.tagName === 'BUTTON';
        if (sel === 'a') return n.tagName === 'A';
        if (sel === "input[type='submit']") return n.tagName === 'INPUT' && n.attrs.type === 'submit';
        if (sel === "input[type='radio']") return n.tagName === 'INPUT' && n.attrs.type === 'radio';
        if (sel === "[role='radio']") return n.attrs.role === 'radio';
        if (sel === "[role='dialog']") return n.attrs.role === 'dialog';
        if (sel === '.gs_in_ra') return n.classList.contains('gs_in_ra');
        if (sel === '.gs_md_d') return n.classList.contains('gs_md_d');
        if (sel === '.gs_md_w') return n.classList.contains('gs_md_w');
        if (sel === '.gs_md_wnw') return n.classList.contains('gs_md_wnw');
        return false;
      };
      return all.filter(n => sels.some(sel => matchOne(n, sel)));
    },
    click() {
      this.clicked += 1;
      if (this.tagName === 'INPUT' && this.attrs.type === 'radio') this.checked = true;
    }
  };
  Object.defineProperty(node, 'textContent', {
    get() { return [this._text, ...(this.children || []).map(c => c.textContent || '')].filter(Boolean).join(' '); },
    set(v) { this._text = String(v || ''); }
  });
  return node;
}

function append(parent, child) { child.parentElement = parent; parent.children.push(child); return child; }

function harness() {
  const dialog = makeNode({role:'dialog'});
  const title = append(dialog, makeNode({text:'Export articles', tag:'h2'}));
  const row1 = append(dialog, makeNode());
  const radio1 = append(row1, makeNode({tag:'input', id:'scope-page'})); radio1.attrs.type = 'radio'; radio1.checked = true;
  const label1 = append(row1, makeNode({text:'Export articles on this page', tag:'label'})); label1.attrs.for = 'scope-page';
  const row2 = append(dialog, makeNode());
  const radio2 = append(row2, makeNode({tag:'input', id:'scope-label'})); radio2.attrs.type = 'radio';
  const label2 = append(row2, makeNode({text:'Export all articles with this label', tag:'label'})); label2.attrs.for = 'scope-label';
  const submit = append(dialog, makeNode({text:'EXPORT', tag:'button'}));

  // clicking the second label should also select its associated radio (fallback path)
  label2.click = function() { this.clicked += 1; radio2.checked = true; };

  const messages = [];
  const run = {
    status: Core.STATUS.RUNNING,
    stage: Core.STAGE.EXPORT_MENU,
    savedIds: Array.from({length:405}, (_,i)=>String(i+1)),
    targetTotal: 405,
    label: 'litsync_test',
    exportMode: 'full',
    exportFormatChosen: true
  };

  const document = {
    body: {innerText:''}, hidden:false,
    querySelectorAll(selector) {
      const roots = [dialog, ...dialog.querySelectorAll('h1,h2,h3,div,span,label,a,button,input[type=radio]')];
      if (selector === 'h1,h2,h3,div,span') return roots.filter(n => ['H1','H2','H3','DIV','SPAN'].includes(n.tagName));
      if (selector === 'label,.gs_in_ra,span,div') return roots.filter(n => n.tagName === 'LABEL' || n.tagName === 'SPAN' || n.tagName === 'DIV' || n.classList?.contains('gs_in_ra'));
      if (selector === "[role='dialog'],.gs_md_d,.gs_md_w,.gs_md_wnw") return [dialog];
      return [];
    },
    getElementById(id) {
      return [dialog, ...dialog.querySelectorAll("input[type='radio'],label,span,div")].find(n => n.id === id || n.attrs?.id === id) || null;
    },
    querySelector() { return null; },
    addEventListener() {}
  };

  const chrome = {
    runtime: {
      onMessage: {addListener(){}},
      async sendMessage(message) {
        messages.push(message);
        if (message.type === 'LSSC_GET_SNAPSHOT') return {ok:true, run};
        if (message.type === 'LSSC_PATCH_RUN') { Object.assign(run, message.patch); return {ok:true, run}; }
        if (message.type === 'LSSC_LOG') return {ok:true};
        if (message.type === 'LSSC_EXPORT_TRIGGERED') return {ok:true};
        return {ok:true};
      }
    }
  };

  const sandbox = {
    console, chrome, document,
    window: {addEventListener(){}},
    location: {href:'https://scholar.google.com/scholar?scilib=1'},
    getComputedStyle: style,
    MouseEvent: class { constructor(type, init){this.type=type;Object.assign(this,init);} },
    Event: class { constructor(type, init){this.type=type;Object.assign(this,init);} },
    setTimeout() { return 0; }, clearTimeout() {},
    Promise, Date,
    globalThis: null,
    LitSyncScholarCore: Core
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/scholar.js'),'utf8'), sandbox, {filename:'scholar.js'});
  return {hooks:sandbox.__LSSC_TEST_HOOKS__, dialog, radio1, radio2, submit, messages, run};
}

test('export confirmation chooses ALL articles with label, not current page', async () => {
  const h = harness();
  assert.equal(h.hooks.radioSelected(h.radio1), true);
  assert.equal(h.hooks.radioSelected(h.radio2), false);
  await h.hooks.completeExportArticlesDialog(h.run, h.dialog);
  assert.equal(h.radio2.checked, true);
  assert.equal(h.submit.clicked, 1);
  assert.ok(h.messages.some(m => m.type === 'LSSC_EXPORT_TRIGGERED'));
  const triggerIndex = h.messages.findIndex(m => m.type === 'LSSC_EXPORT_TRIGGERED');
  const patchIndex = h.messages.findIndex(m => m.type === 'LSSC_PATCH_RUN' && /Exporting all 405/.test(m.patch.message || ''));
  assert.ok(patchIndex >= 0 && triggerIndex > patchIndex);
});

test('export dialog detector finds the visible confirmation dialog', () => {
  const h = harness();
  assert.equal(h.hooks.findExportArticlesDialog(), h.dialog);
  assert.equal(h.hooks.exportScopeControl(h.dialog).textContent, 'Export all articles with this label');
});

test('real Scholar-style nested modal with hidden native radios is handled', async () => {
  const outer = makeNode({className:'gs_md_d'});
  const header = append(outer, makeNode({className:'gs_md_w'}));
  const title = append(header, makeNode({text:'Export articles', tag:'h2'}));
  const body = append(outer, makeNode({className:'gs_md_bdy'}));

  const pageWrap = append(body, makeNode({tag:'span', className:'gs_in_ra'}));
  const pageRadio = append(pageWrap, makeNode({tag:'input', id:'gs_exp_page', hidden:true}));
  pageRadio.attrs.type = 'radio'; pageRadio.checked = true;
  const pageLabel = append(pageWrap, makeNode({text:'Export articles on this page', tag:'label'}));
  pageLabel.attrs.for = 'gs_exp_page';

  const allWrap = append(body, makeNode({tag:'span', className:'gs_in_ra'}));
  const allRadio = append(allWrap, makeNode({tag:'input', id:'gs_exp_all', hidden:true}));
  allRadio.attrs.type = 'radio';
  const allLabel = append(allWrap, makeNode({text:'Export all articles with this label', tag:'label'}));
  allLabel.attrs.for = 'gs_exp_all';
  allLabel.click = function() { this.clicked += 1; pageRadio.checked = false; allRadio.checked = true; };
  allRadio.click = function() { this.clicked += 1; pageRadio.checked = false; allRadio.checked = true; };

  const footer = append(outer, makeNode({className:'gs_md_ftr'}));
  const submit = append(footer, makeNode({text:'EXPORT', tag:'button'}));

  const allNodes = () => [outer, ...outer.querySelectorAll("h1,h2,h3,div,span,label,a,button,input[type='radio']")];
  const document = {
    body: {innerText:''}, hidden:false,
    querySelectorAll(selector) {
      const nodes = allNodes();
      if (selector === 'h1,h2,h3,div,span') return nodes.filter(n => ['H1','H2','H3','DIV','SPAN'].includes(n.tagName));
      if (selector === 'label,.gs_in_ra,span,div') return nodes.filter(n => n.tagName === 'LABEL' || n.tagName === 'SPAN' || n.tagName === 'DIV' || n.classList?.contains('gs_in_ra'));
      if (selector === "[role='dialog'],.gs_md_d,.gs_md_w,.gs_md_wnw") return nodes.filter(n => n.classList?.contains('gs_md_d') || n.classList?.contains('gs_md_w'));
      return [];
    },
    getElementById(id) { return allNodes().find(n => n.id === id || n.attrs?.id === id) || null; },
    querySelector() { return null; }, addEventListener() {}
  };

  const run = {status:Core.STATUS.RUNNING, stage:Core.STAGE.EXPORT_MENU, savedIds:Array.from({length:405},(_,i)=>String(i+1)), targetTotal:405, label:'litsync_real', exportMode:'full', exportFormatChosen:true};
  const messages=[];
  const chrome={runtime:{onMessage:{addListener(){}}, async sendMessage(message){
    messages.push(message);
    if(message.type==='LSSC_GET_SNAPSHOT') return {ok:true,run};
    if(message.type==='LSSC_PATCH_RUN'){Object.assign(run,message.patch);return {ok:true,run};}
    return {ok:true};
  }}};
  const sandbox={console,chrome,document,window:{addEventListener(){}},location:{href:'https://scholar.google.com/scholar?scilib=1'},getComputedStyle:style,
    MouseEvent:class{constructor(type,init){this.type=type;Object.assign(this,init);}},Event:class{constructor(type,init){this.type=type;Object.assign(this,init);}},
    setTimeout(){return 0;},clearTimeout(){},Promise,Date,globalThis:null,LitSyncScholarCore:Core};
  sandbox.globalThis=sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/scholar.js'),'utf8'),sandbox,{filename:'scholar.js'});
  const hooks=sandbox.__LSSC_TEST_HOOKS__;

  assert.equal(hooks.findExportArticlesDialog(), outer, 'must select the whole modal, not header-only .gs_md_w');
  const control=hooks.exportScopeControl(outer);
  assert.equal(control, allLabel, 'must target the all-label option even though native radio is hidden');
  assert.equal(hooks.radioSelected(control), false);
  await hooks.completeExportArticlesDialog(run, outer);
  assert.equal(allRadio.checked, true);
  assert.equal(pageRadio.checked, false);
  assert.equal(submit.clicked, 1);
  assert.ok(messages.some(m=>m.type==='LSSC_EXPORT_TRIGGERED'));
});

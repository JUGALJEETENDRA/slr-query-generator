const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname, '..');
const source = name => fs.readFileSync(path.join(root, name), 'utf8');
const fixture = name => source(`tests/fixtures/${name}.html`);
const event = () => ({listeners: [], addListener(fn) {this.listeners.push(fn);}, async emit(...args) {for (const fn of this.listeners) await fn(...args);}});
function page(html, url = 'https://ieeexplore.ieee.org/search/advanced/command') {
  const dom = new JSDOM(html, {url, runScripts: 'outside-only'}), w = dom.window;
  w.HTMLElement.prototype.getClientRects = function() {return this.closest('[hidden]') ? [] : [{width:100,height:20}];};
  Object.defineProperty(w.HTMLElement.prototype, 'innerText', {get() {return this.textContent;}});
  w.eval(source('src/core.js')); w.eval(source('src/dom.js'));
  return dom;
}
function background() {
  const store = {}, tabs = new Map([[7, {id:7,url:'https://ieeexplore.ieee.org/search/advanced/command'}]]), downloads = [];
  let nextTab = 8;
  const chrome = {
    runtime: {getURL: p => `chrome-extension://test/${p}`, onMessage:event(), onStartup:event(), onInstalled:event()},
    storage: {local: {get:async k => ({[k]:structuredClone(store[k])}), set:async o => Object.assign(store,structuredClone(o))}},
    tabs: {query:async () => [tabs.get(7)], get:async id => {if (!tabs.has(id)) throw Error('closed'); return tabs.get(id);},
      create:async o => {const t={id:nextTab++,...o}; tabs.set(t.id,t);return t;}, update:async (id,o) => {const t={...tabs.get(id),...o};tabs.set(id,t);return t;}, onRemoved:event()},
    downloads: {onCreated:event(),onChanged:event(),search:async o => downloads.filter(d => o.id == null || d.id===o.id)},
    alarms: {create:async()=>{},onAlarm:event()}
  };
  const ctx = vm.createContext({chrome, crypto:webcrypto, URL, TextEncoder, Date, console});
  ctx.importScripts = () => vm.runInContext(source('src/core.js'),ctx);
  vm.runInContext(source('src/background.js'),ctx);
  const send = (m,sender={url:chrome.runtime.getURL('popup/popup.html')}) => new Promise(resolve => chrome.runtime.onMessage.listeners[0](m,sender,resolve));
  const state = () => structuredClone(store.litsync_ieee_native_v1);
  return {chrome,store,tabs,downloads,send,state,ctx};
}
const payload = {schema_version:1, query_versions:{balanced:{ieee_xplore:' ("All Metadata":"alpha") '},high_recall:{ieee_xplore:'("All Metadata":"alpha" OR "All Metadata":"beta")'}},query_fingerprints:{balanced:'b',high_recall:'h'}};
const bridgeSender = {url:'http://127.0.0.1:8000/',frameId:0};
async function started(version='balanced') {const b=background();await b.send({type:'SYNC',data:payload},bridgeSender);await b.send({type:'START',version});return b;}
async function until(fn) {for(let i=0;i<300;i++){if(fn())return;await new Promise(r=>setTimeout(r,5));}throw Error('test timed out');}
module.exports={root,source,fixture,page,background,payload,bridgeSender,started,until};

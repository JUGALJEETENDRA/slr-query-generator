const status = document.querySelector('#status');
const scopusQuery = document.querySelector('#scopus-query');
const webOfScienceQuery = document.querySelector('#web-of-science-query');
const copyScopus = document.querySelector('#copy-scopus');
const copyWebOfScience = document.querySelector('#copy-web-of-science');
const startWebOfScience = document.querySelector('#start-web-of-science');

chrome.storage.local.get(['litsyncQueryContext', 'litsyncScopusContext', 'litsyncScopusAutomation', 'litsyncWebOfScienceAutomation']).then(({ litsyncQueryContext, litsyncScopusContext, litsyncScopusAutomation: scopusAutomation, litsyncWebOfScienceAutomation: webOfScienceAutomation }) => {
  const context = litsyncQueryContext || litsyncScopusContext;
  const automation = webOfScienceAutomation?.updatedAt > scopusAutomation?.updatedAt
    ? webOfScienceAutomation : scopusAutomation;
  if (automation?.message) status.textContent = automation.message;
  if (!context) return;
  scopusQuery.value = context.scopusQuery || '';
  webOfScienceQuery.value = context.webOfScienceQuery || '';
  copyScopus.disabled = !context.scopusQuery;
  copyWebOfScience.disabled = !context.webOfScienceQuery;
  startWebOfScience.disabled = !context.webOfScienceQuery;
  if (!automation?.message) status.textContent = `Query version: ${context.queryVersion}`;
});

function copyQuery(field, button) {
  return async () => {
    await navigator.clipboard.writeText(field.value);
    button.textContent = 'Copied';
  };
}

copyScopus.addEventListener('click', copyQuery(scopusQuery, copyScopus));
copyWebOfScience.addEventListener('click', copyQuery(webOfScienceQuery, copyWebOfScience));

startWebOfScience.addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'LITSYNC_START_WEB_OF_SCIENCE_AUTOMATION' }, (response) => {
    status.textContent = response?.ok
      ? 'Web of Science automation started.'
      : (response?.error || 'Open the Web of Science tab, then try again.');
  });
});

document.querySelector('#open').textContent = 'Start Scopus search and CSV export';
document.querySelector('#open').addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'LITSYNC_START_SCOPUS_AUTOMATION' }, (response) => {
    status.textContent = response?.ok
      ? 'Scopus automation started.'
      : (response?.error || 'Open the Scopus tab, then try again.');
  });
});

# LitSync

LitSync is a local website for running the main stages of a systematic literature review:

1. Generate database-specific search queries.
2. Merge and deduplicate exported study records.
3. Screen titles and abstracts.

It also provides PRISMA 2020 artifacts, manual review tools, validation exports, and standalone browser extensions for Google Scholar, PubMed, and IEEE Xplore.

## What LitSync does

| Step | Purpose | Main output |
| --- | --- | --- |
| Query Generator | Uses Gemini Web to create Balanced and High Recall queries for five academic databases | Google Scholar, Scopus, Web of Science, IEEE Xplore, and PubMed queries |
| LitSync Merge | Imports CSV, XLS, and XLSX exports, normalizes fields, and combines duplicate records while preserving the richest metadata | `clean_dataset.csv` |
| CSV Screener | Screens titles and abstracts with Local AI through Ollama or Gemini Web | Complete, KEEP, MAYBE, REJECT, review-queue, and summary exports |

The website runs on your computer at `http://127.0.0.1:8000`. It is not hosted on a public server.

## Requirements

Install these before starting:

- Windows 10 or Windows 11
- Python 3.11 or newer
- Microsoft Edge or Google Chrome
- An internet connection and a Google account for Gemini Web query generation
- Ollama only if you want to use Local AI screening

Node.js is not required to run LitSync. It is only needed to run browser-extension development tests.

## Installation on Windows

### 1. Download the project

Open PowerShell and run:

```powershell
cd $HOME
git clone --branch gemini-working-final --single-branch https://github.com/JUGALJEETENDRA/slr-query-generator.git litsync-screening-clean
cd litsync-screening-clean
```

If the project is already on your computer, open PowerShell in its folder instead:

```powershell
cd C:\Users\Harshil\litsync-screening-clean
```

### 2. Create a Python virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After activation, the PowerShell prompt normally starts with `(.venv)`.

If PowerShell does not allow the activation script, use Command Prompt:

```bat
cd /d C:\Users\Harshil\litsync-screening-clean
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install LitSync

With the virtual environment active, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

The Playwright Chromium installation is required because LitSync uses a dedicated browser profile for Gemini Web.

### 4. Create the local configuration file

```powershell
Copy-Item .env.example .env
```

The default `.env` contents are sufficient for normal use:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
SEMANTIC_SCHOLAR_API_KEY=
```

The Semantic Scholar key is optional and can remain blank.

### 5. Optional: install Local AI screening

This step is required only when using **Local AI** in Step 3. Gemini Web screening does not require Ollama.

Install Ollama from [ollama.com/download](https://ollama.com/download), open it, and run:

```powershell
ollama pull qwen3.5:4b
ollama list
```

Keep Ollama running while using Local AI screening.

## Start LitSync

Open PowerShell in the project folder and run:

```powershell
cd C:\Users\Harshil\litsync-screening-clean
.\.venv\Scripts\Activate.ps1
python run_litsync.py
```

Then open:

```text
http://127.0.0.1:8000
```

Keep the PowerShell window open while using LitSync. Press `Ctrl+C` in that window to stop the server.

On Windows, you can also double-click `start.bat`. It uses `.venv` automatically when that environment exists.

## First Gemini Web setup

Query generation uses Gemini Web. Local AI is still available separately for screening.

1. Start LitSync and open `http://127.0.0.1:8000`.
2. Enter a research question in Step 1.
3. Click **Generate Queries**.
4. LitSync opens its dedicated Chromium window.
5. If Gemini asks you to sign in, sign in manually with your Google account.
6. Complete any normal Gemini welcome or consent screen.
7. Wait until the Gemini message box is visible.
8. Return to LitSync and click **Generate Queries** again if the first attempt timed out.

The dedicated login is saved in:

```text
browser_profiles/gemini
```

Do not open the same profile manually while LitSync is using it. If LitSync reports that the profile is locked, close the dedicated Gemini Chromium window and retry.

## Install the browser extensions

LitSync has three independent Manifest V3 extensions:

| Collector | Version | Browser workflow |
| --- | --- | --- |
| Google Scholar Collector | 0.2.6 | Collects results into a dedicated Scholar library label and verifies the native CSV export |
| PubMed Collector | 0.1.0 | Search → Save → All results → CSV → Create file |
| IEEE Xplore Collector | 0.1.0 | Command Search → Export → native CSV Download |

Each extension has separate storage and checkpoints. Installing one does not overwrite another.

### Download an extension from LitSync

1. Generate a query in Step 1.
2. Scroll below the generated queries to **Browser Extensions**.
3. Click the download button for Google Scholar, PubMed, or IEEE Xplore.
4. Open the browser Downloads list and locate the ZIP file.
5. Extract the ZIP into its own permanent folder. Do not load the ZIP file directly.

Example extraction folder:

```text
C:\Users\Harshil\Documents\LitSync Extensions\PubMed
```

The extracted folder you load must directly contain `manifest.json`.

### Load an unpacked extension in Microsoft Edge

1. Open Edge.
2. Enter `edge://extensions` in the address bar.
3. Turn on **Developer mode**.
4. Click **Load unpacked**.
5. Select the extracted extension folder containing `manifest.json`.
6. Repeat these steps for each collector you want to install.
7. Pin the LitSync collector icons from Edge's Extensions menu if you want easy access.
8. Refresh the LitSync page and the academic-database page after installing or reloading an extension.

### Load an unpacked extension in Google Chrome

1. Open Chrome.
2. Enter `chrome://extensions` in the address bar.
3. Turn on **Developer mode**.
4. Click **Load unpacked**.
5. Select the extracted extension folder containing `manifest.json`.
6. Refresh LitSync and the academic-database page.

### Load extensions directly from this repository

Developers can load these folders without downloading the ZIP files:

```text
browser_extensions/litsync-scholar-collector-v023
browser_extensions/litsync-pubmed-collector
browser_extensions/litsync-ieee-collector
```

The Scholar folder name is historical; its current manifest version is 0.2.6.

## Use Step 1: Query Generator

1. Open the **Query Generator** tab.
2. Enter a clear research question.
3. Click **Generate Queries**.
4. Wait for Gemini Web to return the concepts.
5. Review the generated database queries.
6. Choose **Balanced** or **High Recall**.
7. Copy a query manually, or use a connected standalone collector.

Balanced is the default. High Recall adds more related terminology.

Changing the research question invalidates the old synchronized extension context. Generate the queries again before starting a collector.

## Use the Google Scholar Collector

1. Sign in to Google Scholar in the same browser where the extension is installed.
2. Generate the LitSync queries and choose Balanced or High Recall.
3. Open Google Scholar.
4. Open **LitSync Scholar Collector** from the browser toolbar.
5. Confirm that the exact query is displayed.
6. Click **Start collection**.
7. Leave the Scholar tab open while collection continues.
8. If Scholar requests verification or another manual action, complete it normally and click **Resume**.
9. Wait for the popup to report successful CSV completion.

The collector has a maximum collection target of 1,000 accessible Scholar results. It uses a dedicated label for each new run and maintains a durable checkpoint.

Do not click **Reset** or **Discard** when you intend to continue an existing run.

## Use the PubMed Collector

1. Open [PubMed](https://pubmed.ncbi.nlm.nih.gov/) in the browser where the extension is installed.
2. Generate the LitSync queries.
3. Open **LitSync PubMed Collector**.
4. Select Balanced or High Recall.
5. Verify the exact synchronized query.
6. Click **Start collection**.
7. The extension runs PubMed Search, opens Save, chooses All results and CSV, and clicks Create file.
8. Wait for **Complete — PubMed CSV downloaded**.

The extension uses PubMed's native CSV export. It does not scrape articles, paginate through records, or construct a CSV itself.

## Use the IEEE Xplore Collector

1. Sign in to IEEE Xplore in the browser where the extension is installed.
2. Open [IEEE Command Search](https://ieeexplore.ieee.org/search/advanced/command) in a desktop-width window.
3. Generate the LitSync queries.
4. Open **LitSync IEEE Collector**.
5. Select Balanced or High Recall and verify the exact query.
6. Click **Start collection**.
7. The extension runs Command Search and uses IEEE's native Export and CSV Download controls.
8. Wait for **Complete — IEEE CSV downloaded**.

The IEEE native export supports up to the limit shown by IEEE. The collector does not click Download PDFs, scrape individual records, or create its own CSV.

## Collector resume and safety

- Opening a collector popup never starts a new job.
- Only **Start collection** authorizes a new job.
- A running job may continue across normal page navigation.
- **Resume** continues the existing checkpoint.
- **Discard** removes that collector's checkpoint and should be used only when you intentionally want to abandon it.
- Never start two runs in the same collector at the same time.
- Handle login pages, CAPTCHA, access warnings, and session-expired screens manually. The collectors do not bypass them.

## Use Step 2: merge and deduplicate

1. Export or collect records from the databases you searched.
2. Open the **LitSync** tab.
3. Select one or more CSV, XLS, or XLSX files.
4. Click **Deduplicate & Merge**.
5. Review the displayed input, deduplicated, and duplicate counts.
6. Download `clean_dataset.csv`.

Duplicate papers are matched using DOI and normalized title information. When duplicates contain different amounts of metadata, LitSync keeps one record and fills its empty fields with the richest available values. A blank value cannot overwrite populated metadata.

## Use Step 3: screen papers

1. Open the **CSV Screener** tab.
2. Choose a screening engine:
   - **Local AI** keeps screening data on the computer and requires Ollama.
   - **Gemini Web** uses the saved Gemini browser session.
3. Upload the clean CSV from Step 2.
4. Enter the research context if additional explanation is useful.
5. Enter inclusion criteria.
6. Enter exclusion criteria.
7. Enable the resume checkbox only when continuing an interrupted identical run.
8. Click **Screen Papers**.
9. Keep LitSync and its server running until the job finishes.

The dashboard reports TOTAL, KEEP, MAYBE, REJECT, progress, and runtime. MAYBE records remain available for optional human review.

Available screening downloads include all screened papers, KEEP, MAYBE, the human review queue, REJECT, and the screening summary.

## PRISMA and validation artifacts

LitSync maintains a PRISMA 2020 record for the active import and screening workflow. The website provides an SVG flow diagram, JSON manifest, and CSV manifest.

The diagram reports identification, actual duplicate removal, title/abstract screening, exclusions, pending manual review, and provisional title/abstract inclusions. It does not claim that full-text eligibility assessment has been performed.

Completed screening jobs can also produce a blinded 60-paper gold-validation sample. Fill the `Gold_Decision` column with `KEEP`, `REJECT`, or `UNSURE`, save the file as CSV, and upload it through the Gold Validation section.

## Files created while LitSync runs

Runtime data is stored in project-managed folders such as:

```text
uploads/
outputs/
private/
browser_profiles/
artifacts/
```

Do not delete these folders while a screening or collector-related validation is running. Screening exports and checkpoints may be needed for resume and auditability.

## Stop and restart

To stop LitSync, return to the PowerShell window running the server and press `Ctrl+C`.

To start it again:

```powershell
cd C:\Users\Harshil\litsync-screening-clean
.\.venv\Scripts\Activate.ps1
python run_litsync.py
```

Do not stop the server during an active screening run.

## Troubleshooting

### `python` or `py` is not recognized

Install Python from [python.org/downloads](https://www.python.org/downloads/) and enable **Add Python to PATH** during installation.

### LitSync does not open

Check that the server window says `LitSync is available at http://localhost:8000`, then open `http://127.0.0.1:8000`. If port 8000 is already in use, close the older LitSync server before starting a new one.

### Gemini does not generate a query

- Check the dedicated Chromium window opened by LitSync.
- Sign in to Gemini manually if requested.
- Complete any consent page.
- Confirm that the Gemini message box is visible.
- Close an older LitSync Gemini browser if the saved profile is locked.
- Confirm that `python -m playwright install chromium` completed successfully.

### Local AI screening is unavailable

```powershell
ollama list
ollama pull qwen3.5:4b
```

Confirm Ollama is running at `http://localhost:11434`.

### The collector says the query is not synchronized

1. Refresh LitSync after installing or reloading the extension.
2. Generate the queries again.
3. Keep the results visible in Step 1.
4. Refresh the database tab.
5. Reopen the collector popup.

### Load unpacked reports an error

Make sure you selected the extracted folder that directly contains `manifest.json`. Do not select the ZIP file or a parent folder containing another folder of the same name.

### A collector stops for login, CAPTCHA, or access verification

Complete the required action manually on the database website, then use **Resume**. Do not reset the run unless you intend to abandon its checkpoint.

## Run the tests

Install development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Run the complete Python suite:

```powershell
python -m pytest -q
```

Run each collector suite:

```powershell
cd browser_extensions\litsync-scholar-collector-v023
npm install --no-package-lock
npm test

cd ..\litsync-pubmed-collector
npm install --no-package-lock
npm test

cd ..\litsync-ieee-collector
npm install --no-package-lock
npm test

cd ..\..
```

Validate Python syntax and the Git diff:

```powershell
python -m compileall -q litsync_app
git diff --check
```

## Project structure

```text
litsync_app/          FastAPI application, query generation, deduplication, screening, and PRISMA
web/                  LitSync website
browser_extensions/   Standalone Scholar, PubMed, and IEEE collectors and packaged ZIP files
tests/                Python tests
run_litsync.py        Supported application launcher
start.bat             Windows launcher
requirements.txt      Runtime Python dependencies
requirements-dev.txt  Test dependencies
```

## Data and access boundaries

- Local AI screening sends screening data only to the locally configured Ollama service.
- Gemini Web query generation and Gemini Web screening send their prompts to Gemini through the signed-in browser session.
- Database collectors operate only through the normal authenticated website interfaces available to the user.
- Collectors do not bypass login, CAPTCHA, subscription access, export restrictions, or rate limits.
- Collector ZIP downloads are served from a fixed allowlist; arbitrary local files are not exposed.

## License

Add the project's license terms in a root `LICENSE` file before public distribution.

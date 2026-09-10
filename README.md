# Thai News Intelligence Dashboard

NLP-powered analysis of Thai news and the audience reaction to it. The system
collects Thai **YouTube live chat** from news-channel streams, groups it into
analysable windows, and runs a full Thai NLP pipeline over it: topic
classification into 15 categories, sentiment analysis, keyword extraction,
extractive summarisation and named-entity recognition — all presented through a
dashboard that shows *why* it reached each result.

Built as a university NLP course project. Every number on screen is computed by
the pipeline; nothing is hard-coded or mocked.

---

## Contents

- [Quick start (Windows)](#quick-start-windows)
- [Windows gotchas](#windows-gotchas-read-this-if-something-fails)
- [What it does](#what-it-does)
- [Results](#results)
- [Architecture](#architecture)
- [Data](#data)
- [Command reference](#command-reference)
- [API reference](#api-reference)
- [Replacing the NLP models](#replacing-the-nlp-models)
- [Testing](#testing)
- [Known limitations](#known-limitations)

---

## Quick start (Windows)

Requirements: **Python 3.11+** (developed on 3.14) and **Node.js 20+**
(developed on 24). No internet needed after install — the repository ships real
collected chat and the trained models.

### 1. Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python seed.py
uvicorn app.main:app --reload
```

`seed.py` builds the database from committed files in about 40 seconds: 50
sample articles plus 25,928 real Thai chat messages from six streams, analysed
into 116 windows. It needs no network access.

Leave that terminal running. The API is now at <http://127.0.0.1:8000>, with
interactive docs at <http://127.0.0.1:8000/docs>.

### 2. Frontend

In a **second** terminal:

```powershell
cd frontend
npm install
npm run dev
```

### 3. Open the dashboard

**<http://localhost:5173>**

---

## Windows gotchas (read this if something fails)

These are the actual failures hit while building the project on Windows.

**`.venv\Scripts\activate` fails with "running scripts is disabled on this system"**

PowerShell blocks `.ps1` scripts by default, which affects `activate` and every
`npm` command. Two options:

```powershell
# Option A - permanent fix (user scope, no admin needed)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Option B - avoid activation entirely; works with no policy change
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe seed.py
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

For npm under Option B, use `npm.cmd run dev` instead of `npm run dev`.

**Thai text prints as `?????` or crashes with `UnicodeEncodeError`**

The Windows console defaults to code page 1252, which cannot represent Thai. All
the project's scripts reconfigure their own output, so this only affects ad-hoc
commands you type yourself. Fix it for the session with:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**The dashboard says "Cannot reach the API"**

The backend is not running, or is still starting. The page shows the exact
command to run. If you have just edited a backend file, `--reload` restarts the
worker and requests fail for a second or two — refresh.

**`http://127.0.0.1:5173` refused but `localhost:5173` works**

Fixed in this project (the dev server binds all interfaces), but if you see it
elsewhere it is Vite binding IPv6-only while `127.0.0.1` is IPv4.

**Port already in use**

`strictPort` is on, so Vite fails loudly rather than silently moving. Stop the
other process, or change the port in `frontend/vite.config.ts`.

---

## What it does

### Data collection

Thai news channels stream 24/7 on YouTube with active chat. That chat is
audience reaction to news, in Thai, at volume. Three interchangeable collectors:

| Collector | API key | Live | Replay | Notes |
|---|---|---|---|---|
| **`ytdlp`** (default) | no | yes | **yes** | Reads the chat replay of finished streams, which is what makes the dataset reproducible |
| `youtube_api` | yes | yes | no | Official Data API v3. Cannot read finished streams' chat |
| `file` | no | — | yes | Replays a committed snapshot; **fully offline** |

### Windowing

A single chat message ("สู้ๆ นะครับ") is far too short to carry a topic or a
summary, so consecutive messages are grouped into **windows** of 200 messages
(or 60 minutes, whichever comes first — both configurable). A window is the unit
the pipeline analyses.

Measured on real data: Thai news chat runs about **2 messages/minute**, so a
tight time bound would fire long before the message count and produce ~14-message
windows of 657 characters. With the current settings, windows hold 127–200
messages and 6,000–9,000 characters.

### The pipeline

```
raw text → cleaning → Thai tokenisation → stopword removal → feature extraction
        → topic classification → sentiment analysis → keyword extraction
        → summarisation → named entities → dashboard
```

The **NLP Pipeline** page shows all ten stages with the *real* intermediate
output for any document you select — not a static diagram.

---

## Results

Measured on a stratified held-out split (20%, seed 42) that the models never
saw. `python evaluate.py` regenerates these.

| Task | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | 5-fold CV F1 |
|---|---|---|---|---|---|
| **Topic** (15 classes) | 0.727 | 0.761 | 0.727 | **0.725** | 0.727 ± 0.033 |
| **Sentiment** (3 classes) | 0.740 | 0.695 | 0.678 | **0.682** | 0.705 ± 0.043 |

Random guessing would score 0.067 and 0.333 respectively. The cross-validation
means closely match the hold-out, which indicates the split is not being
overfitted.

### Trained models vs. rule-based baselines

Both are scored on the **same held-out rows**. This is the only fair comparison:
the gazetteer and lexicon were hand-written for this domain, so scoring them on
data they were designed around would flatter them.

| Task | Trained (macro-F1) | Baseline (macro-F1) | Gain |
|---|---|---|---|
| Topic | 0.725 (TF-IDF + logistic regression) | 0.680 (gazetteer) | **+0.045** |
| Sentiment | 0.682 (TF-IDF + logistic regression) | 0.422 (lexicon) | **+0.259** |

Two honest observations:

- **The hand-written gazetteer is genuinely competitive** for topic
  classification, and beats the trained model on `accident` (0.909 vs 0.769) and
  `education` (0.900 vs 0.778). Training bought far less here than for sentiment.
- **Training was decisive for sentiment** (+0.259), where a word-list approach
  cannot capture context.

### Best and worst classes

| Strong | F1 | Weak | F1 |
|---|---|---|---|
| sports | 1.000 | economy | 0.455 |
| politics | 0.889 | business | 0.471 |
| crime | 0.842 | other | 0.571 |
| entertainment | 0.824 | technology | 0.571 |

`economy`, `business` and `technology` genuinely overlap in vocabulary
(บริษัท, ตลาด, ลงทุน, ระบบ), which is where most of the confusion sits — visible
in the confusion matrix on the Evaluation page.

---

## Architecture

```
thai-news-nlp/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, CORS, lifespan
│   │   ├── config.py                every tunable, .env-overridable
│   │   ├── taxonomy.py              the 15 topics + 3 sentiments (single source of truth)
│   │   ├── database/                engine, session, UtcDateTime column type
│   │   ├── models/                  news, nlp_analyses, chat_streams, chat_messages
│   │   ├── schemas/                 Pydantic request/response models
│   │   ├── api/                     meta, news, statistics, analyze, evaluation
│   │   ├── services/
│   │   │   ├── collectors/          ytdlp | youtube_api | file, behind one Protocol
│   │   │   ├── windowing.py         messages → analysable windows
│   │   │   ├── chat_store.py        idempotent persistence
│   │   │   ├── ingest.py            chat → analysed documents
│   │   │   ├── statistics.py        dashboard aggregation (SQL)
│   │   │   └── pipeline.py          orchestrates the 10 stages
│   │   └── nlp/                     ← the swappable layer
│   │       ├── base.py              Protocols every component implements
│   │       ├── registry.py          the ONLY place models are chosen
│   │       ├── matching.py          Thai phrase + compound term matching
│   │       ├── tokenizer.py         PyThaiNLP newmm
│   │       ├── preprocessing.py     cleaning, normalisation, noise detection
│   │       ├── features.py          TF-IDF word + character n-grams
│   │       ├── keyword_extractor.py TF-IDF keywords with corpus IDF
│   │       ├── summarizer.py        extractive (+ optional LLM)
│   │       ├── entities.py          rule/gazetteer NER
│   │       └── backends/            sklearn, gazetteer, lexicon, transformer
│   ├── models/                      trained artefacts + metrics.json (committed)
│   ├── tests/                       210 tests
│   ├── build_dataset.py             merge + validate the labelled dataset
│   ├── train.py                     train the classifiers
│   ├── evaluate.py                  write models/metrics.json
│   ├── collect.py                   collect YouTube chat
│   └── seed.py                      populate the database offline
├── frontend/
│   └── src/
│       ├── components/              ui.tsx (badges, cards, states), charts.tsx
│       ├── layouts/AppLayout.tsx    sidebar, topbar, dark mode
│       ├── pages/                   Dashboard, Explorer, NewsDetail, Analyze,
│       │                            Pipeline, Evaluation, About
│       ├── services/api.ts          typed client
│       ├── hooks/                   data fetching, theme, debounce, toasts
│       └── types/                   mirrors the Pydantic schemas
├── data/
│   ├── news_dataset.csv             747 labelled rows (training)
│   ├── sample_news.csv              50-row display seed
│   ├── dataset_parts/               per-category sources
│   └── chat_snapshots/              6 real streams, 25,928 messages
├── .env.example
└── README.md
```

### Stack

**Backend** FastAPI · SQLAlchemy 2.0 · SQLite · PyThaiNLP · scikit-learn ·
pandas · numpy · yt-dlp
**Frontend** React 19 · TypeScript · Vite · Tailwind CSS 4 · Recharts ·
React Router

---

## Data

### Training set — `data/news_dataset.csv`

747 hand-authored Thai news rows across all 15 categories (49–51 each), each
labelled with **both** topic and sentiment.

```csv
title,content,topic,sentiment
"รถกระบะเสียหลักชนเสาไฟฟ้า มีผู้ได้รับบาดเจ็บ 3 ราย","รถกระบะเสียหลัก...",accident,negative
"ทีมชาติไทยคว้าชัยในการแข่งขันฟุตบอลนัดสำคัญ","ทีมชาติไทยเอาชนะ...",sports,positive
```

Sentiment varies genuinely *inside* each topic — an arrest is positive within
`crime`, a successful rescue is positive within `accident` — so the sentiment
model cannot learn topic as a proxy for polarity.

`build_dataset.py` regenerates this from `data/dataset_parts/` and **refuses to
proceed** on duplicate titles or content, unknown labels, or rows that are too
short. A silent duplicate would leak training rows into the test split and
inflate every metric downstream.

### Collected chat — `data/chat_snapshots/`

25,928 real Thai messages from six streams across five channels (Thairath
News/Sport, ข่าวช่อง8, เรื่องเล่าเช้านี้, บิ๊กแชมป์ FC). Committed so the
project is reproducible and demonstrable with no network access.

### The 15 categories

| | | | |
|---|---|---|---|
| อุบัติเหตุ (Accident) | อาชญากรรม (Crime) | การเมือง (Politics) | เศรษฐกิจ (Economy) |
| เทคโนโลยี (Technology) | กีฬา (Sports) | สุขภาพ (Health) | สังคม (Society) |
| ภัยพิบัติ (Disaster) | สิ่งแวดล้อม (Environment) | การศึกษา (Education) | บันเทิง (Entertainment) |
| ต่างประเทศ (International) | ธุรกิจ (Business) | อื่นๆ (Other) | |

---

## Command reference

All from `backend/` with the virtualenv active.

```powershell
# Database
python seed.py                       # populate from committed files (offline)
python seed.py --reset               # drop tables first
python seed.py --articles-only       # skip chat snapshots (fast)

# Dataset and models
python build_dataset.py              # merge + validate data/dataset_parts/
python train.py                      # train both classifiers
python train.py --task topic         # one task
python train.py --algorithm svm      # LinearSVC instead of logistic regression
python evaluate.py                   # write models/metrics.json

# Collecting new chat
python collect.py --search "ข่าว ไทยรัฐ live"       # find streams
python collect.py VIDEO_ID --snapshot               # collect + save offline copy
python collect.py FILE.jsonl --collector file       # replay offline
python collect.py --list-snapshots

# Run and test
uvicorn app.main:app --reload
python -m pytest
```

Frontend, from `frontend/`:

```powershell
npm run dev        # dev server
npm run build      # production build
npm run test       # 13 rendering tests
npx tsc -b --noEmit
```

---

## API reference

Interactive docs: <http://127.0.0.1:8000/docs>

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, plus which NLP backends are active |
| GET | `/api/topics` | The 15 categories with Thai labels and colours |
| GET | `/api/sentiments` | The 3 sentiment classes |
| GET | `/api/pipeline` | Which backend serves each stage, and whether it is trained |
| GET | `/api/news` | List/filter documents (see below) |
| GET | `/api/news/{id}` | One document with its full NLP analysis |
| GET | `/api/news/{id}/messages` | The chat messages behind a window |
| POST | `/api/news` | Analyse and store a document |
| DELETE | `/api/news/{id}` | Delete a document |
| POST | `/api/analyze` | Analyse text; `?store=true` to persist |
| GET | `/api/statistics` | Every stat card and chart series |
| GET | `/api/statistics/trend` | Volume over time (`daily`/`weekly`/`monthly`) |
| GET | `/api/statistics/keywords` | Keyword cloud data |
| GET | `/api/streams` | Collected YouTube streams |
| GET | `/api/evaluation` | Model metrics and confusion matrices |
| POST | `/api/ingest/youtube` | Collect + analyse a stream's chat |
| GET | `/api/ingest/snapshots` | Offline snapshots available |
| POST | `/api/ingest/reanalyse` | Re-run the pipeline (after swapping a model) |

`GET /api/news` accepts `search`, `topic`, `sentiment`, `source_type`,
`date_from`, `date_to`, `min_confidence`, `sort`
(`newest`/`oldest`/`confidence`/`messages`), `page`, `page_size`. Unknown values
return `422` rather than silently returning everything.

---

## Replacing the NLP models

Every NLP component implements a Protocol in `app/nlp/base.py` and is built
**only** through `app/nlp/registry.py`. No API handler, service or component
constructs a model directly, so swapping one touches nothing else.

To switch backends, change one line in `.env`:

```ini
NLP_TOKENIZER=newmm                  # newmm | newmm-safe | longest | mm
NLP_TOPIC_BACKEND=sklearn            # sklearn | transformer
NLP_SENTIMENT_BACKEND=sklearn        # sklearn | lexicon | transformer
NLP_CHAT_SENTIMENT_BACKEND=lexicon   # separate backend for short chat messages
NLP_SUMMARIZER_BACKEND=extractive    # extractive | llm
NLP_NER_BACKEND=rules                # rules | pythainlp
```

To add a new model: write one adapter class satisfying the relevant Protocol,
register it in `registry.py`, and select it in `.env`. Then
`POST /api/ingest/reanalyse` to re-run the corpus through it.

The registry is **capability aware**: if a configured backend's trained artefact
is missing, it falls back to the rule-based baseline and records why.
`/api/pipeline` reports what is *actually* serving, so the UI can label results
as a baseline honestly rather than implying a fitted model.

### Why chat sentiment uses a separate backend

The trained sentiment model is better on news prose, but measurably worse on
short informal chat — it labelled 59% of real chat messages positive and scored
`แย่ที่สุด` ("the worst") as **positive**, while the hand-built chat lexicon got
every spot check right. On news text the reverse holds. Each is therefore used in
the domain it was built for, which is why `NLP_CHAT_SENTIMENT_BACKEND` exists.

### Optional: Hugging Face transformers

Not installed by default, since it adds roughly 2.5 GB. The adapter is written
and lazily imported:

```powershell
pip install torch transformers
# then in .env
NLP_SENTIMENT_BACKEND=transformer
```

### Optional: LLM summarisation

The default extractive summarizer is fully offline, as required. To use an
external LLM instead, set `LLM_API_KEY` in `.env` and
`NLP_SUMMARIZER_BACKEND=llm`. It falls back to the extractive summarizer on any
failure, so the application never breaks without a key.

---

## Testing

```powershell
cd backend  && python -m pytest      # 210 tests
cd frontend && npm run test          # 13 rendering tests
```

Backend coverage includes Thai preprocessing and tokenisation, the YouTube
InnerTube chat parser (including malformed input), windowing bounds, idempotent
storage, every API endpoint with its error paths, model artefact loading
(missing, corrupt, version-mismatched), domain routing, and snapshot integrity.

The frontend tests mount each real page against fixtures captured from the
running API. That matters because an HTTP 200 on an SPA route only proves
`index.html` was served — it says nothing about whether React rendered or threw.
They assert real values reach the DOM, that a blank form cannot be submitted,
that the theme toggle works, and that an unreachable backend produces an
actionable message.

---

## Known limitations

Stated plainly rather than hidden:

- **The training set is authored, not sampled from real published news.** It is
  labelled consistently and validated for duplicates, but it is not a citable
  corpus. Sentiment could alternatively be trained on `pythainlp/wisesight_sentiment`
  (~26k real Thai social-media rows) for benchmark-comparable figures.
- **Neutral is the weakest sentiment class** (F1 0.473) with only 152 training
  examples against 345 positive. The class imbalance is compensated with
  balanced class weights, but more neutral data is the real fix.
- **`economy` / `business` / `technology` confuse each other** (F1 0.45–0.57).
  Their vocabulary genuinely overlaps.
- **Topic confidence is lower on chat than on news** (≈0.41 vs ≈0.73). The model
  is trained on news prose, so lower confidence on out-of-domain chat is correct
  calibration rather than a defect — the labels themselves stay sensible.
- **The corpus skews to sports.** Three of six collected streams are sports, and
  sports chat is far denser than news chat, so the dashboard is sports-heavy.
  Collect more news streams with `collect.py` to rebalance.
- **`yt-dlp` depends on YouTube's internal API** and can break when YouTube
  changes. The committed snapshots and the `file` collector mean the project
  still runs when that happens.
- **NER is rule-based**, so it recognises people, places and organisations only
  where Thai marks them with a lead-in word (นาย, จังหวัด, กระทรวง).
  `NLP_NER_BACKEND=pythainlp` switches to a CRF model, which needs a download.

---

## Licence and attribution

Coursework project. Collected chat remains the property of its authors and is
included only for academic analysis. PyThaiNLP, scikit-learn, FastAPI, React,
Recharts and yt-dlp are used under their respective licences.

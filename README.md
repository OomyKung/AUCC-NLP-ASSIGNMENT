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

Topic and news sentiment are measured on a stratified held-out split of the news
dataset (20%, seed 42) that the models never saw; `python evaluate.py` regenerates
those. Chat sentiment is measured on a different corpus's official test split, so
it is reported separately below rather than mixed into the same rows.

These are the models that actually serve requests. The Evaluation page marks the
active one and opens on it, so the page cannot describe a model the API is not
running.

| Task | Serving model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **Topic** (15 classes) | blend (TF-IDF + gazetteer) | 0.767 | 0.786 | 0.767 | **0.763** |
| **Sentiment** (3 classes) | TF-IDF + logistic regression | 0.753 | 0.724 | 0.689 | **0.698** |
| **Chat sentiment** (3 classes) | TF-IDF trained on Wisesight | 0.710 | 0.668 | 0.678 | **0.673** |

Random guessing would score 0.067, 0.333 and 0.333 respectively. Chat sentiment
is measured on a different corpus — see *Chat sentiment* below.

### How these numbers were obtained

Every choice — hyperparameters, blend weight, feature configuration — was made by
**5-fold cross-validation on the training split only**. The held-out set is scored
once, for reporting. This matters more than any individual figure below, and one
case shows why: a character-only feature configuration scores **0.787 accuracy /
0.743 macro-F1** on the sentiment held-out set, comfortably the best number in
this README. Cross-validation ranks it *below* word+char (0.694 vs 0.708), and the
fold standard deviation is ±0.034. The held-out result is selection noise on 150
rows, so it is not used and not claimed. Selecting on the test set would have
bought a 0.787 headline and a model that was no better.

### Trained models vs. rule-based baselines

Both are scored on the **same held-out rows**. This is the only fair comparison:
the gazetteer and lexicon were hand-written for this domain, so scoring them on
data they were designed around would flatter them.

| Task | Trained (macro-F1) | Baseline (macro-F1) | Gain |
|---|---|---|---|
| Topic | 0.725 (TF-IDF + logistic regression) | 0.680 (gazetteer) | **+0.045** |
| Sentiment | 0.698 (TF-IDF + logistic regression) | 0.422 (lexicon) | **+0.276** |

Two honest observations:

- **The hand-written gazetteer is genuinely competitive** for topic
  classification, and beats the trained model on `accident` and `education`,
  whose vocabulary is unambiguous. Training bought far less here than for
  sentiment.
- **Training was decisive for sentiment** (+0.276), where a word-list approach
  cannot capture context.

### Blending the rule-based and trained models

Because the two make *different* mistakes, averaging their probability
distributions helps — `P = α·P_model + (1−α)·P_rules`, with α fitted on
out-of-fold training predictions.

| Topic (15 classes) | CV macro-F1 | Held-out macro-F1 | Held-out accuracy |
|---|---|---|---|
| TF-IDF alone | 0.720 | 0.725 | 0.727 |
| **Blend, α = 0.85** | **0.740** | **0.763** | **0.767** |
| Stacking (LogReg meta-learner) | 0.707 | 0.684 | 0.693 |

On the 150 held-out topic rows, 18 are correct only under the rules, 26 only
under the model, and 23 under neither — so the best *any* combination of these two
could reach is **0.847 accuracy**. The blend captures a little under half of that
available headroom.

Two negative results worth recording:

- **Stacking is worse than the plain model.** A logistic-regression meta-learner
  over both probability vectors has 30 parameters to fit on 597 rows, and
  overfits (0.707 vs 0.720 CV).
- **Sentiment does not benefit from blending at all.** The same fitting procedure
  chose **α = 1.00** — the trained model alone. Once its hyperparameters were
  properly tuned it had absorbed everything the lexicon contributed, so the
  shipped sentiment backend is `sklearn`, not `blend`. A blend that adds a
  component doing no work would be decoration.

### Chat sentiment: the domain gap, measured in both directions

Per-message chat sentiment is a **separate model** from news sentiment. The reason
is measured, not assumed. Scored on the Wisesight corpus's official test split
(2,614 real Thai social-media messages):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **TF-IDF trained on Wisesight** (serving) | **0.710** | **0.673** |
| Majority class (always neutral) | 0.556 | 0.238 |
| Thai polarity lexicon (rules) | 0.538 | 0.436 |
| News-trained TF-IDF (cross-domain) | 0.319 | 0.319 |

And the mirror image — the Wisesight-trained model on the news held-out split:
**0.287 accuracy**. Each model collapses on the other's domain, which is why the
registry keeps two and routes by stage rather than sharing one.

This replaced the rule-based lexicon that previously served chat, lifting macro-F1
from **0.436 to 0.673 (+0.237)** on the same 2,614 rows.

### Does a transformer help? No, not at this data size

WangchanBERTa is the standard Thai pre-trained language model, and the character
n-gram ablation suggested subword modelling should suit Thai. It was fine-tuned
on the same 597 training rows and scored on the same 150 held-out rows as every
other model.

| Sentiment (3 classes) | Accuracy | Macro-F1 |
|---|---|---|
| **TF-IDF + logistic regression** | **0.753** | **0.698** |
| WangchanBERTa fine-tuned | 0.547 | 0.531 |
| Lexicon (rules, no training) | 0.427 | 0.422 |
| Off-the-shelf Thai sentiment model | 0.253 | 0.200 |

| Topic (15 classes) | Accuracy | Macro-F1 |
|---|---|---|
| **Blend (TF-IDF + gazetteer)** | **0.767** | **0.763** |
| TF-IDF + logistic regression | 0.727 | 0.725 |
| Gazetteer (rules, no training) | 0.680 | 0.680 |
| WangchanBERTa fine-tuned | 0.560 | 0.540 |

TF-IDF beats the transformer by **+0.167 macro-F1 on sentiment** and **+0.185 on
topic** — a large margin on two independent tasks with very different class
counts, which is what makes the result convincing rather than a fluke.

On topic classification a **hand-written word list beats a fine-tuned 110M-parameter
language model** (0.680 vs 0.540).

It is not one unlucky run. Three sentiment settings were tried and all lost:

| Epochs | LR | Batch | Class weights | Macro-F1 |
|---|---|---|---|---|
| 3 | 2e-5 | 8 | yes | **0.531** |
| 4 | 2e-5 | 16 | no | 0.498 |
| 10 | 3e-5 | 16 | yes | 0.346 |

Training *longer* made it worse: at 10 epochs the model collapsed to predicting
`negative` for 121 of 150 rows. The failure is qualitative as well as numerical —
the fine-tuned model calls "ทีมชาติไทยคว้าชัยชนะอย่างงดงาม" (*Thailand won
gloriously*) **negative**, and "มีผู้เสียชีวิต 3 ราย" (*three people died*)
**positive**.

**Why:** 597 examples is far too little to fine-tune a 110M-parameter model,
whereas TF-IDF over character n-grams is well suited to exactly this regime. The
practical lesson is that at this data size, more data would help more than a
bigger model.

The off-the-shelf model is a second, independent instance of the domain-mismatch
result: trained on Wisesight social-media text, it predicts `neutral` for **143
of 150** news rows. Comparing predicted distributions against the truth
(69 positive / 31 neutral / 50 negative) makes each failure mode legible in a way
the headline metric does not:

| Model | Predicted pos / neu / neg |
|---|---|
| TF-IDF | 79 / 24 / 47 — well calibrated |
| WangchanBERTa fine-tuned | 47 / 32 / 71 |
| Lexicon | 49 / 69 / 32 — over-predicts neutral |
| Off-the-shelf | 1 / **143** / 6 — collapsed |

Reproduce with `python train_transformer.py` then `python compare_models.py`.
The transformer path needs `torch`, `transformers`, `sentencepiece` **and**
`protobuf`; without protobuf, transformers routes WangchanBERTa's SentencePiece
vocabulary to a TikToken converter and fails with a misleading error.

### Ablations

| Experiment | Configuration | Macro-F1 |
|---|---|---|
| **Aggregation** | whole document | **0.725** |
| | per-sentence + majority vote | 0.675 |
| **Features (sentiment)** | character n-grams only | **0.697** |
| | word + character | 0.682 |
| | word only | 0.613 |
| **Negation protection** | negators removed | **0.625** |
| | negators protected | 0.613 |

Aggregating text before classification gains **+0.051 macro-F1**, which is the
evidence for grouping chat messages into windows rather than classifying each
one alone.

Character n-grams alone beat word features by **+0.069** on sentiment. Thai is
written without spaces, so tokenisation is itself a model that can fail; a
character view does not depend on it being right.

Protecting negators from stopword removal did **not** help the trained
classifier (−0.012, within noise on 150 rows). The token-level effect is real —
`ไม่ดี` reduces to `['ดี']` when unprotected — and it matters for the lexicon
backend, where negation is handled explicitly. But character n-grams already
capture `ไม่ดี` as a character sequence, so the word-level protection is
redundant once they are present. Reported as measured rather than as assumed.

Reproduce with `python ablation.py`.

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

### Privacy: this is real people's data

The chat is real, written by 5,706 real accounts, and much of it is political
opinion. This repository is public, so no handle is published. `anonymise.py`
replaces every one with a stable keyed hash (blake2s, 4-byte digest — short
enough to read, long enough that the expected collision count across 5,706
handles is ~0.004, because a collision would merge two speakers).

Handles appear in **two** places, and the second is the one that is easy to miss:

| Location | Example | Becomes |
|---|---|---|
| `author` field | `@thairathnews` | `ผู้ชม #3b9ebd90` |
| `@mention` inside message text | `@thairathnews ครับ` | `@viewer-3b9ebd90 ครับ` |

An earlier version of the script handled only the author field, which left 98
raw handles inside message text — viewers replying to each other by name. Both
are now covered, keyed identically, so a mention of someone who also posts
resolves to that poster's pseudonym and the reply structure of the conversation
survives.

Two properties worth stating precisely:

- **No analysis result depends on this.** The author field never reaches the
  pipeline, and `clean_text` strips `@[\w.\-]+` before tokenisation, so the raw
  handle was never a feature. The mention pseudonym is deliberately ASCII so the
  *same* regex removes it completely — verified to produce identical cleaned text
  and identical tokens. A Thai pseudonym would not work here: `\w` stops at Thai
  combining marks and would leave fragments like `ชม` to pollute keywords.
- **It is enforced, not just documented.** `tests/test_privacy.py` scans the
  committed snapshots for raw handles, emails and phone-shaped digit runs, and
  fails the build if any appear. The digit check requires four distinct digits in
  a run, because `5555555555` is Thai laughter, not a phone number.

```powershell
python anonymise.py --check     # report what would change
python anonymise.py             # rewrite snapshots and the database
```

The transformation is idempotent, so it is safe to run before every commit.

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

# Model selection and analysis (see Results)
python tune.py                       # 144-point grid search, 5-fold CV
python tune.py --task topic --quick  # smaller grid, one task
python ensemble.py                   # fit + compare blend and stacking
python ablation.py                   # three ablation studies
python compare_models.py             # every approach on one split

# Chat-domain sentiment (needs requirements-training.txt)
python train_chat_sentiment.py       # train on the Wisesight corpus
python train_chat_sentiment.py --keep-questions   # 4-class-comparable run

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

Optional extras, only needed to **retrain** (never to run the app — the fitted
models are committed):

```powershell
pip install -r requirements-training.txt   # corpus download + transformer
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
NLP_TOKENIZER=newmm                    # newmm | newmm-safe | longest | mm
NLP_TOPIC_BACKEND=blend                # sklearn | blend | transformer
NLP_SENTIMENT_BACKEND=sklearn          # sklearn | blend | lexicon | transformer
NLP_CHAT_SENTIMENT_BACKEND=wisesight   # wisesight | sklearn | lexicon | transformer
NLP_SUMMARIZER_BACKEND=extractive      # extractive | llm
NLP_NER_BACKEND=rules                  # rules | pythainlp

# Blend weights, fitted by ensemble.py on out-of-fold training predictions.
NLP_TOPIC_BLEND_ALPHA=0.85             # 1.0 = trained model alone, 0.0 = rules alone
NLP_SENTIMENT_BLEND_ALPHA=1.0          # fitted to 1.0: the lexicon adds nothing here
```

The values above are the defaults, and each is the measured choice rather than a
preference — see [Results](#results). `blend` is the default for topic because it
wins; it is *not* the default for sentiment because the fitted weight came out at
1.0, making it identical to `sklearn` with an extra moving part.

To add a new model: write one adapter class satisfying the relevant Protocol,
register it in `registry.py`, and select it in `.env`. Then
`POST /api/ingest/reanalyse` to re-run the corpus through it.

The registry is **capability aware**: if a configured backend's trained artefact
is missing, it falls back to the rule-based baseline and records why.
`/api/pipeline` reports what is *actually* serving, so the UI can label results
as a baseline honestly rather than implying a fitted model.

### Why chat sentiment uses a separate backend

The two domains are far enough apart that one model cannot serve both, and the
gap was measured in **both** directions: the news-trained model scores 0.319 on
the Wisesight test split, and a Wisesight-trained model scores 0.287 on the news
split. Each collapses on the other's data. That is why
`NLP_CHAT_SENTIMENT_BACKEND` exists as a separate setting.

Chat is served by `wisesight` — TF-IDF + logistic regression trained on ~23.5k
real human-labelled Thai social-media messages. It replaced the hand-built
lexicon, lifting macro-F1 from 0.436 to 0.673 on the same 2,614 test rows. The
lexicon is still the fallback when the artefact is missing, so a fresh clone
works either way.

To retrain it:

```powershell
pip install -r requirements-training.txt
python train_chat_sentiment.py
```

The corpus (CC0-1.0, `pythainlp/wisesight_sentiment`) downloads on demand and is
cached under `data/wisesight/`, which is gitignored — the repository redistributes
no third-party corpus. The **model** is committed, so the download is only needed
to retrain.

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
cd backend  && python -m pytest      # 242 tests
cd frontend && npm run test          # 13 rendering tests
```

Backend coverage includes Thai preprocessing and tokenisation, the YouTube
InnerTube chat parser (including malformed input), windowing bounds, idempotent
storage, every API endpoint with its error paths, model artefact loading
(missing, corrupt, version-mismatched), domain routing, and snapshot integrity.

The blend is tested against stub members so its arithmetic is asserted exactly,
including the case that matters most: when the two members disagree and the
*average* of their distributions picks a third label neither would have chosen.
Corpus-loader tests are skipped with a clear reason when the corpus has not been
downloaded, so a fresh clone reports skips rather than failures.

The frontend tests mount each real page against fixtures captured from the
running API. That matters because an HTTP 200 on an SPA route only proves
`index.html` was served — it says nothing about whether React rendered or threw.
They assert real values reach the DOM, that a blank form cannot be submitted,
that the theme toggle works, and that an unreachable backend produces an
actionable message.

---

## Known limitations

Stated plainly rather than hidden:

- **The news training set is authored, not sampled from real published news.**
  It is labelled consistently and validated for duplicates, but it is not a
  citable corpus, and it is the binding constraint on the topic and news-sentiment
  figures. 747 rows over 15 categories is ~40 training examples per class.
  *Chat* sentiment no longer has this problem: it is trained on the Wisesight
  corpus (CC0, ~23.5k real human-labelled Thai social-media messages) and
  reported on that corpus's official test split.
- **Neutral is the weakest sentiment class** (F1 0.491) with only 152 training
  examples against 345 positive. The class imbalance is compensated with
  balanced class weights, but more neutral data is the real fix.
- **A 144-point hyperparameter search moved topic by ~0.000 and sentiment by
  +0.029 CV macro-F1.** When exhaustive tuning cannot move a model, the limit is
  the data, not the configuration. The transformer result points the same way: a
  110M-parameter pre-trained Thai model scored *worse* than TF-IDF on both tasks,
  which is what happens when there is too little data to fine-tune on.
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

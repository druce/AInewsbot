# AInewsbot

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/druce/AInewsbot)

A Python pipeline for **AI news gathering → topic clustering → summarization → newsletter & podcast creation**. Used for [skynetandchill.com](https://www.skynetandchill.com/).


## Table of Contents

- [Purpose](#1-purpose)
- [Features](#features)
- [Quickstart](#quickstart)
- [Core Components](#2-core-components)
- [Data & Outputs](#3-data--outputs)
- [Dependencies & Setup](#4-dependencies--setup)
- [How to Run](#5-how-to-run)
- [Example Output](#example-output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)


**AInewsbot** is an end-to-end pipeline for **AI news gathering → topic clustering → summarization → AI newsletter & podcast creation**

(potentially other subjects by modifying source URLs and filters / search parameters / prompts)

- Used to help generate a daily newsletter at https://www.skynetandchill.com/

- Also generates a podcast using [podcastfy](https://github.com/souzatharsis/podcastfy), using a complex prompt to write a script and then perform text-to-speech.

[![A podcast created with podcastfy](https://img.youtube.com/vi/Fl0xP1Io72k/0.jpg)](https://www.youtube.com/shorts/AOVOOZQthNU)

Explore interactively in [AInewsbot_langgraph.ipynb](https://github.com/druce/AInewsbot/blob/main/AInewsbot_langgraph.ipynb).


## 1. Purpose

- Automatically gather AI-related news from many sites.
- Filter, cluster, and summarize articles by topic.
- Assemble a daily newsletter and generate a podcast.
- Human-in-the-loop review steps supported.


## Features

- Multi-source scraping and deduplication
- LLM-powered classification, summarization, clustering
- Newsletter and podcast generation
- SQLite-based history tracking


## Quickstart

1. Copy `dotenv.txt` to `.env` and add your `OPENAI_API_KEY`. (Code also supports Google and Claude, fairly straightforward to modify for models supported by LangChain. But current prompts heavily optimized for latest OpenAI models with improved structured JSON outputs, reasoning, will probably need a fair bit of tuning for other models.)

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. This version uses Firefox to download pages. Point to a `FIREFOX_PROFILE_PATH` in `ainb_const.py` to use your own organic profile. You can use the browser of your choice by editing the code and picking a different Playwright driver in `ainb_llm.py`, and pointing to the profile of your choice. You can launch an interactive browser in that profile and log in to various services, and the automation should be able to take advantage of any login cookies saved.

4. Edit start pages to download in `sources.yaml`. Of course, avoid scraping anything in violation of any site's `robots.txt` and terms of service. Use direct APIs to market data services that you subscribe to. Some low-cost APIs such as NewsAPI or Newscatcher are implemented in my [marketdata](https://github.com/druce/marketdata) repo.

5. Run the main script:

   ```bash
   python AInewsbot_langgraph.py
   ```

## 2. Core Components

### Orchestration
- `AInewsbot_langgraph.py`: Top-level orchestrator. Follows the workflow in the image below.
  - Fetch source pages specified in `sources.yaml` (and NewsAPI) → Extract & dedupe URLs → - Classify headlines as AI or not AI, filter previously seen
  - Scrape indivdual stories & summarize →  Embed & cluster topics, order by topic
  - Prompt LLM for newsletter → Polish / re-edit → Send email
- `AInewsbot.sh`: Shell wrapper so you can activate the Conda env and run the pipeline on a schedule

### Configuration & Prompts
- `sources.yaml`: List of news sources (URL, include/exclude regex, scrolling instructions, etc.).
- `ainb_const.py`: Paths, API keys (via `.env`), model settings,
- `ainb_prompts.py`: LLM prompts (classification, topic extraction, summarization etc.)

### Scraping
- `ainb_webscrape.py`:
  - Downloads “landing pages” using Playwright
  - Extracts story links
  - Downloads full-article HTML
  - Captures screenshots to potentially show during YouTube podcast

### Utilities & Storage
- `ainb_utilities.py`: Logging wrapper, file cleanup, SQLite insert/query of seen URLs, text normalization.
- `articles.db`: Tracks previously fetched URLs to avoid re-fetching and presenting previously discussed articles.

### LLM Integration
- `ainb_llm.py`: LangGraph wrappers around ChatOpenAI.
  - Take a current dataframe of news stories (~100 per day) and apply a prompt to each row asynchronously (i.e. with 100 parallel LLM calls for classificaation, topic extraction, filtering, summarization)
  - Structured JSON classification (like, is it AI-related?)
  - Topic extraction
  - Summary generation
- `ainb_prompts.py`: Prompts used when calling LLM.

### Notebooks & Experiments
- `AInewsbot_langgraph.ipynb`: Interactive pipeline runner, topic clustering, visualizations.
- `AInewsbot_test_llms.ipynb`: test best way to call various LLMs

---

## 3. Data & Outputs

- `htmldata/`: Raw downloaded landing-page HTML
- `htmlpages/`: Individual story HTMLs
- `screenshots/`: Screenshots for podcast
- `data/transcripts/`, `data/audio/`: Podcast transcripts and audio (via `podcastfy`)
- `summary.md`, `bullets.md`, `bullets.html`: Drafts of the newsletter
- `reducer.pkl`: Pickled UMAP dimensionality reduction model

---

## 4. Dependencies & Setup

- Python 3.x (see `requirements.txt`)
- Playwright
- LangChain and OpenAI LLMs (other LLMs possible)
- See `requirements.txt` for additional requirements

**Environment Variables:**

- Use the provided `dotenv.txt` as a scaffold for your `.env` file.
- For basic operation, only `OPENAI_API_KEY` is required.
- Other supported keys (optional, for advanced features and previous experiments): News APIs, LangSmith, Google Cloud, YouTube upload, Anthropic Claude, ElevenLabs, Perplexity, Pinecone, Reddit, etc.

Example `.env` (minimal):

```env
OPENAI_API_KEY=sk-...
```

---

## 5. How to Run

- For basic usage, run `python AInewsbot_langgraph.py`

```bash
$ python AInewsbot_langgraph.py --help

usage: AInewsbot_langgraph.py [-h] [-n] [-d BEFORE_DATE] [-b BROWSERS] [-e MAX_EDITS]

options:
  -h, --help            show this help message and exit
  -n, --nofetch         Disable web fetch, use existing HTML files in htmldata directory
  -d BEFORE_DATE, --before-date BEFORE_DATE
                        Force processing of articles before this date even if already processed (YYYY-MM-DD HH:MM:SS format)
  -b BROWSERS, --browsers BROWSERS
                        Number of browser instances to run in parallel (default: 4)
  -e MAX_EDITS, --max-edits MAX_EDITS
                        Maximum number of summary rewrites
```

- For advanced usage, schedule runs of `AInewsbot.sh`, customize sources in `sources.yaml`, change additional configs in `ainb_const.py`, or run interactively in `AInewsbot_langgraph.ipynb`.

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.png?raw=true)

---

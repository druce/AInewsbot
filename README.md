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


AInewsbot is a Python pipeline and notebook suite for automating the discovery, clustering, and summarization of the latest AI news. It assembles a daily newsletter and can even generate a podcast script and audio, making it ideal for content creators and AI enthusiasts.

- Powers the daily newsletter at [skynetandchill.com](https://www.skynetandchill.com/).
- Generates podcasts using [podcastfy](https://github.com/souzatharsis/podcastfy), leveraging LLMs for scriptwriting and text-to-speech.
- Easily adaptable to other topics by changing source URLs and keywords.

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

1. Copy `dotenv.txt` to `.env` and add your `OPENAI_API_KEY`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Edit `GECKODRIVER_PATH` and `FIREFOX_PROFILE_PATH` in `ainb_const.py`.
4. Run the main script:

   ```bash
   python AInewsbot_langgraph.py
   ```

## 2. Core Components

### Orchestration
- `AInewsbot_langgraph.py`: The "main" orchestrator. Follows the workflow in the image below.
  - Fetch source pages specified in `sources.yaml` (and NewsAPI) → Extract & dedupe URLs → - Classify headlines as AI or not AI, filter previously seen
  - Scrape indivdual stories & summarize →  Embed & cluster topics, order by topic
  - Prompt LLM for newsletter → Optional re-edit → Send email (via `smtplib`)
- `AInewsbot.sh`: Shell wrapper to activate the Conda env and launch the pipeline on a schedule

### Configuration & Prompts
- `sources.yaml`: List of news sources (URL, include/exclude regex, scrolling instructions, etc.).
- `ainb_const.py`: Paths, API keys (via `.env`), model settings, LLM prompts (classification, topic extraction, summarization).

### Scraping
- `ainb_webscrape.py`:
  - Downloads “landing pages” using Selenium (Firefox/geckodriver)
  - Extracts story links
  - Downloads full-article HTML
  - Captures screenshots to potentially show during YouTube podcast

### Utilities & Storage
- `ainb_utilities.py`: Logging wrapper, file cleanup, SQLite insert/query of seen URLs, text normalization.
- `articles.db`: Tracks previously fetched URLs to avoid re-fetching and presenting previously discussed articles.

### LLM Integration
- `ainb_llm.py`: LangChain wrappers around ChatOpenAI.
  - Take a current dataframe of news stories (~100 per day) and apply a prompt to each asynchronously (i.e. with 100 parallel LLM calls)
  - Structured JSON classification (like, is it AI-related?)
  - Topic extraction
  - Summary generation

### Notebooks & Experiments
- `AInewsbot_langgraph.ipynb`: Interactive pipeline runner, topic clustering visualizations.
- `AInewsbot_test_llms.ipynb`: test best way to call various LLMs
- `reducer.pkl`, `AIdf.pkl`: Clustering tests & artifacts.

---

## 3. Data & Outputs

- `htmldata/`: Raw downloaded landing-page HTML
- `htmlpages/`: Individual story HTMLs
- `screenshots/`: Screenshots for podcast
- `data/transcripts/`, `data/audio/`: Podcast transcripts and audio (via `podcastfy`)
- `summary.md`, `bullets.md`, `bullets.html`: Drafts of the newsletter
- `my_dict.pkl`, `AIdf.pkl`, `reducer.pkl`: Pickled models (TF-IDF, dimensionality reduction)

---

## 4. Dependencies & Setup

- Python 3.x (see `requirements.txt`)
- Selenium + geckodriver + Firefox (with a custom profile)
- LangChain and OpenAI LLMs (other LLMs possible)
- See `requirements.txt` for additional requirements

**Environment Variables:**

- Use the provided `dotenv.txt` as a scaffold for your `.env` file.
- For basic operation, only `OPENAI_API_KEY` is required.
- Other supported keys (optional, for advanced features): News APIs, LangSmith, Google Cloud, YouTube upload, Anthropic Claude, ElevenLabs, Perplexity, Pinecone, Reddit, etc.

Example `.env` (minimal):

```env
OPENAI_API_KEY=sk-...
```

---

## 5. How to Run

- For advanced usage, schedule runs of `AInewsbot.sh`, customize sources in `sources.yaml`, or run interactively in `AInewsbot_langgraph.ipynb`.

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.png?raw=true)

---

# AInewsbot

A Python notebook/script to help find the latest news about AI

**AInewsbot** is an end-to-end pipeline for **AI news gathering → topic clustering → summarization → AI newsletter & podcast creation**

(potentially other subjects by modifying source URLs and search keywords)

- Used to help generate a daily newsletter at https://www.skynetandchill.com/

- Also generates a podcast using [podcastfy](https://github.com/souzatharsis/podcastfy), using a complex prompt to write a script and then perform text-to-speech.

[![A podcast created with podcastfy](https://img.youtube.com/vi/Fl0xP1Io72k/0.jpg)](https://www.youtube.com/shorts/AOVOOZQthNU)

[AInewsbot_langraph.ipynb](https://github.com/druce/AInewsbot/blob/main/AInewsbot_langgraph.ipynb)

---

## 1. Purpose

- Automatically gather AI-related news from many sites.
- Filter and cluster the headlines by topic, scrape and summarize each article.
- Assemble a daily newsletter (and even a podcast).
- Built using:
  - OpenAI + other LLMs via LangChain
  - Selenium / BeautifulSoup scraping
  - SciPy clustering
  - SQLite history tracking
  - Human-in-the-loop review steps

---

## 2. Core Components

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
  - Structured JSON classification (like, is it AI-related?)
  - Topic extraction
  - Summary generation
  - Prompt-token budgeting

### Orchestration
- `AInewsbot_langgraph.py`: The "main" orchestrator.
  - Fetch source pages specified in `sources.yaml` (and NewsAPI) → Extract & dedupe URLs → - Classify headlines as AI or not AI, filter previously seen
  - Scrape indivdual stories & summarize →  Embed & cluster topics, order by topic
  - Prompt LLM for newsletter → Optional re-edit → Send email (via `smtplib`)
- `AInewsbot.sh`: Shell wrapper to activate the Conda env and launch the pipeline on a schedule

### 🧪 Notebooks & Experiments
- `AInewsbot_langgraph.ipynb`: Interactive pipeline runner, topic clustering visualizations.
- `AInewsbot_test_llms.ipynb`: test various LLMs
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

- Python 3.x
- Selenium + geckodriver + Firefox (with a custom profile)
- LangChain and OpenAI LLMs. LangChain is cross-platform and easy to point to different LLM vendors. I have run it in the past with Gemini. Currently it uses 4.1-mini, 4.1, and o3 for simple, complex, and reasoning prompts. Since especially 4.1 is well-designed for agentic workflows prompts are optimized for OpenAI and would take some optimization to run properly using other vendors.
- Libraries:
	- see `requirements.txt`
	- scraping:`beautifulsoup4`, `requests`, `fake-useragent`, `trafilatura`
	- clustering: `pandas`, `numpy`, `scipy`,
	- LLM: `langchain_openai`, `tiktoken`, `pydantic` (json mode), `tenacity` (retry)
	- `dotenv` for `.env` loading
  - `nbformat`, `jupyter` for notebook execution

---

## 5. How to Run

1. Populate your `.env` file with required API keys.
2. Edit `GECKODRIVER_PATH` and `FIREFOX_PROFILE_PATH` in `ainb_const.py`.
3. Run the main script:

   ```bash
   python AInewsbot_langgraph.py

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.png?raw=true)


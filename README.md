# AInewsbot
A Python notebook/script to help find the latest news about AI (potentially other subjects by modifying source URLs and search keywords)

[AInewsbot_langraph.ipynb](https://github.com/druce/AInewsbot/blob/main/AInewsbot_langgraph.ipynb)

- Saves a list of HTML files from sources.yaml (tech news sites)
- Extracts URLs for the news stories from the HTML pages based on a regexp
- Also searches for AI news on a bunch of sites using [newscatcher](https://www.newscatcherapi.com/) (see my [marketdata](https://github.com/druce/marketdata) repo for some other news APIs)
- Filters URLs to remove duplicates, articles seen before (using a SQLite history database), and non-AI-related articles (using a ChatGPT prompt)
- Performs headline topic analysis and sorts by topic to help the AI structure the response by topic (use DBSCAN clustering on dimensionality-reduced headline embeddings)
- Scrapes and summarizes individual articles
- Have o1-preview compose the newsletter using a prompt + article summaries + topic keywords
- Re-edits using an an additional prompt to improve the output (can iterate until satisfied)
- Emails the summary newsletter
- Requires Firefox / geckodriver, doesn't run headless, needs interactive session
- Uses human-in-the loop to ensure AI doesn't go off the rails. Human should check after downloading files that all are present, if any triggered a bot block, download those manually.
- Human should review and edit categories proposed
- Usually the newsletter composed using OpenAI o1-preview is pretty good as a first iteration, sometimes it's a bit hit-or-miss and need to retry. 
- Used to help generate a daily newsletter at https://www.skynetandchill.com/
- Also generates a podcast using podcastfy, which uses a complex prompt to write a script and then perform text-to-speech. Did some manual editing to incorporate images, set up image timings, upload to YouTube, could possibly be automated but still work in progress.

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.jpeg?raw=true)

## Podcast generated with podcastfy.ai

[![A podcast created with podcastfy](https://img.youtube.com/vi/Fl0xP1Io72k/0.jpg)](https://www.youtube.com/shorts/AOVOOZQthNU)

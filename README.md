# AInewsbot
A Python notebook/script to help find the latest news about AI (or other subjects)

[AInewsbot_langraph.ipynb](https://github.com/druce/AInewsbot/blob/main/AInewsbot_langgraph.ipynb)

- Save a list of HTML files from sources.yaml (tech news sites)
- Extract URLs for the news stories based on a regexp
- Filter URLs to remove duplicates, articles seen before (using a SQLite history), and non-AI articles (using a ChatGPT prompt)
- Perform headline topic analysis and sort by topic to help the AI structure the response by topic
- Scrape and summarize individual articles
- Compose, edit, and email the summary newsletter
- Requires Firefox / geckodriver, doesn't run headless, needs interactive session
- Human should check after downloading files that all are present, if any triggered a bot block, download those manually.
- Human should review and edit categories proposed
- Usually the newsletter composed using OpenAI o1-preview is pretty good as a first iteration, sometimes it's hit-or-miss. The summary bullets of all the day's stories also give a pretty comprehensive overview.

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.jpeg?raw=true)

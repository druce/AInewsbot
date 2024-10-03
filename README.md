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
- Human should review and edit categories proposed, should check after downloading files that all are present, if any got an are-you-a-bot block, download manually.
- Usually the newsletter composed using OpenAI o1-preview is pretty good as a first iteration, sometimes hit-or-miss. But reading the summary bullets gives a pretty comprehensive overview.

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.jpeg?raw=true)

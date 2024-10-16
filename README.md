# AInewsbot
A Python notebook/script to help find the latest news about AI (potentially other subjects my modifying source URLs and search keywords)

[AInewsbot_langraph.ipynb](https://github.com/druce/AInewsbot/blob/main/AInewsbot_langgraph.ipynb)

- Save a list of HTML files from sources.yaml (tech news sites)
- Extract URLs for the news stories from the HTML pages based on a regexp
- Also searched for AI news on a bunch of sites using [newscatcher](https://www.newscatcherapi.com/) (see my marketdata repo for some other news APIs)
- Filter URLs to remove duplicates, articles seen before (using a SQLite history database), and non-AI-related articles (using a ChatGPT prompt)
- Perform headline topic analysis and sort by topic to help the AI structure the response by topic (use DBSCAN clustering on dimensionality-reduced headline embeddings)
- Scrape and summarize individual articles
- Have o1-preview compose the newsletter using a prompt + article summaries + topic keywords
- Re-edit using an an additional prompt to improve the output (can iterate until satisfied)
- Email the summary newsletter
- Requires Firefox / geckodriver, doesn't run headless, needs interactive session
- Uses human-in-the loop to ensure AI doesn't go off the rails. Human should check after downloading files that all are present, if any triggered a bot block, download those manually.
- Human should review and edit categories proposed
- Usually the newsletter composed using OpenAI o1-preview is pretty good as a first iteration, sometimes it's a bit hit-or-miss and need to retry. 

![flowchart](https://github.com/druce/AInewsbot/blob/main/graph.jpeg?raw=true)

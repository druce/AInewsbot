# AInewsbot
A Python notebook to help find the latest news about AI

AInewsbot.ipynb

- Open URLs of news sites specififed in `sources` dict using Selenium and Firefox
- Save HTML of each URL in htmldata directory
- Extract URLs from all files, create a pandas dataframe with url, title, src
- Use ChatGPT to filter only AI-related headlines
- Use SQLite to filter headlines previously seen

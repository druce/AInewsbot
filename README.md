# AInewsbot
A Python notebook/script to help find the latest news about AI (or other subjects)

AInewsbot.ipynb

- Open URLs of news sites specififed in sources.yaml using Selenium and Firefox
- Save HTML of each URL in htmldata directory
- Extract URLs from all files, create a pandas dataframe with url, title, src
- Use ChatGPT to filter only AI-related headlines
- Use SQLite to store headlines previously seen and not show them again
- send a mail with the latest headlines
- requires Firefox / geckodriver, doesn't run headless currently, needs interactive session

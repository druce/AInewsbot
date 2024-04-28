#!/Users/drucev/anaconda3/envs/ainewsbot/bin/python

# AInewsbot.py
#
# - Open URLs of news sites specififed in `sources.yaml` dict using Selenium and Firefox
# - Save HTML of each URL in htmldata directory
# - Extract URLs from all files, create a pandas dataframe with url, title, src
# - Use ChatGPT to filter only AI-related headlines by sending a prompt and formatted table of headlines
# - Use SQLite to filter headlines previously seen
#
# Alternative manual workflow to get HTML files if necessary
# - Use Chrome, open e.g. Tech News bookmark folder, right-click and open all bookmarks in new window
# - on Google News, make sure switch to AI tab
# - on Google News, Feedly, Reddit, scroll to additional pages as desired
# - Use SingleFile extension, 'save all tabs'
# - Move files to htmldata directory
# - Run lower part of notebook to process the data
#
############################################################################################################

import os
from datetime import datetime
import yaml
import dotenv
import sqlite3

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

# import openai
from openai import OpenAI

from ainb_const import (DOWNLOAD_DIR,
                        SOURCECONFIG, PROMPT)
from ainb_utilities import log, delete_files, filter_unseen_urls_db, insert_article, unicode_to_ascii, nearest_neighbor_sort
from ainb_webscrape import init_browser, get_file, parse_file
from ainb_llm import paginate_df, process_pages
############################################################################################################
# initialize configs
############################################################################################################

# load secrets, credentials
dotenv.load_dotenv()

#  load sources to scrape from sources.yaml
with open(SOURCECONFIG, "r") as stream:
    try:
        sources = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

log("Load sources")
sources_reverse = {}
for k, v in sources.items():
    log(f"{k} -> {v['url']} -> {v['title']}.html")
    v['sourcename'] = k
    # map filename (title) to source name
    sources_reverse['title'] = k


############################################################################################################
# Get HTML files
############################################################################################################
# empty download directory
delete_files(DOWNLOAD_DIR)

# launch browser via selenium driver
driver = init_browser()

# save each file specified from sources
log("Saving HTML files")
saved_pages = []
for sourcename, sourcedict in sources.items():
    log(f'Processing {sourcename}')
    sourcefile = get_file(sourcedict, driver=driver)
    saved_pages.append((sourcename, sourcefile))

# Close the browser
log("Quit webdriver")
driver.quit()
# finished downloading files

############################################################################################################
# Parse news URLs and titles from downloaded HTML files
############################################################################################################

all_urls = []

log("parsing html files")
for sourcename, filename in saved_pages:
    log(f"{sourcename}", "parse loop")
    sources[sourcename]["latest"] = filename
    srcurl = sources[sourcename]['url']
    exclude_pattern = sources[sourcename].get("exclude")
    include_pattern = sources[sourcename].get("include")
    minlength = sources[sourcename].get("minlength", 28) | 28

    links = parse_file(filename,
                       srcurl,
                       exclude_pattern,
                       include_pattern,
                       minlength=minlength
                       )
    log(f"{len(links)} links found", "parse loop")

    all_urls.extend(links)

log(f"found {len(all_urls)} links", "parse loop")

# make a pandas dataframe of all the links found
orig_df = (
    pd.DataFrame(all_urls)
    .groupby("url")
    .first()
    .reset_index()
    .sort_values("src")[["src", "title", "url"]]
    .reset_index(drop=True)
    .reset_index(drop=False)
    .rename(columns={"index": "id"})
)

############################################################################################################
# Filter URLs, ignore previously sent, check if AI-related using ChatGPT
############################################################################################################
filtered_df = filter_unseen_urls_db(orig_df)
# # Filter AI-related headlines using a prompt to OpenAI

# make pages that fit in a reasonably sized prompt
pages = paginate_df(filtered_df)

client = OpenAI()
enriched_urls = process_pages(client, PROMPT, pages)

enriched_df = pd.DataFrame(enriched_urls)
enriched_df.head()

log("isAI", len(enriched_df.loc[enriched_df["isAI"]]))
log("not isAI", len(enriched_df.loc[~enriched_df["isAI"]]))

merged_df = pd.merge(filtered_df, enriched_df, on="id", how="outer")
merged_df['date'] = datetime.now().date()
merged_df.head()

# should be empty, shouldn't get back rows that don't match to existing
log(f"Unmatched response rows: {len(merged_df.loc[merged_df['src'].isna()])}")
# should be empty, should get back all rows from orig
log(f"Unmatched source rows: {len(merged_df.loc[merged_df['isAI'].isna()])}")

# update SQLite database with all seen articles
conn = sqlite3.connect('articles.db')
cursor = conn.cursor()
for row in merged_df.itertuples():
    insert_article(conn, cursor, row.src, row.title,
                   row.url, row.isAI, row.date)

AIdf = merged_df.loc[merged_df["isAI"]].reset_index()
log(f"Found {len(AIdf)} AI headlines")

# dedupe identical headlines
AIdf['title_clean'] = AIdf['title'].apply(unicode_to_ascii)
AIdf['title_clean'] = AIdf['title_clean'].map(lambda s: "".join(s.split()))
AIdf = AIdf.sort_values("src") \
    .groupby("title_clean") \
    .first() \
    .reset_index()
log(f"Found {len(AIdf)} unique AI headlines")


############################################################################################################
# Save ordered list and send email
############################################################################################################

# Attempt to order by topic by getting embeddings and solving a traveling salesman problem
log(f"Fetching embeddings for {len(AIdf)} headlines")
embedding_model = 'text-embedding-3-small'
response = client.embeddings.create(input=AIdf['title'].tolist(),
                                    model=embedding_model)
embedding_df = pd.DataFrame([e.dict()['embedding'] for e in response.data])
embedding_array = embedding_df.values

# find index of most central headline
centroid = embedding_array.mean(axis=0)
distances = np.linalg.norm(embedding_array - centroid, axis=1)
start_index = np.argmin(distances)

# Get the sorted indices and use them to sort the df
sorted_indices = nearest_neighbor_sort(embedding_array, start_index)
AIdf = AIdf.iloc[sorted_indices]

# create html message with formatted headlines
html_str = ""
for i, j in enumerate(sorted_indices):
    row = AIdf.iloc[j]
    log(f"[{i}. {row.title} - {row.src}]({row.url})")
    html_str += f'{i}.<a href="{row.url}">{row.title} - {row.src}</a><br />\n'

# send mail
log("Sending mail")
from_addr = os.getenv("GMAIL_USER")
to_addr = os.getenv("GMAIL_USER")
subject = 'AI news ' + datetime.now().strftime('%H:%M:%S')
body = f"""
<html>
    <head></head>
    <body>
    <div>
    {html_str}
    </div>
    </body>
</html>
"""

# Setup the MIME
message = MIMEMultipart()
message['From'] = os.getenv("GMAIL_USER")
message['To'] = os.getenv("GMAIL_USER")
message['Subject'] = subject
message.attach(MIMEText(body, 'html'))

# Create SMTP session
with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()  # Secure the connection
    server.login(os.getenv("GMAIL_USER"), os.getenv("GMAIL_PASSWORD"))
    text = message.as_string()
    server.sendmail(from_addr, to_addr, text)

log("Finished")

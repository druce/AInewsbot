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

# import pdb
import os
from datetime import datetime
import yaml
import dotenv
import sqlite3
import argparse
import asyncio

import multiprocessing

import markdown

import pandas as pd

# import openai
from openai import OpenAI

from ainb_const import (DOWNLOAD_DIR,
                        SOURCECONFIG, MODEL, FINAL_SUMMARY_PROMPT)
from ainb_utilities import log, delete_files, filter_unseen_urls_db, insert_article, unicode_to_ascii, agglomerative_cluster_sort, send_gmail
from ainb_webscrape import parse_file, process_source_queue_factory, process_url_queue_factory, launch_drivers
from ainb_llm import paginate_df, fetch_pages, fetch_all_summaries

############################################################################################################
# initialize configs
############################################################################################################


# load secrets, credentials
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nofetch', action='store_true',
                    help='Disable web fetch, use existing HTML files in htmldata directory')
args = parser.parse_args()

# Set the boolean flag based on the command line argument
disable_web_fetch = args.nofetch
enable_web_fetch = not disable_web_fetch

#  load sources to scrape from sources.yaml
with open(SOURCECONFIG, "r") as stream:
    try:
        sources = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

log(f"Load {len(sources)} sources")
sources_reverse = {}
for k, v in sources.items():
    log(f"{k} -> {v['url']} -> {v['title']}.html")
    v['sourcename'] = k
    # map filename (title) to source name
    sources_reverse[v['title']] = k

############################################################################################################
# Get HTML files
############################################################################################################
if disable_web_fetch:
    # get list of files in htmldata directory
    # List all paths in the directory matching today's date
    log(f"Web fetch disabled, using existing files in {DOWNLOAD_DIR}")
    nfiles = 50
    files = [os.path.join(DOWNLOAD_DIR, file)
             for file in os.listdir(DOWNLOAD_DIR)]

    # Get the current date
    datestr = datetime.now().strftime("%m_%d_%Y")

    # filter files only
    files = [file for file in files if os.path.isfile(file)]

    # Sort files by modification time and take top 50
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    file = files[:nfiles]

    # filter files by with today's date ending in .html
    files = [
        file for file in files if datestr in file and file.endswith(".html")]
    log(len(files))
    for file in files:
        log(file)

    saved_pages = []
    for file in files:
        filename = os.path.basename(file)
        # locate date like '01_14_2024' in filename
        position = filename.find(" (" + datestr)
        basename = filename[:position]
        # match to source name
        sourcename = sources_reverse.get(basename)
        if sourcename is None:
            log(f"Skipping {basename}, no sourcename metadata")
            continue
        sources[sourcename]['latest'] = file
        saved_pages.append((sourcename, file))
else:
    # delete existing files and fetch all using selenium
    log(f"Delete existing HTML files in {DOWNLOAD_DIR}")
    delete_files(DOWNLOAD_DIR)

    log("Fetching HTML files")

    # Create a queue for multiprocessing and populate it with 20 numbers
    queue = multiprocessing.Queue()
    for item in sources.values():
        queue.put(item)

    callable = process_source_queue_factory(queue)

    NBROWSERS = 3
    saved_pages = launch_drivers(NBROWSERS, callable)

    # update sources with latest filename
    for sourcename, filename in saved_pages:
        sources[sourcename]['latest'] = filename

############################################################################################################
# Parse news URLs and titles from downloaded HTML files
############################################################################################################

# Parse news URLs and titles from downloaded HTML files

log("parsing html files")
all_urls = []
for sourcename, filename in saved_pages:
    print(sourcename, '->', filename, flush=True)
    log(f"{sourcename}", "parse loop")
    links = parse_file(sources[sourcename])
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
orig_df.head()

############################################################################################################
# Filter URLs, ignore previously sent, check if AI-related using ChatGPT
############################################################################################################
filtered_df = filter_unseen_urls_db(orig_df)
# # Filter AI-related headlines using a prompt to OpenAI

# make pages that fit in a reasonably sized prompt
pages = paginate_df(filtered_df)

enriched_urls = asyncio.run(fetch_pages(pages))
# enriched_urls = process_pages(client, PROMPT, pages)

enriched_df = pd.DataFrame(enriched_urls)

log("isAI", len(enriched_df.loc[enriched_df["isAI"]]))
log("not isAI", len(enriched_df.loc[~enriched_df["isAI"]]))

merged_df = pd.merge(filtered_df, enriched_df, on="id", how="outer")
merged_df['date'] = datetime.now().date()

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
# get embeddings, small model not as efffective
embedding_model = 'text-embedding-3-large'
response = client.embeddings.create(input=AIdf['title'].tolist(),
                                    model=embedding_model)
embedding_df = pd.DataFrame([e.model_dump()['embedding']
                            for e in response.data])
# embedding_array = embedding_df.values

# # find index of most central headline
# centroid = embedding_array.mean(axis=0)
# distances = np.linalg.norm(embedding_array - centroid, axis=1)
# start_index = np.argmin(distances)

# # Get the sorted indices and use them to sort the df
# sorted_indices = nearest_neighbor_sort(embedding_df.values, start_index)
# not really sure if this is better than the greedy traveling salesman method

sorted_indices = agglomerative_cluster_sort(embedding_df)
AIdf = AIdf.iloc[sorted_indices].reset_index(drop=True)

# create html message with formatted headlines
html_str = ""
for row in AIdf.itertuples():
    log(f"[{row.Index}. {row.title} - {row.src}]({row.url})")
    html_str += f'{row.Index}.<a href="{row.url}">{row.title} - {row.src}</a><br />\n'

# save headlines
with open('headlines.html', 'w') as f:
    f.write(html_str)

# send mail
log("Sending headlines email")
subject = f'AI headlines {datetime.now().strftime("%H:%M:%S")}'
send_gmail(subject, html_str)

log("Enqueuing URLs for scraping")
queue = multiprocessing.Queue()
for row in AIdf.itertuples():
    queue.put((row.id, row.url, row.title))

callable = process_url_queue_factory(queue)

num_browsers = 4
log(f"fetching {len(AIdf)} pages using {num_browsers} browsers")
results = launch_drivers(num_browsers, callable)

# flatten results
saved_pages = []
for r in results:
    saved_pages.extend(r)
# merge with AIdf to get path
pages_df = pd.DataFrame(results, columns=["id", "url", "title", "path"])
pages_df.columns = ['id', 'url', 'title', 'path']
AIdf = pd.merge(AIdf, pages_df[["id", "path"]], on='id', how="inner")

log("Summarizing scraped articles as bullet points")
responses = asyncio.run(fetch_all_summaries(AIdf))

log("Converting bullet points to HTML")
# bring summaries into dict
response_dict = {}
for i, response in responses:
    try:
        response_str = response["choices"][0]["message"]["content"]
        response_dict[i] = response_str
    except Exception as exc:
        log(exc)

markdown_str = ''
for i, row in enumerate(AIdf.itertuples()):
    mdstr = f"[{i+1}. {row.title}]({row.url})  \n\n{response_dict[row.id]} \n\n"
    markdown_str += mdstr

# save bullets
with open('bullets.md', 'w') as f:
    f.write(markdown_str)

# Convert Markdown to HTML
html_str = markdown.markdown(markdown_str, extensions=['extra'])

log("Sending bullet points email")
subject = f'AI news bullets {datetime.now().strftime("%H:%M:%S")}'
send_gmail(subject, html_str)

# summarize bullet points
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": FINAL_SUMMARY_PROMPT + markdown_str
               }],
    n=1,
    temperature=0.5
)
markdown_str = response.choices[0].message.content
# Convert Markdown to HTML
html_str = markdown.markdown(markdown_str)
log("Sending full summary email ")
subject = f'AI news summary {datetime.now().strftime("%H:%M:%S")}'
send_gmail(subject, html_str)


log("Finished")

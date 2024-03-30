#!/Users/drucev/anaconda3/envs/ainewsbot/bin/python

# AInewsbot.py
#
# - Open URLs of news sites specififed in `sources` dict using Selenium and Firefox
# - Save HTML of each URL in htmldata directory
# - Extract URLs from all files, create a pandas dataframe with url, title, src
# - Use ChatGPT to filter only AI-related headlines by sending a prompt and formatted table of headlines
# - Use SQLite to filter headlines previously seen
# - OPENAI_API_KEY should be in the environment or in a .env file
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

import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse
import time
import yaml

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# use firefox because it updates less often, can disable updates
# recommend importing profile from Chrome for cookies, passwords
# looks less like a bot with more user cruft in the profile
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


# import bs4
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

# import openai
from openai import OpenAI
import tiktoken

import dotenv

import sqlite3

# import IPython
# from IPython.display import HTML, Markdown, display

# from atproto import Client

# import PIL
# from PIL import Image

############################################################################################################

# load credentials if necessary
dotenv.load_dotenv()

DOWNLOAD_DIR = "htmldata"
MODEL = "gpt-4-turbo-preview"

MAX_INPUT_TOKENS = 3072
MAX_OUTPUT_TOKENS = 4096
MAX_RETRIES = 3
TEMPERATURE = 0

with open("sources.yaml", "r") as stream:
    try:
        sources = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

sources_reverse = {v["title"]: k for k, v in sources.items()}

############################################################################################################


def delete_files(download_dir):
    "delete non-hidden files in specified directory"

    # Iterate over all files in the directory
    for filename in os.listdir(download_dir):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(download_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                # If you want to remove subdirectories as well, use os.rmdir() here
                pass
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# you need this if you have not-descriptive link titles like 'link', can get a page title from html or tags
def get_og_tags(url):
    """get a dict of Open Graph og: tags such as title in the HEAD of a URL"""
    retdict = {}
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            head = soup.head
            if head:
                og_tags = head.find_all(
                    property=lambda prop: prop and prop.startswith("og:")
                )
                for tag in og_tags:
                    if "content" in tag.attrs:
                        retdict[tag["property"]] = tag["content"]

                page_title = ""
                title_tag = soup.find("title")
                if title_tag:
                    page_title = title_tag.text
                    if page_title:
                        retdict["title"] = page_title
        return retdict
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return retdict


def get_path_from_url(url):
    """
    Extracts the path following the top-level domain name from a URL.

    :param url: The URL string.
    :return: The path component of the URL.
    """
    parsed_url = urlparse(url)
    return parsed_url.path


def count_tokens(s):
    enc = tiktoken.encoding_for_model(MODEL)
    assert enc.decode(enc.encode("hello world")) == "hello world"
    return len(enc.encode(s))


def trimmed_href(link):
    """
    Trims everything in the string after a question mark such as a session ID.

    :param s: The input string.
    :return: The trimmed string.
    """
    # Find the position of the question mark
    s = link.get("href")
    if s:
        question_mark_index = s.find("?")

        # If a question mark is found, trim the string up to that point
        if question_mark_index != -1:
            return s[:question_mark_index]
        else:
            # Return the original string if no question mark is found
            return s
    else:
        return s


############################################################################################################
# get HTML files
############################################################################################################

print(datetime.now().strftime('%H:%M:%S'), "Started", flush=True)

# delete html files in download directory
delete_files(DOWNLOAD_DIR)

client = OpenAI()

# download files via selenium and firefox

firefox_app_path = '/Applications/Firefox.app'
# Path to geckodriver
geckodriver_path = '/Users/drucev/webdrivers/geckodriver'
# Set up Firefox options to use existing profile
# important for some sites that need a login, also a generic profile fingerprint that looks like a bot might get blocked
firefox_profile_path = '/Users/drucev/Library/Application Support/Firefox/Profiles/k8k0lcjj.default-release'
options = Options()
options.profile = firefox_profile_path
options.headless = True

print(datetime.now().strftime('%H:%M:%S'),
      "Initialized browser profile", flush=True)

# Create a Service object with the path
service = Service(geckodriver_path)

print(datetime.now().strftime('%H:%M:%S'),
      "Initialized browser service", flush=True)

# Set up the Firefox driver
driver = webdriver.Firefox(service=service, options=options)

print(datetime.now().strftime('%H:%M:%S'), "Initialized webdriver", flush=True)
sleeptime = 10

for sourcename, sourcedict in sources.items():
    print(datetime.now().strftime('%H:%M:%S'),
          f'Processing {sourcename}', flush=True)
    title = sourcedict["title"]
    url = sourcedict["url"]
    scroll = sourcedict.get("scroll", 0)
    click = sourcedict.get("click")

    # Open the page
    driver.get(url)

    # Wait for the page to load
    time.sleep(sleeptime)  # Adjust the sleep time as necessary

    if click:
        print(datetime.now().strftime('%H:%M:%S'),
              f"Clicking on {click}", flush=True)
        button = driver.find_element(By.XPATH, click)
        if button:
            button.click()
            print(datetime.now().strftime('%H:%M:%S'), "Clicked", flush=True)

    for _ in range(scroll):
        # scroll to bottom of infinite scrolling window
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        print(datetime.now().strftime('%H:%M:%S'),
              "Loading additional infinite scroll items", flush=True)
        time.sleep(sleeptime)  # wait for it to load additional items

    # Get the HTML source of the page
    html_source = driver.page_source

    # check encoding, default utf-8
    encoding = None  # Default to UTF-8 if not specified
    # Retrieve the content-type meta tag from the driver of HTML meta charset tag or http-equiv tag
    try:
        # retrieve the encoding from the driver by executing javascript
        encoding = driver.execute_script("return document.characterSet;")
        print(datetime.now().strftime('%H:%M:%S'),
              f'Encoding {encoding} (JS)', flush=True)
    except Exception:
        pass

    if encoding:
        pass
    else:
        try:
            # retrieve the encoding from the meta charset tag
            meta_tag = driver.find_element(By.XPATH, "//meta[@charset]")
            encoding = meta_tag.get_attribute("charset")
            print(datetime.now().strftime('%H:%M:%S'),
                  f'Encoding {encoding} (meta charset)', flush=True)
        except Exception:
            pass

    if encoding:
        pass
    else:
        try:
            # retrieve the encoding from the meta http-equiv tag
            meta_tag = driver.find_element(
                By.XPATH, "//meta[@http-equiv='Content-Type']")
            content_type = meta_tag.get_attribute("content")
            # Typical format is "text/html; charset=UTF-8"
            charset_start = content_type.find("charset=")
            if charset_start != -1:
                encoding = content_type[charset_start + 8:]
            print(datetime.now().strftime('%H:%M:%S'),
                  f'Encoding {encoding} (meta http-equiv)', flush=True)
        except Exception:  # as err:
            pass

    if encoding == 'windows-1252' or encoding is None:
        # Default to UTF-8 if not specified
        encoding = "utf-8"

    # Save the HTML to a local file
    datestr = datetime.now().strftime("%m_%d_%Y %I_%M_%S %p")
    outfile = f'{title} ({datestr}).html'
    print(datetime.now().strftime('%H:%M:%S'),
          f"Saving {outfile} as {encoding}", flush=True)
    with open(DOWNLOAD_DIR + "/" + outfile, 'w', encoding=encoding) as file:
        file.write(html_source)

# Close the browser
driver.quit()
print(datetime.now().strftime('%H:%M:%S'), "Quit webdriver", flush=True)
# finish downloading files

############################################################################################################
# process downloaded files

# List all paths in the directory matching today's date
nfiles = 50

# Get the current date
today = datetime.now()
year, month, day = today.year, today.month, today.day

datestr = datetime.now().strftime("%m_%d_%Y")

files = [os.path.join(DOWNLOAD_DIR, file) for file in os.listdir(DOWNLOAD_DIR)]
# filter files only
files = [file for file in files if os.path.isfile(file)]

# Sort files by modification time and take top 50
files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
file = files[:nfiles]

# filter files by with today's date ending in .html
files = [file for file in files if datestr in file and file.endswith(".html")]
print(datetime.now().strftime('%H:%M:%S'),
      f"Found {len(files)} files", flush=True)

############################################################################################################
# Parse news URLs and titles from downloaded HTML files
############################################################################################################

# parse all the URLs that look like news articles
# into all_urls list of dicts with url, title, src
all_urls = []

for file in files:
    # Extract filename from path
    filename = os.path.basename(file)

    # Find the position of '1_14_2024' in the filename
    position = filename.find(" (" + datestr)
    basename = filename[:position]

    sourcename = sources_reverse.get(basename)
    if sourcename is None:
        print(f"Skipping {basename}, no sourcename metadata")
        continue

    print(f"# {sourcename}")
    sources[sourcename]["latest"] = file

    # get contents
    with open(file, "r") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags
    if soup:
        links = soup.find_all("a")
    else:
        print(f"Skipping {sourcename}, unable to parse")

    # convert relative links to absolute links using base URL if present
    base_tag = soup.find('base')
    base_url = base_tag.get('href') if base_tag else sources[sourcename]["url"]
    for link in links:
        #  print(link.get("href"))
        link["href"] = urljoin(base_url, link.get('href', ""))
        #  print(link["href"])

    # print(len(links))
    # links = [l for l in links if l]
    # links = [l.strip() for l in links]

    print(len(links))

    # filter links by exclusion patterns
    for pattern in sources[sourcename].get("exclude", []):
        # print(pattern)
        # print([ l.get("href") for l in links])
        links = [
            link
            for link in links
            if link.get("href") is not None and not re.match(pattern, link.get("href"))
        ]
        # print(len(links))

    # filter links by inclusion patterns
    for pattern in sources[sourcename].get("include", []):
        # print(pattern, len(links))
        # filter links by inclusion pattern
        # print(pattern)
        # print(type(pattern))
        newlinks = []
        for link in links:
            href = link.get("href")
            # print(href)
            if href and re.match(pattern, href):
                newlinks.append(link)
        links = newlinks
        # print(len(links))

    # drop empty text
    links = [link for link in links if link.get_text(strip=True)]

    # drop empty url path, i.e. url = toplevel domain
    links = [link for link in links if len(
        get_path_from_url(trimmed_href(link))) > 1]
    # drop anything that is not http, like javascript: or mailto:
    links = [link for link in links if link.get(
        "href") and link.get("href").startswith("http")]
    # drop some ArsTechnica links that are just the number of comments and dupe the primary link
    links = [link for link in links if not re.match(
        "^(d+)$", link.get_text(strip=True))]

    for link in links:
        url = trimmed_href(link)
        title = link.get_text(strip=True)
        if title == "LINK":
            # try to update title
            og_dict = get_og_tags(url)
            if og_dict.get("og:title"):
                title = og_dict.get("og:title")

        # skip some low quality links that don't have full headline, like link to a Twitter or Threads account
        if len(title) <= 28 and title != "LINK":
            continue

        all_urls.append({"title": title, "url": url, "src": sourcename})
        # print(f"[{title}]({url})")

    print(len(links))
    print()


# make a pandas dataframe
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
# filter URLs
############################################################################################################

# filter ones not seen before
conn = sqlite3.connect('articles.db')
# Retrieve all URLs from the SQLite table
existing_urls = pd.read_sql_query("SELECT url FROM news_articles", conn)
# Close the SQLite connection
conn.close()

# Convert the URLs to a list for easier comparison
existing_urls_list = existing_urls['url'].tolist()
print(datetime.now().strftime('%H:%M:%S'),
      f"Found total URLs: {len(existing_urls_list)}", flush=True)

# Filter the original DataFrame
# Keep rows where the URL is not in the existing_urls_list
filtered_df = orig_df[~orig_df['url'].isin(existing_urls_list)]
print(datetime.now().strftime('%H:%M:%S'),
      f"Found new URLs: {len(filtered_df)}", flush=True)

# # Filter AI-related headlines using a prompt to OpenAI

# make pages that fit in a reasonably sized prompt
MAXPAGELEN = 50
pages = []
current_page = []
pagelength = 0

for row in filtered_df.itertuples():
    curlink = {"id": row.Index, "title": row.title}
    curlength = count_tokens(json.dumps(curlink))
    # Check if adding the current string would exceed the limit
    if len(current_page) >= MAXPAGELEN or pagelength + curlength > MAX_INPUT_TOKENS:
        # If so, start a new page
        pages.append(current_page)
        current_page = [curlink]
        pagelength = curlength
    else:
        # Otherwise, add the string to the current page
        current_page.append(curlink)
        pagelength += curlength

# add the last page if it's not empty
if current_page:
    pages.append(current_page)


def get_response_json(
    client,
    messages,
    verbose=False,
    model=MODEL,
    # max_input_tokens=MAX_INPUT_TOKENS,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    max_retries=MAX_RETRIES,
    temperature=TEMPERATURE,
):
    if type(messages) is not list:  # allow passing one string for convenience
        messages = [{"role": "user", "content": messages}]

    if verbose:
        print("\n".join([str(msg) for msg in messages]))

    # truncate number of tokens
    # retry loop, have received untrapped 500 errors like too busy
    for i in range(max_retries):
        if i > 0:
            print(f"Attempt {i+1}...")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            # no exception thrown
            return response
        except Exception as error:
            print(f"An exception occurred on attempt {i+1}:", error)
            time.sleep(sleeptime)
            continue  # try again
        # retries exceeded if you got this far
    print("Retries exceeded.")
    return None


prompt = """
You will act as a research assistant classifying news stories as related to artificial intelligence (AI) or unrelated to AI.

Your task is to read JSON format objects from an input list of news stories using the schema below delimited by |, and output JSON format objects for each using the schema below delimited by ~.

Define a list of objects representing news stories in JSON format as in the following example:
|
{'stories':
[{'id': 97, 'title': 'AI to predict dementia, detect cancer'},
 {'id': 103,'title': 'Figure robot learns to make coffee by watching humans for 10 hours'},
 {'id': 103,'title': 'Baby trapped in refrigerator eats own foot'},
 {'id': 210,'title': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'},
 {'id': 298,'title': 'The 5 most interesting PC monitors from CES 2024'},
 ]
}
|

Based on the title, you will classify each story as being about AI or not.

For each object, you will output the input id field, and a field named isAI which is true if the input title is about AI and false if the input title is not about AI.

When extracting information please make sure it matches the JSON format below exactly. Do not output any attributes that do not appear in the schema below.
~
{'stories':
[{'id': 97, 'isAI': true},
 {'id': 103, 'isAI': true},
 {'id': 103, 'isAI': false},
 {'id': 210, 'isAI': true},
 {'id': 298, 'isAI': false}]
}
~

You may interpret the term AI broadly as pertaining to
- machine learning models
- large language models
- robotics
- reinforcement learning
- computer vision
- OpenAI
- ChatGPT
- other closely related topics.

You will return an array of valid JSON objects.

The field 'id' in the output must match the field 'id' in the input EXACTLY.

The field 'isAI' must be either true or false.

The list of news stories to classify and enrich is:


"""

responses = []
enriched_urls = []
for i, p in enumerate(pages):
    print(
        f"{datetime.now().strftime('%H:%M:%S')} send page {i+1} of {len(pages)}, {len(p)} items "
    )
    # print(prompt + json.dumps(p))
    response = get_response_json(client, prompt + json.dumps(p))
    responses.append(response.choices[0].message.content)
    retval = json.loads(responses[-1])
    retlist = []
    # usually comes back as a dict with a single arbitrary key like "stories" with a list value
    if type(retval) is dict:
        for k, v in retval.items():
            if type(v) is list:
                retlist.extend(v)
            else:
                retlist.append(v)
        print(
            f"{datetime.now().strftime('%H:%M:%S')} got dict with {len(retlist)} items "
        )
    elif type(retval) is list:  # in case it comes back as a list
        retlist = retval
        print(
            f"{datetime.now().strftime('%H:%M:%S')} got list with {len(retlist)} items "
        )
    else:
        print(str(type(retval)))
    enriched_urls.extend(retlist)


enriched_df = pd.DataFrame(enriched_urls)
enriched_df.head()

print("isAI", len(enriched_df.loc[enriched_df["isAI"]]))
print("not isAI", len(enriched_df.loc[~enriched_df["isAI"]]))

merged_df = pd.merge(filtered_df, enriched_df, on="id", how="outer")
merged_df['date'] = datetime.now().date()
merged_df.head()

# ideally should be empty, shouldn't get back rows that don't match to existing
print("Unmatched response rows: ", len(merged_df.loc[merged_df["src"].isna()]))

# should be empty, should get back all rows from orig
print("Unmatched source rows:", len(merged_df.loc[merged_df["isAI"].isna()]))

# # update SQLite database
conn = sqlite3.connect('articles.db')
cursor = conn.cursor()

# # Create table with a date column
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS news_articles (
#     id INTEGER PRIMARY KEY,
#     src TEXT,
#     title TEXT,
#     url TEXT UNIQUE,
#     isAI BOOLEAN,
#     article_date DATE
# )
# ''')
# conn.commit()
# conn.close()


# Function to insert a new article
def insert_article(cursor, src, title, url, isAI, article_date):
    try:
        cursor.execute("INSERT OR IGNORE INTO news_articles (src, title, url, isAI, article_date) VALUES (?, ?, ?, ?, ?)",
                       (src, title, url, isAI, article_date))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Duplicate entry for URL: {url}")
    except Exception as err:
        print(err)


for row in merged_df.itertuples():
    # print(row)
    insert_article(cursor, row.src, row.title, row.url, row.isAI, row.date)

AIdf = merged_df.loc[merged_df["isAI"]].reset_index()

############################################################################################################
# save order list and send email
############################################################################################################

# Attempt to order by topic by getting embeddings and solving a traveling salesman problem
print(datetime.now().strftime('%H:%M:%S'),
      f"Fetching embeddings for {len(AIdf)} headlines", flush=True)
embedding_model = 'text-embedding-3-small'
response = client.embeddings.create(input=AIdf['title'].tolist(),
                                    model=embedding_model)
embedding_df = pd.DataFrame([e.dict()['embedding'] for e in response.data])

# naive greedy solution to traveling salesman problem
embedding_array = embedding_df.values


def nearest_neighbor_sort(embedding_array):
    # Compute the pairwise Euclidean distances between all embeddings
    distances = cdist(embedding_array, embedding_array, metric='euclidean')

    # Start from the first headline as the initial point
    path = [0]
    visited = set(path)

    while len(path) < len(embedding_array):
        last = path[-1]
        # Set the distances to already visited nodes to infinity to avoid revisiting
        distances[:, last][list(visited)] = np.inf
        # Find the nearest neighbor
        nearest = np.argmin(distances[:, last])
        path.append(nearest)
        visited.add(nearest)

    return np.array(path)


# Get the sorted indices and use them to sort the df
sorted_indices = nearest_neighbor_sort(embedding_array)
sorted_embedding_array = embedding_array[sorted_indices]

html_str = ""
for i, j in enumerate(sorted_indices):
    row = AIdf.iloc[j]
    print(f"[{i}. {row.title} - {row.src}]({row.url})")
    html_str += f'{i}.<a href="{row.url}">{row.title} - {row.src}</a><br />\n'

print(datetime.now().strftime('%H:%M:%S'), "Sending mail", flush=True)

# send mail
# credentials
gmail_user = os.getenv("GMAIL_USER")
gmail_password = os.getenv("GMAIL_PASSWORD")

# Email content
from_addr = gmail_user
to_addr = gmail_user
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
message['From'] = from_addr
message['To'] = to_addr
message['Subject'] = subject
message.attach(MIMEText(body, 'html'))

# Create SMTP session
with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()  # Secure the connection
    server.login(gmail_user, gmail_password)
    text = message.as_string()
    server.sendmail(from_addr, to_addr, text)

print(datetime.now().strftime('%H:%M:%S'), "Finished", flush=True)

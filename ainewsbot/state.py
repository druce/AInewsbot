"""
This module contains all the state functions for LangGraph nodes used in the AI News Bot project.

Each function is designed to perform a specific task in the pipeline, such as initializing the agent state,
downloading sources, extracting URLs, filtering content, performing topic analysis, clustering, summarizing,
and composing newsletters. These functions interact with the `AgentState` object, which serves as the central
state management structure for the LangGraph workflow.

The functions are intended to be used as nodes in a LangGraph state graph, enabling modular and reusable
processing steps for tasks like web scraping, content filtering, topic extraction, and email generation.

Key Features:
- Integration with external APIs (e.g., OpenAI, NewsCatcher, GNews, NewsAPI).
- Support for multiprocessing and asynchronous operations for efficient data processing.
- Use of machine learning models for clustering, topic extraction, and content summarization.
- Modular design for easy customization and extension of the pipeline.
"""
# import pdb
import os
import re
import pickle
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import asyncio
from typing import TypedDict
from urllib.parse import urlparse
import sqlite3
import random
import math
import yaml
import markdown

import requests
import tldextract
from IPython.display import display  # , Audio

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from openai import OpenAI

import langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.errors import NodeInterrupt
# import subprocess

from .llm import (paginate_df, process_dataframes, fetch_all_summaries,
                  filter_page_async, filter_df, filter_df_rows,
                  get_all_canonical_topic_results, clean_topics,
                  Stories, TopicSpecList, TopicHeadline, TopicCategoryList, SingleTopicSpec,
                  Sites, Newsletter
                  )
# from ainb_sllm import (
#     sfetch_all_summaries, sfilter_page_async)
# from ainb_sllm import sfetch_all_summaries
from .scrape import (
    parse_file, fetch_queue, fetch_source_queue)
from .utilities import (log, delete_files, filter_unseen_urls_db,
                        nearest_neighbor_sort, send_gmail, unicode_to_ascii)
from .config import (DOWNLOAD_DIR, PAGES_DIR, SOURCECONFIG, SOURCES_EXPECTED,
                     CANONICAL_TOPICS, SQLITE_DB,
                     HOSTNAME_SKIPLIST, SITE_NAME_SKIPLIST, SOURCE_REPUTATION,
                     SCREENSHOT_DIR, MINIMUM_STORY_RATING)

from .prompts import (
    FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT,
    TOPIC_SYSTEM_PROMPT, TOPIC_USER_PROMPT,
    TOPIC_FILTER_SYSTEM_PROMPT, TOPIC_FILTER_USER_PROMPT,
    TOPIC_WRITER_SYSTEM_PROMPT, TOPIC_WRITER_USER_PROMPT,
    TOPIC_REWRITE_SYSTEM_PROMPT, TOPIC_REWRITE_USER_PROMPT,
    LOW_QUALITY_SYSTEM_PROMPT, LOW_QUALITY_USER_PROMPT,
    ON_TOPIC_SYSTEM_PROMPT, ON_TOPIC_USER_PROMPT,
    IMPORTANCE_SYSTEM_PROMPT, IMPORTANCE_USER_PROMPT,
    TOP_CATEGORIES_SYSTEM_PROMPT, TOP_CATEGORIES_USER_PROMPT,
    FINAL_SUMMARY_SYSTEM_PROMPT, FINAL_SUMMARY_USER_PROMPT,
    REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT,
    TOPIC_ROUTER_SYSTEM_PROMPT, TOPIC_ROUTER_USER_PROMPT,
    PROMPT_BATTLE_SYSTEM_PROMPT, PROMPT_BATTLE_USER_PROMPT,
    DEDUPLICATE_SYSTEM_PROMPT, DEDUPLICATE_USER_PROMPT,
    SITE_NAME_PROMPT,
)


class AgentState(TypedDict):
    """
    State of the LangGraph agent.
    Each node in the graph is a function that takes the current state and returns the updated state.
    """

    # the current working set of headlines (pandas dataframe not supported)
    AIdf: list[dict]
    # ignore stories before this date for deduplication (force reprocess since)
    before_date: str
    do_download: bool  # if False use existing files, else download from sources
    model_low: str     # cheap fast model like gpt-4o-mini or flash
    model_medium: str  # medium model like gpt-4o or gemini-1.5-pro
    model_high: str    # slow expensive thinking model like o3-mini
    sources: dict  # sources to scrap
    sources_reverse: dict[str, str]  # map file names to sources
    bullets: list[str]  # bullet points for summary email
    summary: str  # final summary
    cluster_topics: list[str]  # list of cluster topics
    topics_str: str  # edited topics
    n_edits: int  # count edit iterations so we don't keep editing forever
    max_edits: int  # max number of edits to make
    edit_complete: bool  # edit will update if no more edits to make
    n_browsers: int  # number of browsers to use for scraping


def make_bullet(row, include_topics=True):
    """Given a row in AIdf, return a markdown block with links, topics, bullet points """
    # rowid = str(row.id) + ". " if hasattr(row, "id") else ""
    title = row.title if hasattr(row, "title") else ""
    site_name = row.site_name if hasattr(row, "site_name") else ""
    actual_url = row.actual_url if hasattr(row, "actual_url") else ""
    # bullet = "\n\n" + row.bullet if hasattr(row, "bullet") else ""
    summary = "\n\n" + str(row.summary) if hasattr(row, "summary") else ""
    topic_str = ""
    if include_topics:
        # add cluster_topic_str to beginning of topics in bullet if not other
        if hasattr(row, "cluster_name") and row.cluster_name is not None and row.cluster_name.lower() not in ["", "other", "none"]:
            topic_str = "\n\nTopics: " + row.cluster_name
        if hasattr(row, "topic_str") and row.topic_str:
            if topic_str:
                topic_str += ", "
            else:
                topic_str = "\n\nTopics: "
            topic_str += str(row.topic_str)
    rating = f"\n\nRating: {max(row.rating, 0):.2f}" if hasattr(
        row, "rating") else ""
    return f"[{title} - {site_name}]({actual_url}){topic_str}{rating}{summary}\n\n"


def fn_initialize(state: AgentState) -> AgentState:
    """
    Initializes the agent state by loading source configurations from SOURCECONFIG (sources.yaml) .

    Args:
        state (AgentState): The current state of the agent.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        AgentState: The updated state of the agent.

    Raises:
        yaml.YAMLError: If there is an error while loading the YAML file.

    """

    #  load sources to scrape from sources.yaml
    with open(SOURCECONFIG, "r", encoding="utf-8") as stream:
        try:
            state['sources'] = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            log("fn_initialize")
            log(exc)

    log(f"Initialized {len(state['sources'])} items in sources from {SOURCECONFIG}")

    # make a reverse dict to map file titles to source names
    state['sources_reverse'] = {}
    for k, v in state['sources'].items():
        log(f"{k} -> {v['url']} -> {v['title']}.html")
        v['sourcename'] = k
        # map filename (title) to source name
        state['sources_reverse'][v['title']] = k

    log(f"Initialized {len(state['sources_reverse'])} items in sources_reverse")

    return state


def fn_download_sources(state: AgentState) -> AgentState:
    """
    Scrapes sources and saves HTML files.
    If state["do_download"] is True, deletes all files in DOWNLOAD_DIR (htmldata) and scrapes fresh copies.
    If state["do_download"] is False, uses existing files in DOWNLOAD_DIR.
    Uses state["sources"] for config info on sources to scrape
    For each source, saves the current filename to state["sources"][sourcename]['latest']
    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent.
    """

    if state.get("do_download"):
        # empty download directories
        delete_files(DOWNLOAD_DIR)
        delete_files(PAGES_DIR)
        delete_files(SCREENSHOT_DIR)

        # save each file specified from sources
        log(
            f"Saving HTML files using async concurrency= {state['n_browsers']}")
        # Create a queue for multiprocessing and populate it
        queue = asyncio.Queue()
        for item in state.get("sources").values():
            asyncio.run(queue.put(item))

        results = asyncio.run(fetch_source_queue(queue, state['n_browsers']))

        # saved_pages = [item for ret_array in results for item in ret_array]
        for source_name, source_file in results:
            log(f"{source_name} -> {source_file}")
            state["sources"][source_name]['latest'] = source_file

        log(f"Saved {len(results)} HTML files")

    else:   # use existing files
        log(f"Web fetch disabled, using existing files in {DOWNLOAD_DIR}")
        # Get the current date
        files = [os.path.join(DOWNLOAD_DIR, file)
                 for file in os.listdir(DOWNLOAD_DIR)]
        # filter files ending in .html
        files = [file for file in files if file.endswith(".html")]
        log(f"Found {len(files)} previously downloaded files")
        for sourcefile in files:
            log(sourcefile)

        for sourcefile in files:
            filename = os.path.basename(sourcefile).split(".")[0]
            # match to source name
            sourcename = state.get("sources_reverse", {}).get(filename)
            if sourcename is None:
                log(f"Skipping {filename}, no sourcename metadata")
                continue
            state["sources"][sourcename]['latest'] = sourcefile

    return state


def fn_extract_urls(state: AgentState) -> AgentState:
    """
    Extracts news URLs from the latest HTML files matching the patterns defined in the state['sources'] configuration info.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent with the extracted URLs stored in state['AIdf'].
    """
    # Parse news URLs and titles from downloaded HTML files
    log("Parsing html files")
    all_urls = []
    for sourcename, sourcedict in state['sources'].items():
        filename = sourcedict.get('latest')
        if not filename:
            log(f"no filename found for {sourcename}")
            continue

        log(sourcename + ' -> ' + filename)
        links = parse_file(state['sources'][sourcename])
        log(f"{len(links)} links found")
        all_urls.extend(links)

    log(f"Saved {len(all_urls)} links")

    # make a pandas dataframe of all the links found
    aidf = (
        pd.DataFrame(all_urls)
        .groupby("url")
        .first()
        .reset_index()
        .sort_values("src")[["src", "title", "url"]]
        .reset_index(drop=True)
        .reset_index(drop=False)
        .rename(columns={"index": "id"})
    )
    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def fn_verify_download(state: AgentState) -> AgentState:
    """
    Verify all sources downloaded by checking src present in AIdf
    If there is a bot block the html file might be present but have no valid links
    """
    sources_downloaded = len(
        pd.DataFrame(state["AIdf"]).groupby("src").count()[['id']])
    missing_sources = SOURCES_EXPECTED-sources_downloaded

    if missing_sources:
        log(
            f"verify_download failed, found {SOURCES_EXPECTED} sources in AIdf, {missing_sources} missing")
        str_missing = str(set(state["sources"].keys(
        )) - set(pd.DataFrame(state["AIdf"]).groupby("src").count()[['id']].index))
        log(f"Missing sources: {str_missing}")
        raise NodeInterrupt(
            f"{missing_sources} missing sources: {str_missing}")
    log(f"verify_download passed, found {SOURCES_EXPECTED} sources in AIdf, {missing_sources} missing")
    return state


# def fn_extract_newscatcher(state: AgentState) -> AgentState:
#     """get AI news via newscatcher API
#     https://docs.newscatcherapi.com/api-docs/endpoints/search-news
#     """

#     q = 'Artificial Intelligence'
#     page_size = 100
#     log(f"Fetching top {page_size} stories matching {q} from Newscatcher")
#     base_url = "https://api.newscatcherapi.com/v2/search"
#     time_24h_ago = datetime.now() - timedelta(hours=24)

#     # Put API key in headers
#     headers = {'x-api-key': os.getenv('NEWSCATCHER_API_KEY')}

#     # Define search parameters
#     params = {
#         'q': q,
#         'lang': 'en',
#         'sources': ','.join(NEWSCATCHER_SOURCES),
#         'from': time_24h_ago.strftime('%Y-%m-%d %H:%M:%S'),
#         'to': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'page_size': page_size,  # by default should be most highly relevant to the search
#         'page': 1
#     }

#     # Make API call with headers and params
#     response = requests.get(base_url, headers=headers,
#                             params=params, timeout=60)

#     # Encode received results
#     results = json.loads(response.text.encode())
#     if response.status_code != 200:
#         log('ERROR: API call failed.')
#         log(results)

#     # merge into existing df
#     newscatcher_df = pd.DataFrame(results['articles'])[['title', 'link']]
#     newscatcher_df['src'] = 'Newscatcher'
#     newscatcher_df = newscatcher_df.rename(columns={'link': 'url'})
# #     display(newscatcher_df.head())
#     aidf = pd.DataFrame(state['AIdf'])
# #     display(AIdf.head())

#     max_id = aidf['id'].max()
#     # add id column to newscatcher_df
#     newscatcher_df['id'] = range(max_id + 1, max_id + 1 + len(newscatcher_df))
#     aidf = pd.concat([aidf, newscatcher_df], ignore_index=True)
#     state['AIdf'] = aidf.to_dict(orient='records')
#     return state


# def fn_extract_gnews(state: AgentState) -> AgentState:
#     """get AI news via GNews API
#     https://gnews.io/docs/v4#search-endpoint-query-parameters
#     100 stories is 50 euros per month , not implemented
#     """

#     base_url = 'https://gnews.io/api/v4/search'
#     q = 'Artificial Intelligence'
#     lang = 'en'
#     country = 'us'
#     maxlinks = 25  # max for lowest subscription tier
#     apikey = os.getenv('GNEWS_API_KEY')
#     time_12h_ago = datetime.now() - timedelta(hours=12)
#     fromtime = time_12h_ago.strftime('%Y-%m-%dT%H:%M:%S')
#     sortby = 'relevance'

#     log(f"Fetching top {maxlinks} stories matching {q} since {fromtime} from GNews")

#     # Define search parameters
#     params = {
#         'q': q,
#         'lang': lang,
#         'country': country,
#         'from': fromtime,
#         'max': maxlinks,
#         'apikey': apikey,
#         'sortby': sortby
#     }

#     # Make API call with headers and params
#     response = requests.get(base_url, params=params, timeout=60)
#     # need to do 4 times with 4 pages of 25 each, concatenate 4 dfs or merge 4 times
#     # # Encode received results
#     if response.status_code != 200:
#         log('ERROR: API call failed.')
#         log(response)
#     results = json.loads(response.text.encode())

#     # # merge into existing df
#     gnews_df = pd.DataFrame(results['articles'])[['title', 'link']]
#     gnews_df['src'] = 'GNews'
#     gnews_df = gnews_df.rename(columns={'link': 'url'})
#     #     display(newscatcher_df.head())
#     aidf = pd.DataFrame(state['AIdf'])
#     #     display(AIdf.head())

#     max_id = aidf['id'].max()
#     # add id column to gnews_df
#     gnews_df['id'] = range(max_id + 1, max_id + 1 + len(gnews_df))
#     aidf = pd.concat([aidf, gnews_df], ignore_index=True)
#     state['AIdf'] = aidf.to_dict(orient='records')


def fn_extract_newsapi(state: AgentState) -> AgentState:
    """
    get AI news via newsapi
    https://newsapi.org/docs/get-started
    from newsapi import NewsApiClient
    """
    news_api_key = os.environ['NEWSAPI_API_KEY']
    aidf = pd.DataFrame(state['AIdf'])

    page_size = 100
    q = 'artificial intelligence'
    date_24h_ago = datetime.now() - timedelta(hours=24)
    formatted_date = date_24h_ago.strftime("%Y-%m-%dT%H:%M:%S")
    log(f"Fetching top {page_size} stories matching {q} since {formatted_date} from NewsAPI")

    baseurl = 'https://newsapi.org/v2/everything'

    # Define search parameters
    params = {
        'q': q,
        'from': formatted_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': news_api_key,
        'pageSize': 100
    }

    # Make API call with headers and params
    response = requests.get(baseurl, params=params, timeout=60)
    if response.status_code != 200:
        log('ERROR: API call failed.')
        log(response.text)

    data = response.json()
    newsapi_df = pd.DataFrame(data['articles'])

    # only 1st page is supported on free account
    # n_articles = data['totalResults']
    # n_additional_pages = n_articles // 100
    # for i in range(n_additional_pages):
    #     page = i+2  # start at page 2
    #     url = f'https://newsapi.org/v2/everything?q=artificial%20intelligence&from=2025-02-16&sortBy=popularity&apiKey={NEWSAPI_API_KEY}&pageSize=100&page={page}'
    #     print(url)
    #     r = requests.get(url)
    #     data = r.json()
    #     tmpdf = pd.DataFrame(data['articles'])
    #     df = pd.concat([df, tmpdf], axis=1).reset_index(drop=True)

    newsapi_df = newsapi_df[['title', 'url']]
    newsapi_df['src'] = 'NewsAPI'
    max_id = aidf['id'].max()
    # add id column to newscatcher_df
    newsapi_df['id'] = range(max_id + 1, max_id + 1 + len(newsapi_df))
    aidf = pd.concat([aidf, newsapi_df], ignore_index=True)
    state['AIdf'] = aidf.to_dict(orient='records')
    return state


def fn_filter_urls(state: AgentState, model_low: any) -> AgentState:
    """
    Filters the URLs in state["AIdf"] to include only those that have not been previously seen,
    and are related to AI according to the response from a ChatGPT prompt.

    Args:
        state (AgentState): The current state of the agent.
        before_date (str, optional): The date before which the URLs should be filtered. Defaults to "".

    Returns:
        AgentState: The updated state of the agent with the filtered URLs stored in state["AIdf"].

    """
    # filter to URL not previously seen
    aidf = pd.DataFrame(state['AIdf'])

    aidf = filter_unseen_urls_db(aidf, before_date=state.get("before_date"))

    if len(aidf) == 0:
        log("No new URLs, returning")
        return state

    # dedupe identical urls
    aidf = aidf.sort_values("src") \
        .groupby("url") \
        .first() \
        .reset_index(drop=False)  \
        .drop(columns=['id']) \
        .reset_index() \
        .rename(columns={'index': 'id'})
    log(f"Found {len(aidf)} unique new headlines")

    # # dedupe identical headlines
    # # filter similar titles differing by type of quote or something
    # TODO: deduplicate by e.g. > 98% cosine similarity on embedding
    # 0.98 Levenshtein distance

    aidf['title'] = aidf['title'].apply(unicode_to_ascii)
    aidf['title_clean'] = aidf['title'].map(lambda s: " ".join(s.split()))
    aidf = aidf.sort_values("src").groupby("title_clean").first().reset_index(
        drop=True).drop(columns=['id']).reset_index().rename(columns={'index': 'id'})
    log(f"Found {len(aidf)} unique cleaned new headlines")
    # filter AI-related headlines using a prompt
    results = asyncio.run(process_dataframes(
        # results = sprocess_dataframes(
        dataframes=paginate_df(aidf[["id", "title"]]),
        system_prompt=FILTER_SYSTEM_PROMPT,
        user_prompt=FILTER_USER_PROMPT,
        output_class=Stories,
        model=model_low,
        item_list_field="items",
        item_id_field="id"
    ))

    # should return a list of Story
    filter_df = pd.DataFrame([
        {"id": story.id, "isAI": story.isAI}
        for story in results
    ])
    try:  # for idempotency
        aidf = aidf.drop(columns=['isAI'])
    except Exception as exc:
        pass
        # error expected, no need to print
        # log("fn_filter_urls")
        # log(exc)

    # merge returned df with isAI column into original df on id column
    aidf = pd.merge(aidf, filter_df, on="id", how="outer")
    log(aidf.columns)
    # set hostname based on actualurl
    # ideally resolve redirects but Google News blocks
    aidf['actual_url'] = aidf['url']
    aidf['hostname'] = aidf['actual_url'].apply(
        lambda url: urlparse(url).netloc)

    # update SQLite database with all seen URLs (we are doing this using url and ignoring redirects)
    # ideally should this later , if something breaks before email, need to rerun with before_date set
    log(f"Inserting {len(aidf)} URLs into {SQLITE_DB}")
    with sqlite3.connect(SQLITE_DB) as conn:
        cursor = conn.cursor()
        try:
            rows_to_insert = [
                (row.src, row.src, row.title, row.url, row.actual_url, row.isAI,
                 datetime.now().date(), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                for row in aidf.itertuples()
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO news_articles (src, actual_src, title, url, actual_url, isAI, article_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows_to_insert
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            log(f"Integrity error during batch insert: {e}")
        except Exception as err:
            log(f"Error during batch insert: {err}")

    # keep headlines that are related to AI
    aidf = aidf.loc[aidf["isAI"] == 1] \
        .reset_index(drop=True) \
        .reset_index() \
        .drop(columns=["id"]) \
        .rename(columns={'index': 'id'})

    log(f"Found {len(aidf)} AI headlines")

    # update actual URLs for Google News redirects
    # I think Google changed something so this no longer works, instead of a 301 redirct
    # get a javascript page that redirects. Also tomorrow we might see different URLs for same stories
    # AIdf = get_google_news_redirects(AIdf)

    conn = sqlite3.connect('articles.db')
    query = "select * from sites"
    sites_df = pd.read_sql_query(query, conn)
    sites_dict = {row.hostname: row.site_name for row in sites_df.itertuples()}
    domains_dict = {row.hostname: row.domain for row in sites_df.itertuples()}
    conn.close()

    # get clean site_name
    aidf['site_name'] = aidf['hostname'].apply(
        lambda hostname: sites_dict.get(hostname, hostname))

    # if any missing clean site names, populate them using OpenAI
    missing_site_names = len(aidf.loc[aidf['site_name'] == ""])
    if missing_site_names:
        log(f"Asking OpenAI for {missing_site_names} missing site names")
        responses = asyncio.run(process_dataframes(paginate_df(aidf[["url"]]),
                                                   "",
                                                   SITE_NAME_PROMPT,
                                                   Sites, model=model_low))
        # update site_dict from responses
        new_hostnames = []
        for r in responses:
            parsed_url = urlparse(r.url)
            hostname = parsed_url.hostname
            new_hostnames.append(hostname)
            sites_dict[hostname] = r.site_name
            log(f"Looked up {r.url} -> {r.site_name}")
            registered_domain = ""
            extracted = tldextract.extract(hostname)
            if extracted.domain and extracted.suffix:
                registered_domain = f"{extracted.domain}.{extracted.suffix}"
                log(f"Looked up {hostname} -> {registered_domain}")
                domains_dict[hostname] = registered_domain

        # update sites table with new names
        for new_hostname in new_hostnames:
            sqlstr = "INSERT OR IGNORE INTO sites (hostname, site_name, domain) VALUES (?, ?, ?);"
            log(
                f"Updated {new_hostname} -> {sites_dict[new_hostname]} ({domains_dict[new_hostname]})")
            conn.execute(
                sqlstr, (new_hostname, sites_dict[new_hostname], domains_dict[new_hostname]))
            conn.commit()
        # reapply to AIdf with updated sites
        aidf['site_name'] = aidf['hostname'].apply(
            lambda hostname: sites_dict.get(hostname, hostname))
    else:
        log("No missing site names")

    # drop banned slop sites

    aidf = aidf.loc[~aidf["hostname"].str.lower().isin(HOSTNAME_SKIPLIST)]
    aidf = aidf.loc[~aidf["site_name"].str.lower().isin(SITE_NAME_SKIPLIST)]

    aidf['reputation'] = aidf['hostname'].apply(
        lambda x: SOURCE_REPUTATION.get(x, 0))

    state["AIdf"] = aidf.to_dict(orient='records')
    return state


def fn_topic_analysis(state: AgentState, model_low: any) -> AgentState:
    """
    Extracts and selects topics for each headline in the state['AIdf'] dataframe, scrubs them, and stores them back in the dataframe.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent with the extracted and selected topics stored in state['AIdf'].
    """

    langchain.verbose = True
    aidf = pd.DataFrame(state['AIdf'])
    aidf["summary"] = aidf["summary"].fillna("")
    aidf["title"] = aidf["title"].fillna("")
    tmpdf = aidf[['id', 'title', 'summary']].copy()
    tmpdf["summary"] = tmpdf["title"] + "\n" + tmpdf["summary"]
    # gemini seems to have had trouble with 50 headlines
    pages = paginate_df(tmpdf[["id", "summary"]], maxpagelen=20)

    # apply topic extraction prompt to AI headlines
    log(f"start free-form topic extraction using {str(type(model_low))}")
    # pdb.set_trace()
    topic_results = asyncio.run(process_dataframes(
        dataframes=pages,
        system_prompt=TOPIC_SYSTEM_PROMPT,
        user_prompt=TOPIC_USER_PROMPT,
        output_class=TopicSpecList,
        model=model_low,
        item_list_field="items",
        item_id_field="id"
    ))
    # topic_results = sprocess_dataframes(
    #     dataframes=pages,
    #     input_prompt=TOPIC_PROMPT,
    #     output_class=TopicSpecList,
    #     model=model_low,
    #     item_list_field="items",
    #     item_id_field="id"
    # )
    topics_df = pd.DataFrame([[item.id, item.extracted_topics] for item in topic_results], columns=[
        "id", "extracted_topics"])
    log(f"{len(topics_df)} free-form topics extracted")

    all_topics = [item.lower() for row in topics_df.itertuples()
                  for item in row.extracted_topics]

    item_counts = Counter(all_topics)
    # use categories that are canonical or show up twice in freeform
    filtered_topics = [item for item in item_counts if item_counts[item]
                       >= 2 and item not in {'technology', 'ai', 'artificial intelligence', 'gen ai', 'no content'}]

    categories = sorted(CANONICAL_TOPICS)
    lcategories = set([c.lower() for c in categories] +
                      [c.lower() for c in filtered_topics])

    # pdb.set_trace()
    log(f"Starting assigned topic extraction using {str(type(model_low))}")
    assigned_topics = asyncio.run(
        get_all_canonical_topic_results(pages, lcategories, model_low))
    # assigned_topics = sget_all_canonical_topic_results(
    #     pages, lcategories, model_low)
    ctr_dict = defaultdict(list)

    for (topic, relevant_list) in assigned_topics:
        for ctr in relevant_list:
            if ctr.relevant:
                ctr_dict[ctr.id].append(topic)

    topics_df['assigned_topics'] = topics_df['id'].apply(
        lambda id: ctr_dict.get(id, ""))

    log("Cleaning and formatting topics")
    # pdb.set_trace()
    topics_df["topics"] = topics_df.apply(clean_topics, axis=1)
    topics_df["topic_str"] = topics_df.apply(
        lambda row: ", ".join(row.topics), axis=1)

    try:  # for idempotency
        aidf = aidf.drop(columns=['topic_str', 'title_topic_str'])
    except Exception as exc:
        pass

    aidf = pd.merge(
        aidf, topics_df[["id", "topic_str"]], on="id", how="outer")

    # filter topics using TOPIC_FILTER_SYSTEM_PROMPT and TOPIC_FILTER_USER_PROMPT
    log("Filtering redundant topics")
    aidf["filter_input"] = aidf.apply(
        lambda row: f"""### <<<ARTICLE SUMMARY>>>
# {row.title}

{row.summary}
### <<<END>>>
### <<<CANDIDATE TOPICS>>>
{row.topic_str}
### <<<END>>>
""",
        axis=1
    )

    aidf = asyncio.run(filter_df_rows(aidf,
                                      model_low,
                                      TOPIC_FILTER_SYSTEM_PROMPT, TOPIC_FILTER_USER_PROMPT,
                                      "topic_list",
                                      "filter_input",
                                      "",
                                      output_class=TopicCategoryList,
                                      output_class_label="items"
                                      ))
    aidf["topic_str"] = aidf["topic_list"].str.join(", ")

    aidf['title_topic_str'] = aidf.apply(
        lambda row: f'{row.title} (Topics: {row.topic_str})', axis=1)
    log("End topic analysis")

    # redo bullets with topics
    aidf["bullet"] = aidf.apply(make_bullet, axis=1)
    state["AIdf"] = aidf.to_dict(orient='records')

    return state


def fn_topic_clusters(state: AgentState, model_low: any) -> AgentState:
    """
    Fetches embeddings for the headlines, creates clusters of similar articles using DBSCAN, and sorts
    using the clusters and a traveling salesman shortest traversal in embedding space.

    Parameters:
    state (AgentState): The state of the agent.

    Returns:
    AgentState: The updated state of the agent.

    """
    aidf = pd.DataFrame(state['AIdf'])

    log(f"Fetching embeddings for {len(aidf)} headlines")
    embedding_model = 'text-embedding-3-large'
    client = OpenAI()
    response = client.embeddings.create(input=aidf['title_topic_str'].tolist(),
                                        model=embedding_model)
    embedding_df = pd.DataFrame(
        [e.model_dump()['embedding'] for e in response.data])

    # greedy traveling salesman sort
    log("Sort with nearest_neighbor_sort")
    sorted_indices = nearest_neighbor_sort(embedding_df)
    aidf['sort_order'] = sorted_indices

    # do dimensionality reduction on embedding_df and cluster analysis
    log("Load umap dimensionality reduction model")
    with open("reducer.pkl", 'rb') as pklfile:
        # Load the model from the file
        reducer = pickle.load(pklfile)
    log("Perform dimensionality reduction")
    reduced_data = reducer.transform(embedding_df)
    log("Cluster with DBSCAN")
    # Adjust eps and min_samples as needed
    dbscan = DBSCAN(eps=0.4, min_samples=3)
    aidf['cluster'] = dbscan.fit_predict(reduced_data)
    log(f"Found {len(aidf['cluster'].unique())-1} clusters")
    aidf.loc[aidf['cluster'] == -1, 'cluster'] = 999

    # sort first by clusters found by DBSCAN, then by semantic ordering
    aidf = aidf.sort_values(['cluster', 'sort_order']).reset_index(
        drop=True).reset_index().drop(columns=["id"]).rename(columns={'index': 'id'})

    # show clusters
    state["cluster_topics"] = []
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
        for i in range(30):
            try:
                tmpdf = aidf.loc[aidf['cluster'] ==
                                 i][["title_topic_str"]]
                if len(tmpdf) == 0:
                    break
                display(tmpdf)
                response = asyncio.run(filter_page_async(
                    tmpdf,
                    TOPIC_WRITER_SYSTEM_PROMPT,
                    TOPIC_WRITER_USER_PROMPT,
                    TopicHeadline,
                    model=model_low,
                ))
                cluster_topic = response.topic_title
                state["cluster_topics"].append(cluster_topic)
                log(f"I dub this cluster: {cluster_topic}")
                # should use topic_index = len(state["cluster_topics"]-1
                aidf["cluster_name"] = aidf['cluster'].apply(lambda i: state["cluster_topics"][i]
                                                             if i < len(state["cluster_topics"])
                                                             else "")

            except Exception as exc:
                log(exc)

    # send mail
    state["bullets"] = [make_bullet(row) for row in aidf.itertuples()]
    markdown_list = [f"{1 + row.id}. " +
                     make_bullet(row, include_topics=False)
                     for row in aidf.itertuples()]
    markdown_str = "\n\n".join(markdown_list)
    # summaries come back with bullet chars but let's make them markdown bullets
    markdown_str = markdown_str.replace('•', '\n  - ')

    # save bullets
    with open('bullets.md', 'w', encoding="utf-8") as f:
        f.write(markdown_str)

    state["AIdf"] = aidf.to_dict(orient='records')
    log(state["cluster_topics"])
    return state

# scrape individual pages


def fn_download_pages(state: AgentState) -> AgentState:
    """
    Uses several Playwright browser sessions to download all the pages referenced in the
    state["AIdf"] DataFrame and store their pathnames.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent with the downloaded pages' pathnames stored in the `state["AIdf"]` DataFrame.
    """
    log("Queuing URLs for scraping")
    aidf = pd.DataFrame(state['AIdf'])

    # Create a queue for multiprocessing and populate it
    queue = asyncio.Queue()
    for row in aidf.itertuples():
        asyncio.run(queue.put((row.id, row.url, row.title)))

    log(
        f"Saving HTML files using async concurrency= {state['n_browsers']}")
    # TODO: update this to get age of page in days for recency bonus
    saved_pages = asyncio.run(fetch_queue(queue, state['n_browsers']))

    pages_df = pd.DataFrame(saved_pages)
    if len(pages_df):
        pages_df.columns = ['id', 'url', 'title', 'path', 'last_updated']

        try:  # for idempotency
            aidf = aidf.drop(columns=['path', 'last_updated'])
        except Exception as exc:
            pass
            # error expected, no need to print
            # log("fn_download_pages")
            # log(exc)
        aidf = pd.merge(
            aidf, pages_df[["id", "path", "last_updated"]], on='id', how="inner")
    state["AIdf"] = aidf.to_dict(orient='records')
    # Pickle AIdf to AIdf.pkl
    aidf.to_pickle("AIdf.pkl")
    return state


def fn_summarize_pages(state: AgentState, model_medium) -> AgentState:
    """
    Reads all the articles, summarizes each one using a ChatGPT prompt, and sends an email with the summaries.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent.

    """
    # possible TODO: do a GPT self-check to evaluate summaries against facts in the article and correct
    log("Starting summarize")
    aidf = pd.DataFrame(state['AIdf'])
    responses = []
    try:
        responses = asyncio.run(fetch_all_summaries(aidf, model=model_medium))
        # responses = sfetch_all_summaries(aidf, model=model_medium)
    except Exception as e:
        log("Error fetching summaries")
        log(e)
    log(f"Received {len(responses)} summaries")
    # pdb.set_trace()
    response_dict = {}
    article_len_dict = {}
    for response, i, article_len in responses:
        response_dict[i] = response
        article_len_dict[i] = article_len

    aidf["summary"] = aidf["id"].map(response_dict.get)
    aidf["article_len"] = aidf["id"].map(article_len_dict.get)
    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def run_elo_round(elo_ranking_dict, battle_history_dict):
    """
    Run one round of an even-numbered tournament:
      - ids: list of player IDs (ints)
      - elo_ranking_dict: mutable dict of floats, indexed by ID
      - battle_history_dict: dict mapping (smaller_id, larger_id) -> result (1 = smaller won, -1 = smaller lost, 0 = draw)
      - K: Elo K‐factor (default 32)

    Returns:
      List of (id1, id2) pairs that were played this round.
    """
    n = len(elo_ranking_dict)
    ids = list(elo_ranking_dict.keys())
    n_pairs = n // 2
    # Find a random matching of unplayed pairs
    for _ in range(10000):
        shuffled = ids[:]
        random.shuffle(shuffled)
        pairs = [(shuffled[i*2], shuffled[i*2+1])
                 for i in range(0, n_pairs)]
        ordered_pairs = [(min(a, b), max(a, b)) for a, b in pairs]
        if all(p not in battle_history_dict for p in ordered_pairs):
            break
    else:
        log(
            "Couldn't find a full round of unplayed pairs after 10000 tries")
        return []

    return ordered_pairs


async def elo_battle(aidf, id1, id2, chain, max_retries=3):
    """
    Conduct a single 'battle' between two stories using the provided LLM chain.

    Args:
        aidf (pd.DataFrame): DataFrame containing articles.
        id1 (int): ID of the first story.
        id2 (int): ID of the second story.
        chain: A Prompt‐>Model‐>Parser chain that returns a string '1', '0', or '-1'.
        max_retries (int): How many times to retry on error or invalid output.

    Returns:
        int:  1 if story1 wins, -1 if story2 wins, 0 if draw.

    Raises:
        ValueError: if no valid response is obtained after max_retries.
    """
    # If either summary is empty, drop both
    headline1 = aidf.loc[id1, "title"]
    summary1 = aidf.loc[id1, "summary"]
    headline2 = aidf.loc[id2, "title"]
    summary2 = aidf.loc[id2, "summary"]

    if not summary1 or not summary2:
        summary1 = ""
        summary2 = ""

    payload = {
        "headline_A": headline1,
        "summary_A":  summary1,
        "headline_B": headline2,
        "summary_B":  summary2,
    }

    for attempt in range(1, max_retries + 1):
        try:
            raw = chain.invoke(payload)
        except Exception as e:
            log(f"ELO battle error ({headline1!r} vs {headline2!r}), attempt {attempt}: {e}")
            continue

        resp = raw.strip()
        if resp in {"1", "0", "-1"}:
            return int(resp), id1, id2

        # Unexpected output—log and retry
        log(f"Unexpected battle result '{resp}' for ({headline1!r} vs {headline2!r}), attempt {attempt}")

    # If we get here, all attempts failed
    raise ValueError(
        f"Failed to obtain valid battle result for {headline1!r} vs {headline2!r} "
        f"after {max_retries} attempts."
    )


async def run_elo_battles(aidf, pairs, chain):
    """
    Run a round of ELO battles between pairs of articles asynchronously.
    """
    tasks = [asyncio.create_task(elo_battle(
        aidf, a, b, chain)) for a, b in pairs]
    results = await asyncio.gather(*tasks)

    return results


def fn_rate_articles(state: AgentState, model_medium) -> AgentState:
    """
    calculate ratings for articles
    for these semantic understanding tasks, o4-mini seems like the one to use based on cookbook doc
    but it takes 5 minutes per elo round and is expensive
    so for now, use gpt-4.1-mini
    """
    aidf = pd.DataFrame(state['AIdf']).fillna({
        'article_len': 1,
        'reputation': 0,
        'on_topic': 0,
        'importance': 0,
        'low_quality': 0,
    })
    log(f"Calculating article rating for {len(aidf)} articles")

    # low quality articles
    aidf = filter_df(aidf, model_medium, LOW_QUALITY_SYSTEM_PROMPT,
                     LOW_QUALITY_USER_PROMPT, "low_quality", "bullet", "summary")
    counts = aidf["low_quality"].value_counts().to_dict()
    log(f"low quality articles: {counts}")

    # on topic articles
    aidf = filter_df(aidf, model_medium, ON_TOPIC_SYSTEM_PROMPT,
                     ON_TOPIC_USER_PROMPT, "on_topic", "bullet", "summary")
    counts = aidf["on_topic"].value_counts().to_dict()
    log(f"on topic articles: {counts}")

    # important articles
    aidf = filter_df(aidf, model_medium, IMPORTANCE_SYSTEM_PROMPT,
                     IMPORTANCE_USER_PROMPT, "importance", "bullet", "summary")
    counts = aidf["importance"].value_counts().to_dict()
    log(f"important articles: {counts}")

    # AI is good at yes or no questions, not necessarily at converting understanding
    # to a rating. Use ELO to rate articles based on a series of pairwise comparisons
    log("running ELO rating")
    n_rounds = max(2, math.ceil(math.log(len(aidf))*2.5-2))
    elo_ranking_dict = {i: 1000 for i in aidf["id"]}
    battle_history_dict = {}
    k_factor = 32
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_BATTLE_SYSTEM_PROMPT),
        ("user", PROMPT_BATTLE_USER_PROMPT)
    ])
    chain = prompt | model_medium | StrOutputParser()
    log(f"number of rounds: {n_rounds}")
    for i in range(n_rounds):
        log(f"running ELO round {i+1} of {n_rounds}")
        pairs = run_elo_round(elo_ranking_dict, battle_history_dict)
        if not pairs:
            break
        results = asyncio.run(run_elo_battles(aidf, pairs, chain))
        for result, a, b in results:
            # Store result
            battle_history_dict[(a, b)] = result
            # Convert result into scores
            if result == 1:
                score_a, score_b = 1.0, 0.0
                log(f"'{aidf.loc[a, 'title']}' > '{aidf.loc[b, 'title']}'")
            elif result == -1:
                score_a, score_b = 0.0, 1.0
                log(f"'{aidf.loc[b, 'title']}' > '{aidf.loc[a, 'title']}'")
            else:
                score_a = score_b = 0.5
                log(f"'{aidf.loc[b, 'title']}' == '{aidf.loc[a, 'title']}'")

            # Current ratings
            rank_a = elo_ranking_dict[a]
            rank_b = elo_ranking_dict[b]

            # Expected scores
            expected_a = 1.0 / (1.0 + 10 ** ((rank_b - rank_a) / 400))
            expected_b = 1.0 / (1.0 + 10 ** ((rank_a - rank_b) / 400))

            # Update ratings in-place
            elo_ranking_dict[a] += k_factor * (score_a - expected_a)
            elo_ranking_dict[b] += k_factor * (score_b - expected_b)

    log("finished ELO rating")
    aidf['elo'] = aidf['id'].map(elo_ranking_dict.get)
    aidf["elo_z"] = (aidf["elo"] - aidf["elo"].mean()) / \
        aidf["elo"].std(ddof=0)

    # add points for freshness
    aidf["age"] = (datetime.now(timezone.utc) -
                   pd.to_datetime(aidf['last_updated']))
    aidf["age"] = aidf["age"].dt.total_seconds() / (24 * 60 * 60)
    aidf["age"] = aidf["age"].clip(lower=0)  # no negative dates
    # only consider articles from the last week
    aidf = aidf[aidf["age"] < 7].copy()
    k = np.log(2)  # 1/2 after 1 day
    # 1 point at age 0, 0 at age 1, -0.5 at age 2, -1 at age infinity
    aidf["recency_score"] = 2 * np.exp(-k * aidf["age"]) - 1

    # could test if the prompts bias for/against certain types of stories, adjust the prompts, or boost ratings if they match those topics

    # len < 100 -> 0
    # len > 10000 -> 2
    # in between log10(x) - 2
    aidf['adjusted_len'] = np.log10(aidf['article_len']) - 2
    aidf['adjusted_len'] = aidf['adjusted_len'].clip(lower=0, upper=2)

    aidf['rating'] = aidf['reputation'] \
        + aidf['adjusted_len'] \
        + aidf['on_topic'] \
        + aidf['importance'] \
        - aidf['low_quality'] \
        + aidf['elo_z'] / 2 \
        + aidf['recency_score']
    # filter out low rated articles
    aidf = aidf[aidf['rating'] >= MINIMUM_STORY_RATING].copy()
    log(f"articles after rating: {len(aidf)}")
    # redo bullets with topics and rating
    aidf["bullet"] = aidf.apply(make_bullet, axis=1)

    # insert into db to keep a record and eventually train models on summaries
    # Only keep the columns you want to insert
    cols = ['url', 'src', 'site_name', 'hostname',
            'title', 'actual_url', 'bullet', 'rating']
    records = aidf[cols].to_records(index=False)
    rows = list(records)

    conn = sqlite3.connect('articles.db')
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO daily_summaries
    (url, src, site_name, hostname, title, actual_url, bullet, rating)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_sql, rows)
    conn.commit()
    conn.close()

    # Convert Markdown to HTML and send mail
    markdown_extensions = [
        # 'tables',
        # 'fenced_code',
        # 'codehilite',
        'attr_list',
        'def_list',
        # 'footnotes',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists'
    ]

    markdown_str = "\n\n".join(aidf['bullet'])
    html_str = markdown.markdown(markdown_str, extensions=markdown_extensions)
    with open('bullets.html', 'w', encoding="utf-8") as f:
        f.write(html_str)

    # send email html_str
    log("Sending bullet points email")
    subject = f'AI news bullets {datetime.now().strftime("%H:%M:%S")}'
    send_gmail(subject, html_str)

    # same with a delimiter and no ID, to save as a txt file to use downstream
    bullet_str = "\n~~~\n".join(aidf['bullet'])
    with open('bullet_str.txt', 'w', encoding='utf-8') as f:
        f.write(bullet_str)

    state["AIdf"] = aidf.to_dict(orient='records')
    return state


def fn_propose_topics(state: AgentState, model_high: any) -> AgentState:
    """
    ask LLM to analyze for top categories
    """
    log(f"Proposing topic clusters using {str(type(model_high))}")

    aidf = pd.DataFrame(state["AIdf"])
    # state["cluster_topics"] should already have cluster names
    state["topics_str"] = '\n'.join(state['cluster_topics'])
    log(f"Initial cluster topics: \n{state['topics_str']}")

    # first extract free-form topics and add to cluster topics
    pages = paginate_df(aidf[["bullet"]], maxpagelen=200)
    response = asyncio.run(process_dataframes(
        pages,
        TOP_CATEGORIES_SYSTEM_PROMPT,
        TOP_CATEGORIES_USER_PROMPT,
        TopicCategoryList,
        model=model_high,
    ))
    state["cluster_topics"].extend(response)
    state["topics_str"] = '\n'.join(state['cluster_topics'])
    log(
        f"Added cluster topics using {str(type(model_high))}: \n{state['topics_str']}"
    )

    # deduplicate and edit topics
    response = asyncio.run(process_dataframes(
        [pd.DataFrame(state["cluster_topics"], columns=['topics'])],
        TOPIC_REWRITE_SYSTEM_PROMPT,
        TOPIC_REWRITE_USER_PROMPT,
        TopicCategoryList,
        model=model_high))

    state["cluster_topics"] = response
    state["topics_str"] = '\n'.join(state['cluster_topics'])
    log(
        f"Final edited cluster topics using {str(type(model_high))}:\n{state['topics_str']}"
    )

    # save topics to local file
    try:
        filename = 'topics.txt'
        with open(filename, 'w', encoding="utf-8") as topicfile:
            topicfile.write(state["topics_str"])
        log(f"Topics successfully saved to {filename}.")
    except Exception as e:
        log(f"An error occurred: {e}")

    return state


def fn_compose_summary(state: AgentState, model_high: any) -> AgentState:
    """Compose summary using FINAL_SUMMARY_PROMPT"""

    log(f"Composing summary using {str(type(model_high))}")
    aidf = pd.DataFrame(state["AIdf"])
    aidf['cluster_name'] = aidf['cluster_name'].fillna('')
    aidf["prompt_input"] = aidf.apply(
        lambda row: f"{row['title_topic_str']}\n\n{row['summary']}", axis=1)

    # make copy of aidf with rows where column 'cluster_name' is null or empty string
    aidf_temp = aidf.loc[aidf['cluster_name'].str.len() == 0].copy()
    router_system_prompt = TOPIC_ROUTER_SYSTEM_PROMPT.format(
        topics=state["topics_str"])
    # gets a df back
    routed_topics = asyncio.run(
        filter_df_rows(aidf_temp,
                       model_high,
                       router_system_prompt,
                       TOPIC_ROUTER_USER_PROMPT,
                       "cluster_name",
                       "prompt_input",
                       output_class=SingleTopicSpec,
                       output_class_label="topic")
    )

    # copy routed topics back to aidf
    aidf.loc[aidf.index.isin(routed_topics.index),
             "cluster_name"] = routed_topics["cluster_name"]

    # get unique cluster names and sort them
    cluster_df = aidf["cluster_name"].value_counts().reset_index()
    cluster_df.columns = ["cluster_name", "count"]
    log(cluster_df.to_dict(orient='records'))

    deduped_dfs = []
    for cluster_name in cluster_df["cluster_name"]:
        tmpdf = aidf.loc[aidf["cluster_name"] == cluster_name].sort_values(
            "rating", ascending=False).copy()
        if len(tmpdf) > 1:  # at least 2 to dedupe
            log(f"Deduping cluster: {cluster_name}")
            # apply filter_df to tmpdf using DEDUPLICATE_SYSTEM_PROMPT and DEDUPLICATE_USER_PROMPT
            # need a class to output
            deduped_dfs.append(
                filter_df(tmpdf,
                          model_high,
                          DEDUPLICATE_SYSTEM_PROMPT,
                          DEDUPLICATE_USER_PROMPT,
                          "dupe_id",
                          "prompt_input",
                          "topic"))
    # concatenate deduped_dfs into a single df
    deduped_df = pd.concat(deduped_dfs)
    # merge dupe_id into aidf
    try:
        aidf = aidf.drop("dupe_id", axis=1)
    except:
        pass
    aidf = pd.merge(aidf, deduped_df[["id", "dupe_id"]], on="id", how="left")
    # count number of rows in aidf where dupe_id is >0 and group by dupe_id
    dupe_counts = aidf.loc[aidf['dupe_id'] > 0].groupby('dupe_id').size()
    log(dupe_counts)
    # for each dupe_id in dupe_counts, add the count to the rating of that id
    for dupe_id in dupe_counts.index:
        aidf.loc[aidf['id'] == dupe_id, 'rating'] += dupe_counts[dupe_id]

    # drop rows where dupe_id is >= 0 (keep rows where dupe_id is -1, ie unique)
    aidf = aidf.loc[aidf['dupe_id'] < 0]
    # trim to < 10
    aidf['rating'] = aidf['rating'].clip(lower=0, upper=10)
    log(f"After deduping: {len(aidf)} rows")

    # sort by cluster and rating
    aidf = aidf.sort_values(
        by=['cluster_name', 'rating'], ascending=[True, False])

    # recompute bullets with updated ratings and cluster_name
    aidf["bullet"] = aidf.apply(make_bullet, axis=1)
    bullet_str = "\n~~~\n".join(aidf['bullet'])
    cat_str = state['topics_str']

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", FINAL_SUMMARY_SYSTEM_PROMPT),
        ("user", FINAL_SUMMARY_USER_PROMPT)
    ])
    ochain = prompt_template | model_high.with_structured_output(Newsletter)
    response = ochain.invoke(dict(cat_str=cat_str, bullet_str=bullet_str))
    state["summary"] = str(response)
    # save bullet_str to local file
    try:
        filename = 'summary.md'
        with open(filename, 'w', encoding="utf-8") as summaryfile:
            summaryfile.write(state.get("summary"))
            log(f"Markdown content successfully saved to {filename}.")
    except Exception as e:
        log(f"An error occurred: {e}")

    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def fn_rewrite_summary(state: AgentState, model_high) -> AgentState:
    """Edit summary using REWRITE_PROMPT"""
    # possible TODO: evaluate with flesch-kincaid readability score
    log(f"Rewriting summary using {str(type(model_high))}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM_PROMPT),
        ("user", REWRITE_USER_PROMPT)
    ])
    # openai_model = ChatOpenAI(model="o3-mini", reasoning_effort="high")
    ochain = prompt_template | model_high | StrOutputParser()
    response_str = ochain.invoke({'summary': state["summary"]})

    state["n_edits"] += 1
    if response_str.strip().lower().startswith('ok'):
        log("No edits made, edit complete")
        state["edit_complete"] = True
    else:
        state["summary"] = response_str
    return state


def fn_is_revision_complete(state: AgentState) -> str:
    """update edit_complete if MAX_EDITS exceeded"
    return "complete" if edit_complete else "incomplete"
    """

    if state["n_edits"] >= state["max_edits"]:
        log("Max edits reached")
        state["edit_complete"] = True

    return "complete" if state["edit_complete"] else "incomplete"


def fn_send_mail(state: AgentState) -> AgentState:
    """Send email with state['summary']"""
    log("Sending summary email")
    # Convert Markdown to HTML
    html_str = markdown.markdown(state['summary'], extensions=['extra'])
    # extract subject, match a top-level Markdown heading (starts with "# ")
    match = re.search(r"^# (.+)$", state["summary"], re.MULTILINE)

    # If a match is found, return the first captured group (the heading text)
    if match:
        subject = match.group(1).strip()
    else:
        subject = f'AI news summary {datetime.now().strftime("%H:%M:%S")}'
    log(f"Email subject {subject}")
    log(f"Email length {len(html_str)}")

    # send email
    send_gmail(subject, html_str)
    return state

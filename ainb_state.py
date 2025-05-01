
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
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import asyncio
from typing import TypedDict
from urllib.parse import urlparse
import sqlite3

import yaml
import markdown

import requests

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

from ainb_llm import (paginate_df, process_dataframes, fetch_all_summaries,
                      filter_page_async,
                      get_all_canonical_topic_results, clean_topics,
                      Stories, TopicSpecList, TopicHeadline, TopicCategoryList, Sites,
                      StoryRatings,  # StoryRatings,
                      Newsletter
                      )
# from ainb_sllm import sfetch_all_summaries
from ainb_webscrape import (
    parse_file, fetch_queue, fetch_source_queue)
from ainb_utilities import (log, delete_files, filter_unseen_urls_db,
                            nearest_neighbor_sort, send_gmail, unicode_to_ascii)
from ainb_const import (DOWNLOAD_DIR, PAGES_DIR, SOURCECONFIG, SOURCES_EXPECTED,
                        CANONICAL_TOPICS, SQLITE_DB,
                        HOSTNAME_SKIPLIST, SITE_NAME_SKIPLIST, SOURCE_REPUTATION,
                        SCREENSHOT_DIR)

from ainb_prompts import (
    FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT,
    TOPIC_SYSTEM_PROMPT, TOPIC_USER_PROMPT,
    TOPIC_WRITER_SYSTEM_PROMPT, TOPIC_WRITER_USER_PROMPT,
    TOPIC_REWRITE_SYSTEM_PROMPT, TOPIC_REWRITE_USER_PROMPT,
    LOW_QUALITY_SYSTEM_PROMPT, LOW_QUALITY_USER_PROMPT,
    ON_TOPIC_SYSTEM_PROMPT, ON_TOPIC_USER_PROMPT,
    IMPORTANCE_SYSTEM_PROMPT, IMPORTANCE_USER_PROMPT,
    TOP_CATEGORIES_SYSTEM_PROMPT, TOP_CATEGORIES_USER_PROMPT,
    FINAL_SUMMARY_SYSTEM_PROMPT, FINAL_SUMMARY_USER_PROMPT,
    REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT,
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
        topic_str = "\n\nTopics: " + \
            str(row.topic_str) if hasattr(row, "topic_str") else ""
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
            print("fn_initialize")
            print(exc)

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
#         print('ERROR: API call failed.')
#         print(results)

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
#         print('ERROR: API call failed.')
#         print(response)
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
    NEWS_API_KEY = os.environ['NEWSAPI_API_KEY']
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
        'apiKey': NEWS_API_KEY,
        'pageSize': 100
    }

    # Make API call with headers and params
    response = requests.get(baseurl, params=params, timeout=60)
    if response.status_code != 200:
        print('ERROR: API call failed.')
        print(response.text)

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
        # print("fn_filter_urls")
        # print(exc)

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
        new_urls = []
        for r in responses:
            if r.url.startswith('https://'):
                r.url = r['url'][8:]
            elif r.url.startswith('http://'):
                r.url = r['url'][7:]
            new_urls.append(r['url'])
            sites_dict[r['url']] = r.site_name
            log(f"Looked up {r['url']} -> {r['site_name']}")
        # update sites table with new names
        for url in new_urls:
            sqlstr = "INSERT OR IGNORE INTO sites (hostname, site_name) VALUES (?, ?);"
            log(f"Updated {url} -> {sites_dict[url]}")
            conn.execute(sqlstr, (url, sites_dict[url]))
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

    # TODO: could add a quality rating for stories based on site reputation, length, complexity of story
    # could then add the quality rating to the summaries and tell the prompt to favor high-quality stories
    # could put summaries into vector store and retrieve stories by topic. but then you will have to deal
    # with duplicates across categories, ask the prompt to dedupe

    langchain.verbose = True
    aidf = pd.DataFrame(state['AIdf'])
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

    # Convert Markdown to HTML
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

    html_str = markdown.markdown(markdown_str, extensions=markdown_extensions)
    with open('bullets.html', 'w', encoding="utf-8") as f:
        f.write(html_str)

    # send email html_str
    log("Sending bullet points email")
    subject = f'AI news bullets {datetime.now().strftime("%H:%M:%S")}'
    send_gmail(subject, html_str)

    # same with a delimiter and no ID
    bullet_str = "\n~~~\n".join(aidf['bullet'])
    with open('bullet_str.txt', 'w', encoding='utf-8') as f:
        f.write(bullet_str)

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
    # pdb.set_trace()
    saved_pages = asyncio.run(fetch_queue(queue, state['n_browsers']))

    pages_df = pd.DataFrame(saved_pages)
    if len(pages_df):
        pages_df.columns = ['id', 'url', 'title', 'path']

        try:  # for idempotency
            aidf = aidf.drop(columns=['path'])
        except Exception as exc:
            pass
            # error expected, no need to print
            # print("fn_download_pages")
            # print(exc)
        aidf = pd.merge(aidf, pages_df[["id", "path"]], on='id', how="inner")
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
    log("Starting summarize")
    aidf = pd.DataFrame(state['AIdf'])
    responses = []
    try:
        responses = asyncio.run(fetch_all_summaries(aidf, model=model_medium))
        # responses = sfetch_all_summaries(aidf, model=model_medium)
    except Exception as e:
        log("Error fetching summaries")
        print(e)
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

# could abstract this to a generic filter function
# description , system_prompt, user_prompt, output_class, model, column_name, filter_value
# then roll all filters into the rating node


def fn_quality_filter(state: AgentState, model_medium) -> AgentState:
    "rate low quality articles using a prompt"
    log("Starting quality filter")
    aidf = pd.DataFrame(state['AIdf'])
    qdf = aidf[["id", "bullet"]].copy().rename(columns={"bullet": "summary"})
    pages = paginate_df(qdf, maxpagelen=50)
    responses = asyncio.run(process_dataframes(
        pages,
        LOW_QUALITY_SYSTEM_PROMPT,
        LOW_QUALITY_USER_PROMPT,
        StoryRatings,
        model=model_medium,
    ))
    response_dict = {}
    for response_obj in responses:
        response_dict[response_obj.id] = response_obj.rating
    aidf["low_quality"] = aidf["id"].map(response_dict.get)
    counts_dict = aidf['low_quality'].value_counts().to_dict()
    log(f"value counts: {counts_dict}")
    # aidf = aidf.loc[aidf['low_quality'] != 1]
    log(f"retained {len(aidf)} articles after applying quality filter")
    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def fn_on_topic_filter(state: AgentState, model_medium) -> AgentState:
    "rate relevant articles using a prompt"

    log("Starting on-topic filter")
    aidf = pd.DataFrame(state['AIdf'])
    qdf = aidf[["id", "bullet"]].copy().rename(columns={"bullet": "summary"})
    pages = paginate_df(qdf, maxpagelen=50)
    responses = asyncio.run(process_dataframes(
        pages,
        ON_TOPIC_SYSTEM_PROMPT,
        ON_TOPIC_USER_PROMPT,
        StoryRatings,
        model=model_medium,
    ))
    response_dict = {}
    for response_obj in responses:
        response_dict[response_obj.id] = response_obj.rating
    aidf["on_topic"] = aidf["id"].map(response_dict.get)
    counts_dict = aidf['on_topic'].value_counts().to_dict()
    log(f"value counts: {counts_dict}")
    # aidf = aidf.loc[aidf['on_topic'] != 0]
    log(f"retained {len(aidf)} articles after applying on-topic filter")
    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def fn_importance_filter(state: AgentState, model_medium) -> AgentState:
    "rate important news articles using a prompt"

    log("Starting importance filter")
    aidf = pd.DataFrame(state['AIdf'])
    qdf = aidf[["id", "bullet"]].copy().rename(columns={"bullet": "summary"})
    pages = paginate_df(qdf, maxpagelen=50)
    responses = asyncio.run(process_dataframes(
        pages,
        IMPORTANCE_SYSTEM_PROMPT,
        IMPORTANCE_USER_PROMPT,
        StoryRatings,
        model=model_medium,
    ))
    response_dict = {}
    for response_obj in responses:
        response_dict[response_obj.id] = response_obj.rating
    aidf["importance"] = aidf["id"].map(response_dict.get)
    counts_dict = aidf['importance'].value_counts().to_dict()
    log(f"value counts: {counts_dict}")
    # aidf = aidf.loc[aidf['importance'] != 0]
    log(f"retained {len(aidf)} articles after applying importance filter")
    state['AIdf'] = aidf.to_dict(orient='records')
    return state


def fn_rate_articles(state: AgentState) -> AgentState:
    """
    calculate ratings for articles
    """
    log("Calculating article ratings")
    aidf = pd.DataFrame(state['AIdf']).fillna({
        'article_len': 1,
        'reputation': 0,
        'on_topic': 0,
        'importance': 0,
        'low_quality': 0,
    })
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
        # redo bullets with topics
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

    return state


def fn_rewrite_summary(state: AgentState, model_high) -> AgentState:
    """Edit summary using REWRITE_PROMPT"""

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

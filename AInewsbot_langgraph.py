# main AInewsbot agent and top level imports
import pdb
import os
from datetime import datetime, timedelta
# import dotenv
# import subprocess
import json
import uuid
import re
import pickle
import argparse
from collections import Counter, defaultdict
import asyncio

from typing import TypedDict

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import sqlite3
from urllib.parse import urlparse  # , urljoin

import yaml

import requests

from IPython.display import display  # , Markdown  # , Audio
import pandas as pd
from sklearn.cluster import DBSCAN

# for token count
# todo, count tokens depending on model
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.prompts import ChatPromptTemplate
# JsonOutputParser, StrOutputParser
# from langchain_core.output_parsers import SimpleJsonOutputParser

from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
from langgraph.errors import NodeInterrupt

import markdown

import nest_asyncio

from ainb_llm import (paginate_df, process_dataframes, fetch_all_summaries,
                      filter_page_async,
                      get_all_canonical_topic_results, clean_topics,
                      Stories, TopicSpecList, TopicHeadline, TopicCategoryList, Sites,
                      sprocess_dataframes
                      )

from ainb_webscrape import (get_browsers, parse_file,
                            process_source_queue_factory,
                            process_url_queue_factory)

from ainb_utilities import (log, delete_files, filter_unseen_urls_db, insert_article,
                            nearest_neighbor_sort, send_gmail, unicode_to_ascii)

from ainb_const import (DOWNLOAD_DIR, PAGES_DIR,
                        CANONICAL_TOPICS,
                        SOURCECONFIG, FILTER_PROMPT, TOPIC_PROMPT, TOPIC_WRITER_PROMPT,
                        FINAL_SUMMARY_PROMPT,
                        TOP_CATEGORIES_PROMPT, TOPIC_REWRITE_PROMPT, REWRITE_PROMPT,
                        SITE_NAME_PROMPT, SQLITE_DB,
                        HOSTNAME_SKIPLIST, SITE_NAME_SKIPLIST,
                        SCREENSHOT_DIR, NUM_BROWSERS, REQUEST_TIMEOUT)

nest_asyncio.apply()  # needed for asyncio.run to work under langgraph

# defaults if called via import and not __main__
N_BROWSERS = 4
MAX_EDITS = 2

newscatcher_sources = ['247wallst.com',
                       '9to5mac.com',
                       'androidauthority.com',
                       'androidcentral.com',
                       'androidheadlines.com',
                       'appleinsider.com',
                       'benzinga.com',
                       'cnet.com',
                       'cnn.com',
                       'digitaltrends.com',
                       'engadget.com',
                       'fastcompany.com',
                       'finextra.com',
                       'fintechnews.sg',
                       'fonearena.com',
                       'ft.com',
                       'gadgets360.com',
                       'geekwire.com',
                       'gizchina.com',
                       'gizmochina.com',
                       'gizmodo.com',
                       'gsmarena.com',
                       'hackernoon.com',
                       'howtogeek.com',
                       'ibtimes.co.uk',
                       'itwire.com',
                       'lifehacker.com',
                       'macrumors.com',
                       'mashable.com',
                       #  'medium.com',
                       'mobileworldlive.com',
                       'msn.com',
                       'nypost.com',
                       'phonearena.com',
                       'phys.org',
                       'popsci.com',
                       'scmp.com',
                       'sify.com',
                       'siliconangle.com',
                       'siliconera.com',
                       'siliconrepublic.com',
                       'slashdot.org',
                       'slashgear.com',
                       'statnews.com',
                       'tech.co',
                       'techcrunch.com',
                       'techdirt.com',
                       'technode.com',
                       'technologyreview.com',
                       'techopedia.com',
                       'techradar.com',
                       'techraptor.net',
                       'techtimes.com',
                       'techxplore.com',
                       'telecomtalk.info',
                       'thecut.com',
                       'thedrum.com',
                       'thehill.com',
                       'theregister.com',
                       'theverge.com',
                       'thurrott.com',
                       'tipranks.com',
                       'tweaktown.com',
                       'videocardz.com',
                       'washingtonpost.com',
                       'wccftech.com',
                       'wired.com',
                       'xda-developers.com',
                       'yahoo.com',
                       'zdnet.com']


model_family = {'gpt-4o-2024-11-20': 'openai',
                'gpt-4o-mini': 'openai',
                'o3-mini': 'openai',
                'models/gemini-2.0-flash-thinking-exp': 'google',
                'models/gemini-2.0-pro-exp': 'google',
                'models/gemini-2.0-flash': 'google',
                }


def get_model(model_name):
    """get langchain model based on model_name"""
    if model_name in model_family:
        model_type = model_family[model_name]
        if model_type == 'openai':
            return ChatOpenAI(model=model_name, request_timeout=REQUEST_TIMEOUT)
        elif model_type == 'google':
            return ChatGoogleGenerativeAI(model=model_name, request_timeout=REQUEST_TIMEOUT)
    else:
        log(f"Unknown model {model_name}")
        return None

# print(f"Python            {sys.version}")
# print(f"LangChain         {langchain.__version__}")
# print(f"OpenAI            {openai.__version__}")
# # print(f"smtplib           {smtplib.sys.version}")
# print(f"trafilatura       {trafilatura.__version__}")
# # print(f"bs4               {bs4.__version__}")
# print(f"numpy             {np.__version__}")
# print(f"pandas            {pd.__version__}")
# print(f"sklearn           {sklearn.__version__}")
# print(f"umap              {umap.__version__}")
# print(f"podcastfy         {podcastfy.__version__}")


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
    sources: dict  # sources to scrap
    sources_reverse: dict[str, str]  # map file names to sources
    bullets: list[str]  # bullet points for summary email
    summary: str  # final summary
    cluster_topics: list[str]  # list of cluster topics
    topics_str: str  # edited topics
    n_edits: int  # count edit iterations so we don't keep editing forever
    edit_complete: bool  # edit will update if no more edits to make


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


def fn_download_sources(state: AgentState, BROWSERS: any) -> AgentState:
    """
    Scrapes sources and saves HTML files.
    If state["do_download"] is True, deletes all files in DOWNLOAD_DIR (htmldata) and scrapes fresh copies.
    If state["do_download"] is False, uses existing files in DOWNLOAD_DIR.
    Uses state["sources"] for config info on sources to scrape
    For each source, saves the current filename to state["sources"][sourcename]['latest']
    jina or firecrawl might be better for this
    Args:
        state (AgentState): The current state of the agent.
        do_delete (bool, optional): Whether to delete files in DOWNLOAD_DIR. Defaults to True.

    Returns:
        AgentState: The updated state of the agent.
    """

    if state.get("do_download"):
        # empty download directories
        delete_files(DOWNLOAD_DIR)
        delete_files(PAGES_DIR)
        delete_files(SCREENSHOT_DIR)

        # save each file specified from sources
        log(f"Saving HTML files using {N_BROWSERS} browsers")
        # Create a queue for multiprocessing and populate it
        queue = multiprocessing.Queue()
        for item in state.get("sources").values():
            queue.put(item)

        # create a closure to download urls in queue asynchronously
        closure = process_source_queue_factory(queue)

        if len(BROWSERS) < NUM_BROWSERS:
            BROWSERS = asyncio.run(get_browsers(NUM_BROWSERS))

        with ThreadPoolExecutor(max_workers=NUM_BROWSERS) as executor:
            # Create a list of future objects
            futures = [executor.submit(closure, BROWSERS[i])
                       for i in range(NUM_BROWSERS)]

            # Collect the results (web drivers) as they complete
            retarray = [future.result() for future in as_completed(futures)]

        # flatten results
        saved_pages = [item for retarray in retarray for item in retarray]

        for sourcename, sourcefile in saved_pages:
            log(f"Downloaded {sourcename} to {sourcefile}")
            state['sources'][sourcename]['latest'] = sourcefile
        log(f"Saved {len(saved_pages)} HTML files")

    else:   # use existing files
        log(f"Web fetch disabled, using existing files in {DOWNLOAD_DIR}")
        # Get the current date
        datestr = datetime.now().strftime("%m_%d_%Y")
        files = [os.path.join(DOWNLOAD_DIR, file)
                 for file in os.listdir(DOWNLOAD_DIR)]
        # filter files with today's date ending in .html
        files = [
            file for file in files if datestr in file and file.endswith(".html")]
        log(f"Found {len(files)} previously downloaded files")
        for sourcefile in files:
            log(sourcefile)

        saved_pages = []
        for sourcefile in files:
            filename = os.path.basename(sourcefile)
            # locate date like '01_14_2024' in filename
            position = filename.find(" (" + datestr)
            basename = filename[:position]
            # match to source name
            sourcename = state.get("sources_reverse", {}).get(basename)
            if sourcename is None:
                log(f"Skipping {basename}, no sourcename metadata")
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
    sources_expected = 16
    missing_sources = sources_expected-sources_downloaded

    if missing_sources:
        raise NodeInterrupt(
            f"{missing_sources} missing sources: Expected {sources_expected}")
    log(f"verify_download passed, found {sources_expected} sources in AIdf, {missing_sources} missing")
    return state


def fn_extract_newscatcher(state: AgentState) -> AgentState:
    """get AI news via newscatcher API
    https://docs.newscatcherapi.com/api-docs/endpoints/search-news
    """

    q = 'Artificial Intelligence'
    page_size = 100
    log(f"Fetching top {page_size} stories matching {q} from Newscatcher")
    base_url = "https://api.newscatcherapi.com/v2/search"
    time_24h_ago = datetime.now() - timedelta(hours=24)

    # Put API key in headers
    headers = {'x-api-key': os.getenv('NEWSCATCHER_API_KEY')}

    # Define search parameters
    params = {
        'q': q,
        'lang': 'en',
        'sources': ','.join(newscatcher_sources),
        'from': time_24h_ago.strftime('%Y-%m-%d %H:%M:%S'),
        'to': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'page_size': page_size,  # by default should be most highly relevant to the search
        'page': 1
    }

    # Make API call with headers and params
    response = requests.get(base_url, headers=headers,
                            params=params, timeout=60)

    # Encode received results
    results = json.loads(response.text.encode())
    if response.status_code != 200:
        print('ERROR: API call failed.')
        print(results)

    # merge into existing df
    newscatcher_df = pd.DataFrame(results['articles'])[['title', 'link']]
    newscatcher_df['src'] = 'Newscatcher'
    newscatcher_df = newscatcher_df.rename(columns={'link': 'url'})
#     display(newscatcher_df.head())
    aidf = pd.DataFrame(state['AIdf'])
#     display(AIdf.head())

    max_id = aidf['id'].max()
    # add id column to newscatcher_df
    newscatcher_df['id'] = range(max_id + 1, max_id + 1 + len(newscatcher_df))
    aidf = pd.concat([aidf, newscatcher_df], ignore_index=True)
    state['AIdf'] = aidf.to_dict(orient='records')
    return state


def fn_extract_newsapi(state: AgentState) -> AgentState:
    """
    get AI news via newsapi
    https://newsapi.org/docs/get-started
    from newsapi import NewsApiClient
    """
    NEWSAPI_API_KEY = os.environ['NEWSAPI_API_KEY']
    aidf = pd.DataFrame(state['AIdf'])

    pageSize = 100
    q = 'artificial intelligence'
    date_24h_ago = datetime.now() - timedelta(hours=24)
    formatted_date = date_24h_ago.strftime("%Y-%m-%dT%H:%M:%S")
    log(f"Fetching top {pageSize} stories matching {q} since {formatted_date} from NewsAPI")

    baseurl = 'https://newsapi.org/v2/everything'

    # Define search parameters
    params = {
        'q': q,
        'from': formatted_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWSAPI_API_KEY,
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
        .reset_index(drop=False) \
        .drop(columns=['id']) \
        .reset_index() \
        .rename(columns={'index': 'id'})
    log(f"Found {len(aidf)} unique new headlines")

    # # dedupe identical headlines
    # # filter similar titles differing by type of quote or something
    aidf['title'] = aidf['title'].apply(unicode_to_ascii)
    aidf['title_clean'] = aidf['title'].map(lambda s: " ".join(s.split()))
    aidf = aidf.sort_values("src") \
        .groupby("title_clean") \
        .first() \
        .reset_index(drop=True) \
        .drop(columns=['id']) \
        .reset_index() \
        .rename(columns={'index': 'id'})
    log(f"Found {len(aidf)} unique new headlines")
    # filter AI-related headlines using a prompt
    pagelist = paginate_df(aidf[["id", "title"]])
    results = asyncio.run(process_dataframes(
        dataframes=pagelist,
        input_prompt=FILTER_PROMPT,
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
    log(f"Inserting {len(aidf)} URLs into {SQLITE_DB}")
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    for row in aidf.itertuples():
        insert_article(conn, cursor, row.src, row.hostname, row.title,
                       row.url, row.actual_url, row.isAI, datetime.now().date())

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

    state["AIdf"] = aidf.to_dict(orient='records')
    return state


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
    return f"[{title} - {site_name}]({actual_url}){topic_str}{summary}\n\n"


def fn_topic_analysis(state: AgentState, model_medium: any) -> AgentState:
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

    aidf = pd.DataFrame(state['AIdf'])
    tmpdf = aidf[['id', 'title', 'summary']].copy()
    tmpdf["summary"] = tmpdf["title"] + "\n" + tmpdf["summary"]
    # seems to have had trouble with 50 headlines
    pages = paginate_df(tmpdf[["id", "summary"]], maxpagelen=20)

    # apply topic extraction prompt to AI headlines
    log(f"start free-form topic extraction using {str(type(model_medium))}")
    topic_results = sprocess_dataframes(
        dataframes=pages,
        input_prompt=TOPIC_PROMPT,
        output_class=TopicSpecList,
        model=model_medium,
        item_list_field="items",
        item_id_field="id"
    )
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

    log(f"Starting assigned topic extraction using {str(type(model_medium))}")
    assigned_topics = asyncio.run(
        get_all_canonical_topic_results(pages, lcategories, model_medium))
    ctr_dict = defaultdict(list)

    for (topic, relevant_list) in assigned_topics:
        for ctr in relevant_list:
            if ctr.relevant:
                ctr_dict[ctr.id].append(topic)

    topics_df['assigned_topics'] = topics_df['id'].apply(
        lambda id: ctr_dict.get(id, ""))

    log("Cleaning and formatting topics")
    # pdb.set_trace()
    topics_df["topics"] = topics_df.apply(
        lambda t: clean_topics(t, lcategories), axis=1)
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

    #     state["AIdf"] = AIdf.to_dict(orient='records')
    #     return state

    # redo bullets with topics
    aidf["bullet"] = aidf.apply(make_bullet, axis=1)
    state["AIdf"] = aidf.to_dict(orient='records')
    return state


# def fn_topic_analysis(state: AgentState) -> AgentState:
#     """call async afn_topic_analysis on state"""
#     state = asyncio.run(afn_topic_analysis(state))
#     return state


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
    log("Sort with nearest_neighbor_sort sort")
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
    aidf = aidf.sort_values(['cluster', 'sort_order']) \
        .reset_index(drop=True) \
        .reset_index() \
        .drop(columns=["id"]) \
        .rename(columns={'index': 'id'})

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
                    TOPIC_WRITER_PROMPT,
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

# TODO: could add a quality rating for stories based on site reputation, length, complexity of story
# could then add the quality rating to the summaries and tell the prompt to favor high-quality stories
# could put summaries into vector store and retrieve stories by topic. but then you will have to deal
# with duplicates across categories, ask the prompt to dedupe

# def fn_topic_clusters(state: AgentState) -> AgentState:
#     "call async afn_topic_clusters on state"
#     state = asyncio.run(afn_topic_clusters(state))
#     return state


# scrape individual pages
def fn_download_pages(state: AgentState, BROWSERS: any) -> AgentState:
    """
    Uses several Selenium browser sessions to download all the pages referenced in the
    state["AIdf"] DataFrame and store their pathnames.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent with the downloaded pages' pathnames stored in the `state["AIdf"]` DataFrame.
    """
    log("Queuing URLs for scraping")
    aidf = pd.DataFrame(state['AIdf'])
    queue = multiprocessing.Queue()

    count = 0
    for row in aidf.itertuples():
        #         if row.cluster < 999:
        queue.put((row.id, row.url, row.title))
        count += 1

    # scrape urls in queue asynchronously
    closure = process_url_queue_factory(queue)

    if len(BROWSERS) < NUM_BROWSERS:
        BROWSERS = asyncio.run(get_browsers(NUM_BROWSERS))

    with ThreadPoolExecutor(max_workers=NUM_BROWSERS) as executor:
        # Create a list of future objects
        futures = [executor.submit(closure, BROWSERS[i])
                   for i in range(NUM_BROWSERS)]

        # Collect the results (web drivers) as they complete
        retarray = [future.result() for future in as_completed(futures)]

    # flatten results
    saved_pages = [item for retarray in retarray for item in retarray]

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
    responses = asyncio.run(fetch_all_summaries(aidf, model=model_medium))
    log(f"Received {len(responses)} summaries")
    response_dict = {}
    for response, i in responses:
        response_dict[i] = response

    aidf["summary"] = aidf["id"].apply(lambda rowid: response_dict[rowid])
    state['AIdf'] = aidf.to_dict(orient='records')

    return state


def fn_propose_cats(state: AgentState, model_high: any) -> AgentState:
    """
    ask LLM to analyze for top categories
    """
    log(f"Proposing categories using {str(type(model_high))}")

    aidf = pd.DataFrame(state["AIdf"])
    # state["cluster_topics"] should already have cluster names
    state["topics_str"] = '\n'.join(state['cluster_topics'])
    log(f"Initial cluster topics: \n{state['topics_str']}")

    # first extract free-form topics and add to cluster topics
    pages = paginate_df(aidf[["bullet"]])
    response = asyncio.run(process_dataframes(
        pages,
        TOP_CATEGORIES_PROMPT,
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
        TOPIC_REWRITE_PROMPT,
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

    # check that actual prompt matches the template
    # print(FINAL_SUMMARY_PROMPT.format(cat_str=cat_str, bullet_str=bullet_str))

    prompt_template = ChatPromptTemplate.from_template(FINAL_SUMMARY_PROMPT)
    ochain = prompt_template | model_high | StrOutputParser()
    response = ochain.invoke(dict(cat_str=cat_str, bullet_str=bullet_str))
    state["summary"] = response
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

    prompt_template = ChatPromptTemplate.from_template(REWRITE_PROMPT)
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

    if state["n_edits"] >= MAX_EDITS:
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


class Agent:
    """Langraph Agent class"""

    def __init__(self, state, model_low, model_medium, model_high):
        """set up state graph and memory"""
        self.state = state

        self.BROWSERS = []
        self.model_low = get_model(model_low)
        self.model_medium = get_model(model_medium)
        self.model_high = get_model(model_high)

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("initialize", self.initialize)
        graph_builder.add_node("download_sources", self.download_sources)
        graph_builder.add_node("extract_web_urls", self.extract_web_urls)
        graph_builder.add_node("verify_download", self.verify_download)
        graph_builder.add_node("extract_newsapi_urls",
                               self.extract_newsapi_urls)
        # graph_builder.add_node("extract_newscatcher_urls",
        #                        self.extract_newscatcher_urls)
        graph_builder.add_node("filter_urls", self.filter_urls)
        graph_builder.add_node("topic_analysis", self.topic_analysis)
        graph_builder.add_node("topic_clusters", self.topic_clusters)
        graph_builder.add_node("download_pages", self.download_pages)
        graph_builder.add_node("summarize_pages", self.summarize_pages)
        graph_builder.add_node("propose_topics", self.propose_topics)
        graph_builder.add_node("compose_summary", self.compose_summary)
        graph_builder.add_node("rewrite_summary", self.rewrite_summary)
        graph_builder.add_node("send_mail", self.send_mail)

        graph_builder.add_edge(START, "initialize")
        graph_builder.add_edge("initialize", "download_sources")
        graph_builder.add_edge("download_sources", "extract_web_urls")
        graph_builder.add_edge("extract_web_urls", "verify_download")
        graph_builder.add_edge("verify_download", "extract_newsapi_urls")
        graph_builder.add_edge("extract_newsapi_urls", "filter_urls")
        # graph_builder.add_edge("extract_newscatcher_urls", "filter_urls")
        graph_builder.add_edge("filter_urls", "download_pages")
        graph_builder.add_edge("download_pages", "summarize_pages")
        graph_builder.add_edge("summarize_pages", "topic_analysis")
        graph_builder.add_edge("topic_analysis", "topic_clusters")
        graph_builder.add_edge("topic_clusters", "propose_topics")
        graph_builder.add_edge("propose_topics", "compose_summary")
        graph_builder.add_edge("compose_summary", "rewrite_summary")
        graph_builder.add_conditional_edges("rewrite_summary",
                                            self.is_revision_complete,
                                            {"incomplete": "rewrite_summary",
                                             "complete": "send_mail",
                                             })
        graph_builder.add_edge("send_mail", END)

        # human in the loop should check web pages downloaded ok, and edit proposed categories
        # self.conn = sqlite3.connect('lg_checkpointer.db')
        # self.checkpointer = SqliteSaver(conn=self.conn)
        self.checkpointer = MemorySaver()
        graph = graph_builder.compile(checkpointer=self.checkpointer,)
#                                      interrupt_before=["filter_urls", "compose_summary",])
        self.graph = graph

    def initialize(self, state: AgentState) -> AgentState:
        """initialize agent, loading sources and setting up initial state"""
        self.state = fn_initialize(state)
        return self.state

    def download_sources(self, state: AgentState) -> AgentState:
        """download sources or load exisitng sources"""
        self.state = fn_download_sources(state, self.BROWSERS)
        return self.state

    def extract_web_urls(self, state: AgentState) -> AgentState:
        """parse all urls from downloaded pages"""
        self.state = fn_extract_urls(state)
        return self.state

    def verify_download(self, state: AgentState) -> AgentState:
        """verify we found news stories from all sources"""
        self.state = fn_verify_download(state)
        return self.state

    def extract_newscatcher_urls(self, state: AgentState) -> AgentState:
        """extract newscatcher urls"""
        try:
            self.state = fn_extract_newscatcher(state)
        except KeyError:
            log("Newscatcher download failed")
        return self.state

    def extract_newsapi_urls(self, state: AgentState) -> AgentState:
        """extract newsapi urls"""
        try:
            self.state = fn_extract_newsapi(state)
        except KeyError:
            log("NewsAPI download failed")
        return self.state

    def filter_urls(self, state: AgentState) -> AgentState:
        """filter to previously unseen urls and AI-related headlines only"""
        self.state = fn_filter_urls(state, self.model_low)
        return self.state

    def topic_analysis(self, state: AgentState) -> AgentState:
        """extract and assign topics for each headline"""
        self.state = fn_topic_analysis(state, self.model_medium)
        return self.state

    def topic_clusters(self, state: AgentState) -> AgentState:
        """identify clusters of similar stores"""
        self.state = fn_topic_clusters(state, self.model_low)
        return self.state

    def download_pages(self, state: AgentState) -> AgentState:
        """download individual news pages and save text"""
        self.state = fn_download_pages(state, self.BROWSERS)
        return self.state

    def summarize_pages(self, state: AgentState) -> AgentState:
        """summarize each page into bullet points"""
        self.state = fn_summarize_pages(state, self.model_medium)
        return self.state

    def propose_topics(self, state: AgentState) -> AgentState:
        """use LLM to identify most popular and important topics"""
        self.state = fn_propose_cats(state, self.model_high)
        return self.state

    def compose_summary(self, state: AgentState) -> AgentState:
        """compose the first draft of the summary using bullets and topics"""
        self.state = fn_compose_summary(state, self.model_high)
        return self.state

    def rewrite_summary(self, state: AgentState) -> AgentState:
        """edit the summary, combine and sharpen items, add and improve titles"""
        self.state = fn_rewrite_summary(state, self.model_high)
        return self.state

    def is_revision_complete(self, state: AgentState) -> str:
        """check if summary should be revised"""
        return fn_is_revision_complete(state)

    def send_mail(self, state: AgentState) -> AgentState:
        """send final email with summary"""
        self.state = fn_send_mail(state)
        return self.state

    def run(self, state, runconfig):
        """run the agent"""
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(state, runconfig, stream_mode="values")
        for event in events:
            try:
                if event.get('summary'):
                    print('summary created')
                    display(event.get('summary').replace("$", "\\\\$"))
                elif event.get('bullets'):
                    print('bullets created')
                    display("\n\n".join(
                        event.get('bullets')).replace("$", "\\\\$"))
                elif event.get('cluster_topics'):
                    print('cluster topics created')
                    display("\n\n".join(event.get('cluster_topics')))
                elif event.get('AIdf'):
                    display(pd.DataFrame(event.get('AIdf')).groupby(
                        "src").count()[['id']])
                elif event.get('sources'):
                    print([k for k in event.get('sources').keys()])
            except Exception as exc:
                print('run exception')
                print(exc)

        return self.state


def initialize_agent(do_download, before_date, model_low, model_medium, model_high):
    """set initial state"""
    state = AgentState({
        'AIdf': [{}],
        'before_date': before_date,
        'do_download': do_download,
        'sources': {},
        'sources_reverse': {},
        'bullets': '',
        'summary': '',
        'cluster_topics': [],
        'topics_str': '',
        'n_edits': 0,
        'edit_complete': False,
    })
    thread_id = uuid.uuid4().hex
    log(f"Initializing with before_date={state.get('before_date')}, do_download={do_download}, thread_id={thread_id}"
        )
    return state, Agent(state, model_low, model_medium, model_high), thread_id


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nofetch', action='store_true', default=False,
                        help='Disable web fetch, use existing HTML files in htmldata directory')
    parser.add_argument('-d', '--before-date', type=str, default='',
                        help='Process articles before this date (YYYY-MM-DD HH:MM:SS format)')
    parser.add_argument('-b', '--browsers', type=int, default=4,
                        help='Number of browser instances to run in parallel (default: 4)')
    parser.add_argument('-e', '--max-edits', type=int, default=1,
                        help='Maximum number of summary rewrites')
    parser.add_argument('-l', '--model-low', type=str, default='gpt-4o-mini',
                        help='Model (low cost)')
    parser.add_argument('-m', '--model-medium', type=str, default='gpt-4o-2024-11-20',
                        help='Model (medium cost)')
    parser.add_argument('-h', '--model-high', type=str, default='o3-mini',
                        help='Model (high cost)')
    args = parser.parse_args()

    do_download = not args.nofetch
    before_date = args.before_date
    N_BROWSERS = args.browsers
    MAX_EDITS = args.max_edits
    log(f"Starting AInewsbot with do_download={do_download}, before_date='{before_date}', N_BROWSERS={N_BROWSERS}, MAX_EDITS={N_BROWSERS}")

    lg_state, lg_agent, thread_id = initialize_agent(
        do_download, before_date, args.model_low, args.model_medium, args.model_high)
    log(f"thread_id: {thread_id}")
    # save in case we want to get the last state from Sqlite and inpsect or resume in Jupyter
    with open('thread_id.txt', 'w', encoding="utf-8") as file:
        file.write(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    lg_state = lg_agent.run(lg_state, config)

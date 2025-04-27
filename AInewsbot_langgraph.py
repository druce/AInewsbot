from ainb_state import *

from ainb_llm import (paginate_df, process_dataframes, fetch_all_summaries,
                      filter_page_async,
                      get_all_canonical_topic_results, clean_topics,
                      Stories, TopicSpecList, TopicHeadline, TopicCategoryList, Sites,
                      StoryRatings,  # StoryRatings,
                      Newsletter
                      #   sprocess_dataframes
                      )
from ainb_webscrape import (get_browsers, parse_file,
                            process_source_queue_factory,
                            process_url_queue_factory)
from ainb_utilities import (log, delete_files, filter_unseen_urls_db,
                            nearest_neighbor_sort, send_gmail, unicode_to_ascii)
from ainb_const import (DOWNLOAD_DIR, PAGES_DIR, SOURCECONFIG, SOURCES_EXPECTED,
                        FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT,
                        TOPIC_SYSTEM_PROMPT, TOPIC_USER_PROMPT,
                        LOW_QUALITY_SYSTEM_PROMPT, LOW_QUALITY_USER_PROMPT,
                        ON_TOPIC_SYSTEM_PROMPT, ON_TOPIC_USER_PROMPT,
                        IMPORTANCE_SYSTEM_PROMPT, IMPORTANCE_USER_PROMPT,
                        FINAL_SUMMARY_SYSTEM_PROMPT, FINAL_SUMMARY_USER_PROMPT,
                        CANONICAL_TOPICS,
                        TOPIC_WRITER_SYSTEM_PROMPT, TOPIC_WRITER_USER_PROMPT,
                        REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT,
                        TOP_CATEGORIES_PROMPT, TOPIC_REWRITE_PROMPT,
                        SITE_NAME_PROMPT, SQLITE_DB,
                        HOSTNAME_SKIPLIST, SITE_NAME_SKIPLIST, SOURCE_REPUTATION,
                        SCREENSHOT_DIR, REQUEST_TIMEOUT,
                        MODEL_FAMILY, NEWSCATCHER_SOURCES)
from openai import OpenAI
from sklearn.cluster import DBSCAN
import pandas as pd
from IPython.display import display, Markdown  # , Audio
import requests
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeInterrupt
import langchain
import nest_asyncio
import numpy as np
import pdb
import os
from datetime import datetime, timedelta
import dotenv
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

import markdown


# from langchain_anthropic import ChatAnthropic


# for token count
# todo, count tokens depending on model

# main AInewsbot agent and top level imports


langchain.verbose = True

# from langchain_core.prompts import ChatPromptTemplate
# JsonOutputParser, StrOutputParser
# from langchain_core.output_parsers import SimpleJsonOutputParser

# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.graph.message import add_messages


nest_asyncio.apply()  # needed for asyncio.run to work under langgraph

# defaults if called via import and not __main__
N_BROWSERS = 4
MAX_EDITS = 2


def get_model(model_name):
    """get langchain model based on model_name"""
    if model_name in MODEL_FAMILY:
        model_type = MODEL_FAMILY[model_name]
        if model_type == 'openai':
            return ChatOpenAI(model=model_name, request_timeout=REQUEST_TIMEOUT)
        elif model_type == 'google':
            return ChatGoogleGenerativeAI(model=model_name, request_timeout=REQUEST_TIMEOUT, verbose=True)
    else:
        log(f"Unknown model {model_name}")
        return None

# print(f"Python            {sys.version}")
# print(f"LangChain         {langchain.__version__}")
# print(f"OpenAI            {openai.__version__}")
# print(f"smtplib           {smtplib.sys.version}")
# print(f"trafilatura       {trafilatura.__version__}")
# print(f"bs4               {bs4.__version__}")
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


def fn_download_sources(state: AgentState, browsers: list) -> AgentState:
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
        log(f"Saving HTML files using {state['n_browsers']} browsers")
        # Create a queue for multiprocessing and populate it
        queue = multiprocessing.Queue()
        for item in state.get("sources").values():
            queue.put(item)

        # create a closure to download urls in queue asynchronously
        closure = process_source_queue_factory(queue)

        if len(browsers) < state["n_browsers"]:
            browsers.extend(asyncio.run(get_browsers(state["n_browsers"])))

        with ThreadPoolExecutor(max_workers=state["n_browsers"]) as executor:
            # Create a list of future objects
            futures = [executor.submit(closure, browsers[i])
                       for i in range(state["n_browsers"])]

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


class Agent:
    """Langraph Agent class"""

    def __init__(self, state: AgentState):
        """set up state graph and memory"""
        self.state = state

        self.model_low = get_model(state["model_low"])
        self.model_medium = get_model(state["model_medium"])
        self.model_high = get_model(state["model_high"])
        self.BROWSERS = []

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("initialize", self.initialize_config)
        graph_builder.add_node("download_sources", self.download_sources)
        graph_builder.add_node("extract_web_urls", self.extract_web_urls)
        graph_builder.add_node("verify_download", self.verify_download)
        graph_builder.add_node("extract_newsapi_urls",
                               self.extract_newsapi_urls)
        graph_builder.add_node("filter_urls", self.filter_urls)
        graph_builder.add_node("download_pages", self.download_pages)
        graph_builder.add_node("summarize_pages", self.summarize_pages)
        graph_builder.add_node("topic_analysis", self.topic_analysis)
        graph_builder.add_node("topic_clusters", self.topic_clusters)
        graph_builder.add_node("quality_filter", self.quality_filter)
        graph_builder.add_node("on_topic_filter", self.on_topic_filter)
        graph_builder.add_node("importance_filter", self.importance_filter)
        graph_builder.add_node("rate_articles", self.rate_articles)
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
        graph_builder.add_edge("filter_urls", "download_pages")
        graph_builder.add_edge("download_pages", "summarize_pages")
        graph_builder.add_edge("summarize_pages", "topic_analysis")
        graph_builder.add_edge("topic_analysis", "topic_clusters")
        graph_builder.add_edge("topic_clusters", "quality_filter")
        graph_builder.add_edge("quality_filter", "on_topic_filter")
        graph_builder.add_edge("on_topic_filter", "importance_filter")
        graph_builder.add_edge("importance_filter", "rate_articles")
        graph_builder.add_edge("rate_articles", "propose_topics")
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

    def initialize_config(self, state: AgentState) -> AgentState:
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

    def filter_urls(self, state: AgentState, model_str: str = "") -> AgentState:
        """filter to previously unseen urls and AI-related headlines only"""
        model = get_model(model_str) if model_str else self.model_low
        self.state = fn_filter_urls(state, model)
        return self.state

    def download_pages(self, state: AgentState) -> AgentState:
        """download individual news pages and save text"""
        # print(len(self.BROWSERS))
        self.state = fn_download_pages(state, self.BROWSERS)
        # print(len(self.BROWSERS))
        return self.state

    def summarize_pages(self, state: AgentState, model_str: str = "") -> AgentState:
        """summarize each page into bullet points"""
        model = get_model(model_str) if model_str else self.model_medium
        self.state = fn_summarize_pages(state, model)
        return self.state

    def topic_analysis(self, state: AgentState, model_str: str = "") -> AgentState:
        """extract and assign topics for each headline"""
        model = get_model(model_str) if model_str else self.model_low
        self.state = fn_topic_analysis(state, model)
        return self.state

    def topic_clusters(self, state: AgentState, model_str: str = "") -> AgentState:
        """identify clusters of similar stores"""
        model = get_model(model_str) if model_str else self.model_low
        self.state = fn_topic_clusters(state, model)
        return self.state

    def quality_filter(self, state: AgentState, model_str: str = "") -> AgentState:
        """filter low-quality stories"""
        model = get_model(model_str) if model_str else self.model_medium
        self.state = fn_quality_filter(state, model)
        return self.state

    def on_topic_filter(self, state: AgentState, model_str: str = "") -> AgentState:
        """filter off_topic stories"""
        model = get_model(model_str) if model_str else self.model_medium
        self.state = fn_on_topic_filter(state, model)
        return self.state

    def importance_filter(self, state: AgentState, model_str: str = "") -> AgentState:
        """filter important stories"""
        model = get_model(model_str) if model_str else self.model_medium
        self.state = fn_importance_filter(state, model)
        return self.state

    def rate_articles(self, state: AgentState) -> AgentState:
        """set article ratings"""
        self.state = fn_rate_articles(state)
        return self.state

    def propose_topics(self, state: AgentState, model_str: str = "") -> AgentState:
        """use LLM to identify most popular and important topics"""
        model = get_model(model_str) if model_str else self.model_high
        self.state = fn_propose_topics(state, model)
        return self.state

    def compose_summary(self, state: AgentState, model_str: str = "") -> AgentState:
        """compose the first draft of the summary using bullets and topics"""
        model = get_model(model_str) if model_str else self.model_high
        self.state = fn_compose_summary(state, model)
        return self.state

    def rewrite_summary(self, state: AgentState, model_str: str = "") -> AgentState:
        """edit the summary, combine and sharpen items, add and improve titles"""
        model = get_model(model_str) if model_str else self.model_high
        self.state = fn_rewrite_summary(state, model)
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
                    display(Markdown(event.get('summary').replace("$", "\\\\$")))
                elif event.get('bullets'):
                    print('bullets created')
                    display(Markdown("\n\n".join(
                        event.get('bullets')).replace("$", "\\\\$")))
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


def initialize_agent(model_low, model_medium, model_high, do_download=True, before_date=None, max_edits=MAX_EDITS, n_browsers=N_BROWSERS):
    """set initial state"""
    state = AgentState({
        'AIdf': [{}],
        'before_date': before_date,
        'do_download': do_download,
        'model_low': model_low,
        'model_medium': model_medium,
        'model_high': model_high,
        'sources': {},
        'sources_reverse': {},
        'bullets': '',
        'summary': '',
        'cluster_topics': [],
        'topics_str': '',
        'n_edits': 0,
        'max_edits': max_edits,
        'edit_complete': False,
        'n_browsers': n_browsers,
    })
    thread_id = uuid.uuid4().hex
    log(f"Initializing with before_date={state.get('before_date')}, do_download={do_download}, thread_id={thread_id}"
        )
    return state, Agent(state), thread_id


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nofetch', action='store_true', default=False,
                        help='Disable web fetch, use existing HTML files in htmldata directory')
    parser.add_argument('-d', '--before-date', type=str, default='',
                        help='Process articles before this date (YYYY-MM-DD HH:MM:SS format)')
    parser.add_argument('-b', '--browsers', type=int, default=4,
                        help='Number of browser instances to run in parallel (default: 4)')
    parser.add_argument('-e', '--max-edits', type=int, default=2,
                        help='Maximum number of summary rewrites')

    args = parser.parse_args()

    do_download = not args.nofetch
    before_date = args.before_date
    N_BROWSERS = args.browsers
    MAX_EDITS = args.max_edits
    log(f"Starting AInewsbot with do_download={do_download}, before_date='{before_date}', N_BROWSERS={N_BROWSERS}, MAX_EDITS={MAX_EDITS}")

    ml, mm, mh = 'gpt-4.1-mini', 'gpt-4.1', 'o4-mini'

    lg_state, lg_agent, thread_id = initialize_agent(ml, mm, mh,
                                                     do_download,
                                                     before_date,
                                                     max_edits=MAX_EDITS,
                                                     n_browsers=N_BROWSERS)

    log(f"thread_id: {thread_id}")
    # save in case we want to get the last state from Sqlite and inpsect or resume in Jupyter
    with open('thread_id.txt', 'w', encoding="utf-8") as file:
        file.write(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    lg_state = lg_agent.run(lg_state, config)

import argparse
import uuid

import nest_asyncio

import pandas as pd

from IPython.display import display, Markdown  # , Audio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

import langchain

from .state import (AgentState,
                    fn_initialize,
                    fn_download_sources,
                    fn_extract_urls,
                    fn_verify_download,
                    # fn_extract_newscatcher,
                    fn_extract_newsapi,
                    fn_filter_urls,
                    fn_topic_analysis,
                    fn_topic_clusters,
                    fn_download_pages,
                    fn_summarize_pages,
                    fn_rate_articles,
                    fn_propose_topics,
                    fn_compose_summary,
                    fn_rewrite_summary,
                    fn_is_revision_complete,
                    fn_send_mail,)

from .utilities import (log, get_model)

# from langchain_anthropic import ChatAnthropic

langchain.verbose = True

# from langchain_core.prompts import ChatPromptTemplate
# JsonOutputParser, StrOutputParser
# from langchain_core.output_parsers import SimpleJsonOutputParser

# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.graph.message import add_messages

nest_asyncio.apply()  # needed for asyncio.run to work under langgraph

# defaults if called via import and not __main__
N_BROWSERS = 8
MAX_EDITS = 2


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


class Agent:
    """Langraph Agent class"""

    def __init__(self, state: AgentState):
        """set up state graph and memory"""
        self.state = state

        self.model_low = get_model(state["model_low"])
        self.model_medium = get_model(state["model_medium"])
        self.model_high = get_model(state["model_high"])

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
        graph_builder.add_edge("topic_clusters", "rate_articles")
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
        self.state = fn_download_sources(state)
        return self.state

    def extract_web_urls(self, state: AgentState) -> AgentState:
        """parse all urls from downloaded pages"""
        self.state = fn_extract_urls(state)
        return self.state

    def verify_download(self, state: AgentState) -> AgentState:
        """verify we found news stories from all sources"""
        self.state = fn_verify_download(state)
        return self.state

    # def extract_newscatcher_urls(self, state: AgentState) -> AgentState:
    #     """extract newscatcher urls"""
    #     try:
    #         self.state = fn_extract_newscatcher(state)
    #     except KeyError:
    #         log("Newscatcher download failed")
    #     return self.state

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
        self.state = fn_download_pages(state)
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
        """identify clusters of similar stories"""
        model = get_model(model_str) if model_str else self.model_low
        self.state = fn_topic_clusters(state, model)
        return self.state

    def rate_articles(self, state: AgentState, model_str: str = "") -> AgentState:
        """set article ratings"""
        model = get_model(model_str) if model_str else self.model_medium
        self.state = fn_rate_articles(state, model)
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


def initialize_agent(model_low, model_medium, model_high, do_download=True, before_date=None, max_edits=MAX_EDITS, n_browsers=N_BROWSERS):  # pylint: disable=redefined-outer-name
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
                        help='Force processing of articles before this date even if already processed (YYYY-MM-DD HH:MM:SS format)')
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

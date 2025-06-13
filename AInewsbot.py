"""
AInewsbot.py
This is the top-level file for the AInewsbot application. It sets up the state graph
and runs it using the LangGraph framework. The individual state functions that define
specific behaviors and transformations are implemented in the `ainb_state.py` module.

The script initializes the agent, configures the state graph, and executes the
workflow for processing news articles, including downloading, filtering, summarizing,
and sending email summaries. It supports command-line arguments for customization.

Modules and Features:
- Defines the `AgentState` TypedDict to represent the state of the agent.
- Implements the `Agent` class to manage the state graph and its execution.
- Provides utility functions to initialize the agent and run the workflow.
- Supports integration with multiple AI models (e.g., OpenAI, Google Generative AI).
- Uses LangGraph for state graph management and execution.

Command-line Arguments:
- `--nofetch`: Disable web fetching and use existing HTML files.
- `--before-date`: Process articles before a specific date.
- `--browsers`: Number of browser instances to run in parallel.
- `--max-edits`: Maximum number of summary rewrites.

Dependencies:
- `ainb_state.py`: Contains the individual state functions.
- `ainb_utilities.py`: Provides logging and utility functions.
- `ainb_const.py`: Defines constants like model families and request timeouts.

Usage:
Run this script directly to execute the AInewsbot workflow, or import it as a module
to use the `Agent` class and related functions programmatically.
"""

import argparse
import nest_asyncio
import langchain

from ainewsbot.agent import initialize_agent
from ainewsbot.utilities import log

langchain.verbose = True

nest_asyncio.apply()  # needed for asyncio.run to work under langgraph

# defaults
N_BROWSERS = 12
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

if __name__ == "__main__":
    # Parse command line arguments
    # TODO: add partial fetch, only fetch specified feeds

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nofetch', action='store_true', default=False,
                        help='Disable web fetch, use existing HTML files in htmldata directory')
    parser.add_argument('-d', '--before-date', type=str, default='',
                        help='Force processing of articles before this date even if already processed (YYYY-MM-DD HH:MM:SS format)')
    parser.add_argument('-b', '--browsers', type=int, default=N_BROWSERS,
                        help=f'Number of browser instances to run in parallel (default: {N_BROWSERS})')
    parser.add_argument('-e', '--max-edits', type=int, default=MAX_EDITS,
                        help=f'Maximum number of summary rewrites (default: {MAX_EDITS})')

    args = parser.parse_args()

    do_download = not args.nofetch
    before_date = args.before_date
    N_BROWSERS = args.browsers
    MAX_EDITS = args.max_edits
    log(f"Starting AInewsbot with do_download={do_download}, before_date='{before_date}', N_BROWSERS={N_BROWSERS}, MAX_EDITS={MAX_EDITS}")

    ml, mm, mh = 'gpt-4.1-mini', 'gpt-4.1-mini', 'o4-mini'

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

import dotenv
import os
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOWNLOAD_DIR = "htmldata"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Path to geckodriver
GECKODRIVER_PATH = '/Users/drucev/webdrivers/geckodriver'
# Path to browser app
# FIREFOX_APP_PATH = '/Applications/Firefox.app'
# Path to profile
FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/k8k0lcjj.default-release'

sleeptime = 10

SQLITE_DB = 'articles.db'

MODEL = "gpt-4o"
LOWCOST_MODEL = "gpt-3.5-turbo-0125"

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
MAX_OUTPUT_TOKENS = 4096    # max in current model
MAX_RETRIES = 3
TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
MINTITLELEN = 28

MAXPAGELEN = 50

PROMPT = """
You will act as a research assistant to categorize news articles based on their relevance
to the topic of artificial intelligence (AI). You will process and classify news headlines
formatted as JSON objects.

Input Specification:
You will receive a list of news stories formatted as JSON objects.
Each object will include an 'id' and a 'title'. For instance:
[{'id': 97, 'title': 'AI to predict dementia, detect cancer'},
 {'id': 103,'title': 'Figure robot learns to make coffee by watching humans for 10 hours'},
 {'id': 103,'title': 'Baby trapped in refrigerator eats own foot'},
 {'id': 210,'title': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'},
 {'id': 298,'title': 'The 5 most interesting PC monitors from CES 2024'},
 ]

Classification Criteria:
Classify each story based on its title to determine whether it primarily pertains to AI.
Broadly define AI-related content to include topics such as machine learning, robotics,
computer vision, reinforcement learning, large language models, and related topics. Also
include specific references to AI-related entities and individuals and products such as
OpenAI, ChatGPT, Elon Musk, Sam Altman, Anthropic Claude, Google Gemini, Copilot,
Perplexity.ai, Midjourney, etc.

Output Specification:
You will return a JSON object with the field 'stories' containing the list of classification results.
For each story, your output will be a JSON object containing the original 'id' and a new field 'isAI',
a boolean indicating if the story is about AI. The output schema must be strictly adhered to, without
any additional fields. Example output:
{'stories':
[{'id': 97, 'isAI': true},
 {'id': 103, 'isAI': true},
 {'id': 103, 'isAI': false},
 {'id': 210, 'isAI': true},
 {'id': 298, 'isAI': false}]
}

Ensure that each output object accurately reflects the corresponding input object in terms of the 'id' field
and that the 'isAI' field accurately represents the AI relevance of the story as determined by the title.

The list of news stories to classify and enrich is:

"""

SUMMARIZE_SYSTEM_PROMPT = """You are a summarization assistant.
You will summarize the main content of provided text from HTML files in 3 bullet points or less.
You will output Markdown format.
You will ignore any content that appears to be navigation menus, footers, sidebars, or other boilerplate content.
You will provide the bullet points only, without any introduction such as 'here are' or any conclusion, or comment.
"""

SUMMARIZE_USER_PROMPT = """Summarize the main points of the following text concisely in at most 3 bullet points:
"""

bb_agent_system_prompt = """
Role: You are an AI stock market assistant tasked with providing investors
with up-to-date, detailed information on individual stocks.

Objective: Assist data-driven stock market investors by giving accurate,
complete, but concise information relevant to their questions about individual
stocks.

Capabilities: You are given a number of tools as functions. Use as many tools
as needed to ensure all information provided is timely, accurate, concise,
relevant, and responsive to the user's query.

Instructions:
1. Input validation. Determine if the input is asking about a specific company
or stock ticker. If not, respond in a friendly, positive, professional tone
that you don't have information to answer and suggest alternative services
or approaches.

2. Symbol extraction. If the query is valid, extract the company name or ticker
symbol from the question. If a company name is given, look up the ticker symbol
using a tool. If the ticker symbol is not found based on the company, try to
correct the spelling and try again, like changing "microsfot" to "microsoft",
or broadening the search, like changng "southwest airlines" to a shorter variation
like "southwest" and increasing "limit" to 10 or more. If the company or ticker is
still unclear based on the question or conversation so far, and the results of the
symbol lookup, then ask the user to clarify which company or ticker.

3. Information retrieval. Determine what data the user is seeking on the symbol
identified. Use the appropriate tools to fetch the requested information. Only use
data obtained from the tools. You may use multiple tools in a sequence. For instance,
first determine the company's symbol, then retrieving company data using the symbol.

4. Compose Response. Provide the answer to the user in a clear and concise format,
in a friendly professional tone, emphasizing the data retrieved, without comment
or analysis unless specifically requested by the user.

Example Interaction:
User asks: "What is the PE ratio for Eli Lilly?"
Chatbot recognizes 'Eli Lilly' as a company name.
Chatbot uses symbol lookup to find the ticker for Eli Lilly.
Chatbot retrieves the PE ratio using the proper function.
Chatbot responds: "The PE ratio for Eli Lilly (symbol: LLY) as of May 12, 2024 is 30."

Check carefully and only call the tools which are specifically named below.
Only use data obtained from these tools.

"""


FINAL_SUMMARY_PROMPT = f"""You are a summarization assistant. I will provide a list of today's news stories about AI
and summary bullet points in markdown format. You are tasked with identifying and summarizing the key themes,
common facts, and recurring elements. Your goal is to create a concise summary containing about 20 of the most frequently
mentioned topics and developments.

Example Bullet Points:

[2. Sentient closes $85M seed round for open-source AI](https://cointelegraph.com/news/sentient-85-million-round-open-source-ai)

- Sentient secured $85 million in a seed funding round led by Peter Thiel's Founders Fund, Pantera Capital, and Framework Ventures for their open-source AI platform.
- The startup aims to incentivize AI developers with its blockchain protocol and incentive mechanism, allowing for the evolution of open artificial general intelligence.
- The tech industry is witnessing a rise in decentralized AI startups combining blockchain

Examples of important stories:

Major investments and funding rounds
Key technological advancements or breakthroughs
Frequently mentioned companies, organizations, or figures
Notable statements by AI leaders
Any other recurring themes or notable patterns

Instructions:
Read the summary bullet points closely.
Use only information provided in them and provide the most common facts without commentary or elaboration.
Write in the professional but engaging, narrative style of a tech reporter for a national publication.
Be balanced, professional, informative, providing accurate, clear, concise answers in a respectful neutral tone.
Focus on the most common elements across the bullet points and group similar items together.
Ensure that you provide at least one link from the provided text for each item in the summary.
You must include at least 10 and no more than 25 items in the summary.

Bullet Points to Summarize:

"""

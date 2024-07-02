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


DOWNLOAD_DIR = "htmldata"
# Path to geckodriver
GECKODRIVER_PATH = '/Users/drucev/webdrivers/geckodriver'
# Path to browser app
# FIREFOX_APP_PATH = '/Applications/Firefox.app'
# Path to profile
FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/k8k0lcjj.default-release'

sleeptime = 10

SQLITE_DB = 'articles.db'

MODEL = "gpt-4-turbo"

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
MAX_OUTPUT_TOKENS = 4096    # max in current model
MAX_RETRIES = 3
TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
MINTITLELEN = 28

MAXPAGELEN = 50

PROMPT = """
Please serve as a research assistant for the purpose of categorizing news articles based on their relevance to artificial intelligence (AI).
Your main responsibility will involve processing and classifying news articles formatted as JSON objects.

Classification Criteria: Based on the title of each story, you are to classify whether the story primarily pertains to AI or not.
Consider AI-related content to broadly include topics such as machine learning, supervised learning, unsupervised learning,
reinforcement learning, robotics, computer vision, large language models, related topics, and specific references
to AI entities like OpenAI, ChatGPT, Anthropic Claude, Google Gemini, Copilot, Perplexity.ai, Midjourney, etc.

Input Specification: You will receive a list of news stories formatted as JSON objects separated by the delimiter "|".
Each object includes an 'id' and a 'title'. For instance:
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

Output Specification: For each story, your output should be a JSON object containing the original 'id' and a new field 'isAI',
which is a boolean indicating if the story is about AI. This output should be enclosed in the delimiter "~".
The output schema must be strictly adhered to, without any additional fields. Example output:
~
{'stories':
[{'id': 97, 'isAI': true},
 {'id': 103, 'isAI': true},
 {'id': 103, 'isAI': false},
 {'id': 210, 'isAI': true},
 {'id': 298, 'isAI': false}]
}
~

Strictly ensure that each output object accurately reflects the corresponding input object in terms of the 'id' field
and that the 'isAI' field accurately represents the AI relevance of the story as determined by the title.

The list of news stories to classify and enrich is:


"""

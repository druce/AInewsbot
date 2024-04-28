
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
You will act as a research assistant classifying news stories as related to artificial intelligence (AI) or unrelated to AI.

Your task is to read JSON format objects from an input list of news stories using the schema below delimited by |,
and output JSON format objects for each using the schema below delimited by ~.

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

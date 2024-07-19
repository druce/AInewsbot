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
LOWCOST_MODEL = "gpt-4o-mini"

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
MAX_OUTPUT_TOKENS = 4096    # max in current model
MAX_RETRIES = 3
TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
MINTITLELEN = 28

MAXPAGELEN = 50

FILTER_PROMPT = """
You will act as a research assistant to categorize news articles based on their relevance
to the topic of artificial intelligence (AI). You will closely read the title of each story
to determine if it is primarily about AI based on the semantic meaning of the title and
the keywords and entities mentioned. The input headlines and outptu classifications will
be formatted as JSON objects.

Input Specification:
You will receive a list of news headlines formatted as JSON objects.
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
a boolean indicating if the story is about AI. You must strictly adhere to this output schema, without
modification. Example output:
{'stories':
[{'id': 97, 'isAI': true},
 {'id': 103, 'isAI': true},
 {'id': 103, 'isAI': false},
 {'id': 210, 'isAI': true},
 {'id': 298, 'isAI': false}]
}

Instructions:
Ensure that each output object accurately reflects the 'id' field of the corresponding input object
and that the 'isAI' field accurately represents the title's relevance to AI.

The list of news stories to classify is:

"""

TOPIC_PROMPT = """
You will act as a research assistant to extract topics from news headlines. You will extract topics, entities,
and keywords from news headlines formatted as JSON objects.

Input Specification:
You will receive a list of news headlines formatted as JSON objects.
Each object will include an 'id' and a 'title'. For instance:
[{'id': 97, 'title': 'AI to predict dementia, detect cancer'},
 {'id': 103,'title': 'Figure robot learns to make coffee by watching humans for 10 hours'},
 {'id': 105,'title': "Former Microsoft CEO Steve Ballmer is now just as rich as his former boss Bill Gates. Here's how he spends his billions."},
 {'id': 210,'title': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'},
 {'id': 298,'title': 'The 5 most interesting PC monitors from CES 2024'},
]

Output Specification:
You will return a JSON object with the field 'topics' containing a flat list of classification results.
For each headline input, your output will be a JSON object containing the original 'id' and a new field 'topics',
with a list of strings representing topics. The output schema must be strictly adhered to, without
any additional fields. Example output:

{'topics':
 [{"id": 97, "topics": ["AI", "dementia", "cancer", "healthcare", "prediction", "diagnostics"]},
  {"id": 103, "topics": ["AI", "robotics", "machine learning", "automation", "coffee making", "Figure"]},
  {"id": 105, "topics": ["Microsoft", "Steve Ballmer", "Bill Gates", "wealth", "billionaires"]},
  {"id": 210, "topics": ["AI", ChatGPT", "product updates", "summarization"]},
  {"id": 298, "topics": ["PCs", "monitors", "CES 2024", "consumer electronics"]},
 ]
}

Instructions:
Ensure that each output object accurately reflects this schema exactly without modification, and that
it matches the corresponding input object in terms of the 'id' field and relevant topics.

The list of headlines to extract topics from:

"""

SUMMARIZE_SYSTEM_PROMPT = """You are a summarization assistant.
You will summarize the main content of provided text from HTML files in 3 bullet points or less.
You will focus on the top 3 points of the text and keep the response concise."
You will ignore any content that appears to be navigation menus, footers, sidebars, or other boilerplate content.
You will output Markdown format.
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


TOP_CATEGORIES_PROMPT = """You will act as a research assistant identifying the top stories and topics
of today's news. I will provide a list of today's news stories about AI and summary bullet points in markdown
format. You are tasked with identifying the top 10-20 stories and topics of today's news. For each top story
or topic, you will create a short title and respond with a list of titles formatted as a JSON object.


Example Input Bullet Points:

[2. Sentient closes $85M seed round for open-source AI](https://cointelegraph.com/news/sentient-85-million-round-open-source-ai)

- Sentient secured $85 million in a seed funding round led by Peter Thiel's Founders Fund, Pantera Capital, and Framework Ventures for their open-source AI platform.
- The startup aims to incentivize AI developers with its blockchain protocol and incentive mechanism, allowing for the evolution of open artificial general intelligence.
- The tech industry is witnessing a rise in decentralized AI startups combining blockchain

Categories of top stories and topics:
Major investments and funding rounds
Major technological advancements or breakthroughs
Frequently mentioned entities such as companies, organizations, or figures
Any other frequently discussed events, statements, entities or notable patterns

Instructions:
Read the summary bullet points closely and use only information provided in them.
Focus on the most common elements across all bullet points.
Titles of top stories and topics must be as short and simple as possible.
You must include at least 10 and no more than 20 topics in the summary.
Respond with only the names of the top stories and topics in JSON format, without any comment, summary or discussion.

Example Output Format:
{'stories' : [
  "Sentient funding",
  "ChatGPT cybersecurity incident",
  "ElevenLabs product release",
  "Microsoft text-to-speech model"
  "Nvidia reguatory issues",
  "AI healthcare successes"
  ]
}

Bullet Points to Analyze:

"""


FINAL_SUMMARY_PROMPT = f"""You are an advanced content analysis assistant, a sophisticated AI system
designed to perform in-depth analysis of news content. Your function is to extract meaningful insights,
categorize information, and identify trends from large volumes of news.

I will provide a list of today's news articles about AI and summary bullet points in markdown format.
Bullet points will contain a title and URL, a list of topics discussed, and a bullet-point summary of
the article. You are tasked with identifying and summarizing the most important news, recurring themes,
common facts and items. Your job is to create a concise summary containing about 20 of the most
frequently mentioned topics and developments.


Example Input Bullet Points:

[2. Sentient closes $85M seed round for open-source AI](https://cointelegraph.com/news/sentient-85-million-round-open-source-ai)

AI startup funding, New AI products

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
Be balanced, professional, informative, providing accurate, clear, concise summaries in a respectful neutral tone.
Focus on the most common elements across the bullet points and group similar items together.
Headers must be as short and simple as possible: use "Health Care" and not "AI developments in Health Care" or "AI in Health Care"
Ensure that you provide at least one link from the provided text for each item in the summary.
You must include at least 10 and no more than 25 items in the summary.

Example Output Format:

# Today's AI News

### Security and Privacy:
- ChatGPT Mac app had a security flaw exposing user conversations in plain text. ([Macworld](https://www.macworld.com/article/2386267/chatgpt-mac-sandboxing-security-flaw-apple-intelligence.html))
- Brazil suspended Meta from using Instagram and Facebook posts for AI training over privacy concerns. ([BBC](https://www.bbc.com/news/articles/c7291l3nvwv))

### Health Care:
- AI can predict Alzheimer's disease with 70% accuracy up to seven years in advance. ([Decrypt](https://decrypt.co/238449/ai-alzheimers-detection-70-percent-accurate-study))
- New AI system detects 13 cancers with 98% accuracy, revolutionizing cancer diagnosis. ([India Express](https://news.google.com/articles/CBMiiAFodHRwczovL2luZGlhbmV4cHJl))

Bullet Points to Summarize:

"""

CANONICAL_TOPICS = [
    "Policy and regulation",
    "Economics",
    "Robots",
    "Autonomous vehicles",
    "LLMs",
    "Healthcare",
    "Fintech",
    "Education",
    "Entertainment",
    "Funding",
    "Venture Capital",
    "Mergers and acquisitions",
    "Deals",
    "IPOs",
    "Ethics",
    "Laws",
    "Cybersecurity",
    "AI doom",
    'Stocks',
    'Bubble',
    'Cryptocurrency',
    'Climate',
    'Energy',
    'Nuclear',
    'Scams',
    'Privacy',
    'Intellectual Property',
    'Code assistants',
    'Customer service',
    'Machine learning',
    'Reinforcement Learning',
    'Open Source',
    'Language Models',
    'Military',
    'Semiconductor chips',
    'Sustainability',
    'Agriculture',
    'Gen AI',
    'Testing',
    'RAG',
    'Data',
    'Disinformation',
    'Deepfakes',
    'Virtual Reality',
    'Cognitive Science',
    'Smart Home',
    'Authors, Writing',
    'Books, Publishing',
    'TV, Film, Movies',
    'Streaming',
    'Hollywood',
    'Music',
    'Art, Design',
    'Fashion',
    'Food, Drink',
    'Travel, Adventure',
    'Health, Fitness',
    'Sports',
    'Gaming',
    'Science',
    'Politics',
    'Finance',
    'History',
    'Society, Culture',
    'Lifestyle, Travel',
    'Jobs, Careers, Labor Market',
    'Products',
    'Opinion',
    'Review',
    'Artificial General Intelligence',
    'Singularity',
    'Consciousness',
    'Facial Recognition',
    'Manufacturing',
    'Supply chain optimization',
    'Transportation',
    'Smart grid',
    'Recommendation systems',

    'Nvidia',
    'Google',
    'OpenAI',
    'Meta',
    'Apple',
    'Microsoft',
    'Perplexity',
    'Salesforce',
    'Uber',
    'AMD',
    'Netflix',
    'Disney',
    'Amazon',
    'Cloudflare',
    'Anthropic',
    'Cohere',
    'Baidu',
    'Big Tech',
    'Samsung',
    'Tesla',
    'Reddit',

    'ChatGPT',
    'WhatsApp',
    'Gemini',
    'Claude',
    'Copilot',

    'Elon Musk',
    'Bill Gates',
    'Sam Altman',
    'Mustafa Suleyman',
    'Sundar Pichai',
    'Yann LeCun',
    'Geoffrey Hinton',
    'Mark Zuckerberg',

    'China',
    'European Union',
    'UK',
    'Russia',
    'Japan',
    'India',
    'Korea',
    'Taiwan',
]

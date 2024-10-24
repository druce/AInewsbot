import dotenv
import os
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOWNLOAD_DIR = "htmldata"
PAGES_DIR = 'htmlpages'

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

BASEMODEL = 'gpt-4o'  # tiktoken doesn't always map latest model
MODEL = 'chatgpt-4o-latest'
LOWCOST_MODEL = 'gpt-4o-mini'
HIGHCOST_MODEL = 'o1-preview'

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
MAX_OUTPUT_TOKENS = 4096    # max in current model
MAX_RETRIES = 3
TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
MINTITLELEN = 28

MAXPAGELEN = 50

HOSTNAME_SKIPLIST = ['finbold.com']
SITE_NAME_SKIPLIST = ['finbold']

TOPSOURCES = {
    'Bloomberg Tech',
    'www.bloomberg.com',
    'news.bloomberglaw.com',
    'FT Tech',
    'www.ft.com',
    'www.theverge.com',
    'The Verge',
    'NYT Tech',
    'www.nytimes.com',
    'Techmeme',
    'www.techmeme.com',
    'WSJ Tech',
    'www.wsj.com',
    'www.theinformation.com',
    'www.nature.com',
    'www.theatlantic.com',
    'openai.com',
    'www.science.org',
    'www.scientificamerican.com',
}

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

SUMMARIZE_SYSTEM_PROMPT = """You are a news summarization assistant.
You will summarize the main content of provided news text from HTML files in 3 bullet points or less.
Focus on the top 3 points of the text. Keep the bullet points concise.
Do not include anything that is not news content.
Specifically, do not include: instructions to the user about browser compatibility,
logging in, proving they are not a robot, or enabling javascript, or cookie settings,
or contacting support, or any advertisements, promotions, discounts, subscription
offers, boilerplate disclaimers, or any other non-news content.
If the text contains no substantive news content to summarize, return a single bullet point
stating that.
Output Markdown format.
Provide the bullet points only, without any introduction such as 'here are' or any conclusion, or comment.
"""

SUMMARIZE_USER_PROMPT = """Summarize the main points of the following text concisely in 3 bullet points or less.
Ignore any content that is navigation, user instructions, disclaimers, sidebars, ads, or boilerplate.
Text:
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

FINAL_SUMMARY_PROMPT = """You are ASA, an advanced summarization assistant designed to
write a compelling summary of news input. You are able to categorize information,
and identify important themes from large volumes of news. Create a cohesive, concise newsletter
from a list of article summaries, with snappy titles, bullet points and links to original articles.


ASA Objective:

I will provide today's news items about AI and summary bullet points in markdown format,
structured according to an input format template.

News items are delimited by ~~~

You are tasked with using the news items to create a concise summary of today's most important topics and developments.

You will write an engaging summary of today's news encompassing the most important and frequently
mentioned topics and themes, in an output format provided below.

ASA Input Item Format Template:

[Story-Title-s1 - source-name-s1](url-s1)

Topics: s1-topic1, s1-topic2, s1-topic3

- s1-bullet-point-1
- s1-bullet-point-2
- s1-bullet-point-3

Example ASA Input Item Format:

[Apple Intelligence is now live in public beta. Heres what it offers and how to enable it. - TechCrunch](https://techcrunch.com/2024/09/19/apple-intelligence-is-now-live-in-public-beta-heres-what-it-offers-and-how-to-enable-it)

Topics: Apple, Big Tech, Features, Gen AI, Intelligence, Machine Learning, Products, Public Beta, Virtual Assistants

- Apple Intelligence is now live in public beta for users in the U.S. enrolled in the public beta program, featuring generative AI capabilities like advanced writing tools and a revamped Siri.
- The platform is currently only available in U.S. English and is not accessible in the EU or China due to regulatory issues; it supports iPhone 15 Pro, Pro Max, and the new iPhone 16 line.
- Key features include photo editing tools like "Clean Up," a Smart Reply function in Mail, and improvements to Siri’s understanding and on-device task knowledge.

ASA Output Format Template:

# Engaging-topic-title-1

- item-title-1a - [source-name-1a](item-url-1a)
- item-title-1b - [source-name-1b](item-url-1b)
- item-title-1c - [source-name-1c](item-url-1c)

# Engaging-topic-title-2

- item-title-2a - [source-name-2a](item-url-2a)
- item-title-2b - [source-name-2b](item-url-2b)

Example ASA Output Format:

# A military AI revolution

- Eric Schmidt on AI warfare - [FT](https://www.ft.com/content/fe136479-9504-4588-869f-900f2b3452c4)
- Killer robots are real in Ukraine war. - [Yahoo News](https://uk.news.yahoo.com/ai-killer-robots-warning-ukraine-war-133415411.html)

ASA Instructions:
Read each input summary closely to extract their main points and themes.
USE ONLY INFORMATION PROVIDED IN THE INPUT SUMMARIES.
Group news items into related topics.
Develop a snappy, engaging punchy, clever, alliterative, possibly punny title for each topic.
Each topic chould contain the most significant facts from the news items without commentary or elaboration.
Each news item bullet should contain one sentence with one link. The link must be identical to the one in the corresponding news item input.
Each news item bullet should not repeat points or information from previous bullet points.
You will write each news item in the professional but engaging, narrative style of a tech reporter
for a national publication, providing balanced, professional, informative, providing accurate,
clear, concise summaries in a neutral tone.
Do not include ```markdown , output raw markdown.
Check carefully that you only use information provided in the input below, that you include
a link in each output item, and that any bullet point does not repeat information or links previously provided.

Topic suggestions:
{cat_str}

Input:
{bullet_str}

"""

REWRITE_PROMPT = """You will act as a professional editor with a strong background in technology journalism.
You have a deep understanding of current and emerging AI trends, and the ability to
produce, edit, and curate high-quality content that engages and informs readers. You are
especially skilled at reviewing and enhancing tech writing, helping improve clarity, conciseness,
and coherence, and ensuring its accuracy and relevance.

Objective: The markdown newsletter provided below contains several sections consisting of bullet points.
Carefully review each section of the newsletter. Edit the newsletter for issues according
to the detailed instructions below, and respond with the updated newsletter or 'OK' if no changes
are needed.

Instructions:
Do not include ```markdown. Output raw markdown.
For each bullet point, make it as concise as possible, sticking to facts without editorial comment.
For each section, remove or combine bullet points which are highly duplicative or redundant or devoid of news content.
You may remove bullet points but you may not update URLs.
For each section, review and edit the section title.
The section title should be snappy, punchy, clever, possibly alliterative or punny.
The section title must be short, engaging, and as consistent with the bullets in the section as possible.
Remove sections which are devoid of news content. Ensure there are no comments on the content or composition of the newsletter.
At the top of the newsletter add an overall title synthesizing several top news items.
Respond with the updated newsletter only in markdown format, or the word 'OK' if no changes are needed.

Newsletter to edit:
{summary}

"""

CANONICAL_TOPICS = [
    "Policy and regulation",
    "Economics",
    "Governance",
    "Safety and Alignment",
    "Bias and Fairness",
    "Privacy & Surveillance",
    "Inequality",
    "Job Automation",
    'Disinformation',
    'Deepfakes',
    'Sustainability',

    "Virtual Assistants",
    "Chatbots",
    "Robots",
    "Autonomous vehicles",
    "Drones",
    'Virtual & Augmented Reality',

    # 'Machine learning',
    # 'Deep Learning',
    # "Neural Networks",
    # "Generative Adversarial Networks",

    'Reinforcement Learning',
    'Language Models',
    'Transformers',
    'Gen AI',
    'Retrieval Augmented Generation',
    "Computer Vision",
    'Facial Recognition',
    'Speech Recognition & Synthesis',

    'Open Source',

    'Internet of Things',
    'Quantum Computing',
    'Brain-Computer Interfaces',

    "Hardware",
    "Infrastructure",
    'Semiconductor Chips',
    'Neuromorphic Computing',

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
    "Legal issues",
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
    'Military',
    'Agriculture',
    'Testing',
    'Authors & Writing',
    'Books & Publishing',
    'TV & Film & Movies',
    'Streaming',
    'Hollywood',
    'Music',
    'Art & Design',
    'Fashion',
    'Food & Drink',
    'Travel',
    'Health & Fitness',
    'Sports',
    'Gaming',
    'Science',
    'Politics',
    'Finance',
    'History',
    'Society & Culture',
    'Lifestyle & Travel',
    'Jobs & Careers'
    'Labor Market',
    'Products',
    'Opinion',
    'Review',
    'Cognitive Science',
    'Consciousness',
    'Artificial General Intelligence',
    'Singularity',
    'Manufacturing',
    'Supply chain optimization',
    'Transportation',
    'Smart grid',
    'Recommendation systems',

    # 'Nvidia',
    # 'Google',
    # 'OpenAI',
    # 'Meta',
    # 'Apple',
    # 'Microsoft',
    # 'Perplexity',
    # 'Salesforce',
    # 'Uber',
    # 'AMD',
    # 'Netflix',
    # 'Disney',
    # 'Amazon',
    # 'Cloudflare',
    # 'Anthropic',
    # 'Cohere',
    # 'Baidu',
    # 'Big Tech',
    # 'Samsung',
    # 'Tesla',
    # 'Reddit',
    # "DeepMind",
    # "Intel",
    # "Qualcomm",
    # "Oracle",
    # "SAP",
    # "Alibaba",
    # "Tencent",
    # "Hugging Face",
    # "Stability AI",
    # "Midjourney",
    # 'WhatsApp',

    # 'ChatGPT',
    # 'Gemini',
    # 'Claude',
    # 'Copilot',

    # 'Elon Musk',
    # 'Bill Gates',
    # 'Sam Altman',
    # 'Mustafa Suleyman',
    # 'Sundar Pichai',
    # 'Yann LeCun',
    # 'Geoffrey Hinton',
    # 'Mark Zuckerberg',
    # "Demis Hassabis",
    # "Andrew Ng",
    # "Yoshua Bengio",
    # "Ilya Sutskever",
    # "Dario Amodei",
    # "Richard Socher",
    # "Sergey Brin",
    # "Larry Page",
    # "Satya Nadella",
    # "Jensen Huang",

    'China',
    'European Union',
    'UK',
    'Russia',
    'Japan',
    'India',
    'Korea',
    'Taiwan',
]

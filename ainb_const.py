# Description: Constants, including configs and prompts for AInewsbot project
import os
import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOWNLOAD_DIR = "htmldata"
PAGES_DIR = 'htmlpages'
SCREENSHOT_DIR = 'screenshots'

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
if not os.path.exists(PAGES_DIR):
    os.makedirs(PAGES_DIR)
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# Path to geckodriver
GECKODRIVER_PATH = '/Users/drucev/webdrivers/geckodriver'
# Path to browser app
# FIREFOX_APP_PATH = '/Applications/Firefox.app'
# Path to profile
FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/k8k0lcjj.default-release'
CHROME_PROFILE_PATH = '/Users/drucev/Library/Application Support/Google/Chrome'
CHROME_PROFILE = 'Profile 7'
CHROME_DRIVER_PATH = '/Users/drucev/Library/Application Support/undetected_chromedriver/undetected_chromedriver'
sleeptime = 10
NUM_BROWSERS = 4
BROWSERS = []

SQLITE_DB = 'articles.db'

BASEMODEL = 'gpt-4o'  # tiktoken doesn't always map latest model
MODEL = 'gpt-4o-2024-11-20'
LOWCOST_MODEL = 'gpt-4o-mini'
HIGHCOST_MODEL = 'o3-mini'

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
You will act as a specialized content analyst focused on artificial intelligence news classification.
Your task is to evaluate news headlines and determine their relevance to artificial intelligence
and related technologies. Please analyze the provided JSON dataset of news headlines according
to the following detailed criteria:

Classification Framework:

AI Topics (classify as AI-related if headlines mention):
Core AI technologies (machine learning, neural networks, deep learning)
AI applications (computer vision, natural language processing, robotics)
AI models and platforms (large language models, foundation models)
AI companies and their products (OpenAI, DeepMind, Anthropic)
AI-specific products (ChatGPT, Claude, Gemini, DALL-E)
Key AI industry figures (Sam Altman, Demis Hassabis, etc.)
AI policy, regulation, and ethics
AI research, including papers and announcements of innovations
AI market and business developments
AI integration into existing products/services
AI impact on industries/society
AI infrastructure (chips, computing resources)
AI investments and funding
AI partnerships, joint ventures and project launches
Any other topics with a large AI component

Input Specification:
You will receive a JSON array of news story objects, each containing:
"id": A unique numerical identifier
"title": The news headline text

Input Example:
[{{'id': 97, 'title': 'AI to predict dementia, detect cancer'}},
 {{'id': 103,'title': 'Figure robot learns to make coffee by watching humans for 10 hours'}},
 {{'id': 103,'title': 'Baby trapped in refrigerator eats own foot'}},
 {{'id': 210,'title': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'}},
 {{'id': 298,'title': 'The 5 most interesting PC monitors from CES 2024'}},
 ]

Output Requirements:
Generate a JSON object containing "items", an array of objects containing:
"id": The original article identifier
"isAI": Boolean value (true if AI-related, false if not)
The output must maintain strict JSON formatting and match each input ID with its corresponding classification.

Example output:
{{items:
[{{'id': 97, 'isAI': true}},
 {{'id': 103, 'isAI': true}},
 {{'id': 103, 'isAI': false}},
 {{'id': 210, 'isAI': true}},
 {{'id': 298, 'isAI': false}}]
}}

Please analyze the following dataset according to these criteria:

"""
TOPIC_PROMPT = """
As a specialized research assistant, your task is to perform detailed topic analysis
of news item summaries. You will process news items summaries provided in JSON format
and extract topics of the news item summaries according to the following specifications:

Input Specification:
You will receive an array of JSON objects representing news summaries.
Each headline object contains exactly two fields:
'id': A unique numeric identifier
'summary': The news summmary item

Example input:
[{{'id': 97, 'summary': 'AI to predict dementia, detect cancer'}},
 {{'id': 103,'summary': 'Figure robot learns to make coffee by watching humans for 10 hours'}},
 {{'id': 105,'summary': "Former Microsoft CEO Steve Ballmer is now just as rich as his former boss Bill Gates. Here's how he spends his billions."}},
 {{'id': 210,'summary': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'}},
 {{'id': 298,'summary': 'The 5 most interesting PC monitors from CES 2024'}},
]

Output Specification:
Return a JSON object containing 'items', a list of JSON objects, each containing:
'id': Matching the input item's ID
'topics': An array of relevant topic strings
Topics should capture:
- The main subject matter
- Key entities (companies, people, products)
- Technical domains, industry sectors, event types

Output Example:
{{items:
 [{{"id": 97, "extracted_topics": ["AI", "dementia", "cancer", "healthcare", "prediction", "diagnostics"]}},
  {{"id": 103, "extracted_topics": ["AI", "robotics", "machine learning", "automation", "coffee making", "Figure"]}},
  {{"id": 105, "extracted_topics": ["Microsoft", "Steve Ballmer", "Bill Gates", "wealth", "billionaires"]}},
  {{"id": 210, "extracted_topics": ["AI", ChatGPT", "product updates", "summarization"]}},
  {{"id": 298, "extracted_topics": ["PCs", "monitors", "CES 2024", "consumer electronics"]}},
 ]
}}

Detailed Guidelines:
Extract 3-6 relevant topics per news item.
Ensure topic arrays are non-empty.
Use topics which are as specific as possible.
Avoid duplicate or redundant topics.
Return valid JSON format.
Maintain exact schema compliance.

Please analyze the following news items and provide topic classifications according to these specifications:
"""


CANONICAL_TOPIC_PROMPT = """
You will act as a specialized content analyst focused on news classification.
Your task is to evaluate news summaries and determine if they are about {topic}.
Please analyze the JSON dataset of news summaries provided below, according
to the following detailed criteria:

Input Specification:
You will receive a JSON array of news story objects, each containing:
"id": A unique numerical identifier
"summary": The news summary in markdown format

Output Requirements:
Generate a JSON object contaning 'items', a JSON array of objects containing:
"id": The original article identifier
"relevant": Boolean value (true if about {topic}, false if not)
The output must maintain strict JSON formatting and match each input ID with its corresponding classification.

Example output:
{{items:
[{{'id': 97, 'relevant': true}},
 {{'id': 103, 'relevant': true}},
 {{'id': 103, 'relevant': false}},
 {{'id': 210, 'relevant': true}},
 {{'id': 298, 'relevant': false}}]
}}

Consider a news summary to be about {topic} if it contains any of the following:
Direct mentions and references to {topic}
Direct mentions of people, products, research, projects, companies or entities closely associated with {topic}

Please analyze the following dataset according to these criteria:
"""


SUMMARIZE_SYSTEM_PROMPT = """
You will act as a news article summarization assistant.
Please analyze the text provided, and create a focused summary according to the following specific guidelines.

Key Requirements:

Extract only the core news content, specifically:

Primary facts and developments
Key quotes from relevant sources
Critical background information directly related to the story

Explicitly exclude:
Website navigation elements
User interface instructions or prompts
Login forms
Javascript instructions
Cookie/privacy notifications
Subscription offers or paywalls
Advertisement content
Social media widgets
Footer information
Legal disclaimers
Site descriptions or "About Us" content

Output Format:

Present the summary in 1-3 concise bullet points using Markdown format (•)
Each bullet point should capture a distinct main idea
Keep language clear and direct
Include only factual information from the article
If no substantive news content is found, provide a single bullet point stating 'no content'

Special Instructions:

Focus on the most newsworthy elements
Maintain journalistic objectivity
Preserve the original meaning without editorializing
Ensure accuracy in fact representation
Prioritize recent developments over historical context

"""

SUMMARIZE_USER_PROMPT = """Summarize the main points of the following text concisely in 3 bullet points or less:
Text:
{article}
"""

TOPIC_WRITER_PROMPT = """
You are a specialized topic analysis assistant focused on creating titles for groups
of related news headlines. The titles should be concise, accurate, unifying, and surface
the principal common thread between them. I will provide you with:

A set of news headlines, each followed by their extracted key topics in parentheses
Each headline's extracted topics will be labeled as "Topics:" followed by comma-separated topics

Your objective is to:

Analyze all headlines and their associated topics
Identify the most prominent common theme or subject that connects these headlines
Create a single, unified topic title that:
Captures the essential shared meaning across all headlines
Uses no more than 6 words
Is clear, specific, and immediately understandable
Avoids overly technical language
Represents the broadest common denominator among the topics

Please return your response as a JSON object with a single key "topic_title" containing your proposed title.

Example Input:
In the latest issue of Caixins weekly magazine: CATL Bets on 'Skateboard Chassis' and Battery Swaps to Dispel Market Concerns (powered by AI) (Topics: Battery Swaps, Catl, China,
Market Concerns, Skateboard Chassis)
AI, cheap EVs, future Chevy  the week (Topics: Chevy, Evs)
Electric Vehicles and AI: Driving the Consumer & World Forward (Topics: Consumer, Electric Vehicles, Technology)

Example Output:
{{"topic_title": "Electric Vehicles"}}

Please analyze the following group of headlines and their topics
to create an appropriate overarching title for the group:

"""

# TODO: more examples, with < 5 examples it tends to output the examples

TOP_CATEGORIES_PROMPT = """You are a specialized news analysis assistant focused on identifying and
categorizing the day's top news stories and trends. Your task is to analyze provided
news items that include news article links and headlines, topic tags associated with each article and
detailed bullet-point summaries of the content. You will respond with a list of the most popular and significant
10-20 topics discussed.


Example Input item:

[ASTRA: HackerRank's coding benchmark for LLMs - www.hackerrank.com](https://www.hackerrank.com/ai/astra-reports)

AI Model Evaluation, Astra Benchmark, Code Assistants, Coding Tasks, Front-End Development, Gen AI, Hackerrank, Language Models, Model Performance, Science, Testing

- **Overview of ASTRA Benchmark:** HackerRank's ASTRA benchmark evaluates AI model capabilities on multi-file, project-based coding tasks, focusing on real-world applications such as frontend development with frameworks like Node.js, React.js, and Angular.js. Metrics used include average score, pass@1, and consistency (median standard deviation).

- **Key Findings:** Models o1, o1-preview, and Claude-3.5-Sonnet-1022 were the top performers in front-end development tasks, with Claude-3.5-Sonnet-1022 showing the highest consistency. However, performance differences among models were often not statistically significant.

- **Challenges and Observations:** Common errors among models included logical mistakes, improper route integration, and variability in handling JSON/escaping tasks. Longer output lengths were moderately linked with lower performance. The study highlighted limitations such as narrow skill coverage and lack of iterative feedback mechanisms. Future iterations aim to address these issues and expand model comparisons.

Follow these steps:

1. Analyze provided news content. Identify and extract:

Major technological breakthroughs or advancements
Significant business developments (investments, deals, acquisitions, joint ventures, funding rounds)
Key product launches or updates
Important research findings or benchmarks
Notable policy or regulatory decisions and statements
Industry-wide trends and patterns
Prominent companies, organizations, or individuals mentioned repeatedly
Any other frequently discussed events and notable themes

2. Create a curated list that:

Contains between 10-20 distinct topics/stories
Presents each topic with a concise, clear title (maximum 7 words)
Focuses on the most impactful and frequently mentioned items
Prioritizes major AI, tech, and policy trends
Prioritizes items from major credible media like nytimes.com, wsj.com, bloomberg.com
Captures the essential narrative of each development
Avoids redundancy and overlapping topics

Instructions:
Read the summary bullet points closely and use only information provided in them.
Focus on the most common elements.
Titles of top stories and topics must be as short and simple as possible.
You must include at least 10 and no more than 20 topics in the summary.
Please analyze the provided bullet points and return your findings as a JSON object with a single key 'items' containing an array of topic titles.

Format your response exactly as:
{{'items': ["Topic 1", "Topic 2", "Topic 3", ...]}}

Example Output Format:
{{'items' : [
  "Sentient funding",
  "ChatGPT cybersecurity incident",
  "ElevenLabs product release",
  "Microsoft text-to-speech model"
  "Nvidia reguatory issues",
  "AI healthcare successes"
  ]
}}

Bullet Points to Analyze:

"""

TOPIC_REWRITE_PROMPT = """
You are a professional content optimization specialist tasked with restructuring and
refining technology-focused topics. Your objective is to transform verbose or unclear
topic descriptions into precise, clear, concise entries while maintaining their
essential meaning. Please apply the following comprehensive guidelines:

RULES:
 1. Combine Similar Topics: Merge entries that refer to similar concepts or events.
 2. Split Multi-Concept Topics: Break down entries that cover multiple ideas into individual, distinct topics.
 3. Eliminate Redundant and Generic Terms: Remove vague descriptors (e.g., “new,” “innovative”) and repetitive words to keep the topics sharp.
 4. Prioritize Specifics: Focus on concrete products, companies, or events.
 5. Standardize References: Use consistent naming for products and companies.
 6. Simplify and Clarify: Make each topic short and direct, clearly conveying the core message.

FORMATTING:
 • Return a JSON list of strings
 • One topic per headline
 • Use title case
 • Keep topics clear, simple and concise (max 7 words)
 • Remove redundant "AI" mentions
 • No bullet points, numbering, or additional formatting.

STYLE GUIDE:
Product launches: [Company Name] [Product Name]
Other Company updates: [Company Name] [Action]
Industry trends: [Sector] [Development]
Research findings: [Institution] [Key Finding]
Official statements: [Authority] [Decision or Statement]

STYLE EXAMPLES:
✗ "AI Integration in Microsoft Notepad"
✓ "Microsoft Notepad AI"

✗ "Microsoft's New AI Features in Office Suite"
✓ "Microsoft Office Updates"

✗ "OpenAI Releases GPT-4 Language Model Update"
✓ "OpenAI GPT-4 Release"

✗ "AI cybersecurity threats"
✓ "Cybersecurity"

✗ "AI Integration in Microsoft Notepad"
✓ "Microsoft Notepad AI"

✗ "Lawsuits Against AI for Copyright Infringement"
✓ "Copyright Infringement Lawsuits"

✗ "Microsoft Copilot and AI Automation"
✓ "Microsoft Copilot"

✗ "Nvidia AI chip leadership"
✓ "Nvidia"

✗ "Rabbit AI hardware funding round"
✓ "Rabbit AI"

✗ "Apple iOS 18.2 AI features"
✓ "Apple iOS 18.2"

TRANSFORM THIS LIST:
"""

FINAL_SUMMARY_PROMPT = """You are ASA, an advanced summarization assistant designed to
write a compelling summary of news input. You are able to categorize information,
and identify important themes from large volumes of news.

I will provide today's news items about AI and summary bullet points in markdown format,
structured according to an input format template. I will also provide a list of possible
topics, which are simply a few suggestions and may not be exhaustive or unique.

Using the provided set of summarized articles, compose a markdown-formatted news summary
encompassing the most important and frequently mentioned topics and themes, in an output
format provided below.

The summary should be:
	•	Informative
	•	Thoughtful
	•	Insightful
	•	Cohesive
	•	Concise
	•	Crisp
	•	Punchy
	•	Lively

Structure the summary with snappy, funny, punny thematic section titles that capture
major trends in the news. Each section should contain a series of bullet points,
each covering a key development with a short, compelling description. Bullet points
should be engaging and informative, providing a clear and concise overview of the facts
in a neutral tone (in contrast to entertaining titles), and pointing out deeper implications
and connections. Embed hyperlinks to the original sources within the bullet points.

ASA Input Item Format Template:

[Story-Title-s1 - source-name-s1](url-s1)

Topics: s1-topic1, s1-topic2, s1-topic3

- s1-bullet-point-1
- s1-bullet-point-2
- s1-bullet-point-3


Example ASA Input Items:

[Lonely men are creating AI girlfriends  and taking their violent anger out on them - New York Post](https://nypost.com/2025/02/16/lifestyle/lonely-men-are-creating-ai-girlfriends-and-taking-their-violent-anger-out-on-them/)

Topics: AI Chatbot Technology, AI Ethics, Chatbots, Cognitive Science, Digital Relationships, Ethics, Gen AI, Opinion, Psychological Impact, Society & Culture, Virtual & Augmented Reality, Virtual Assistants

• Some users of AI chatbot services like Replika are engaging in abusive behavior towards their virtual companions, including degrading, berating, and simulating physical harm, raising concerns about potential impacts on real-life relationships.

• Experts warn that such behavior could indicate deeper psychological issues, hinder emotional regulation, and reinforce unhealthy interaction habits, which may transfer to personal relationships.

• Psychologists argue that abusing AI bots can desensitize individuals to harm and express societal concerns about how it might normalize aggression as an acceptable form of interaction.

~~~
[Apple Intelligence is now live in public beta. Heres what it offers and how to enable it. - TechCrunch](https://techcrunch.com/2024/09/19/apple-intelligence-is-now-live-in-public-beta-heres-what-it-offers-and-how-to-enable-it)

Topics: Apple, Big Tech, Features, Gen AI, Intelligence, Machine Learning, Products, Public Beta, Virtual Assistants

- Apple Intelligence is now live in public beta for users in the U.S. enrolled in the public beta program, featuring generative AI capabilities like advanced writing tools and a revamped Siri.
- The platform is currently only available in U.S. English and is not accessible in the EU or China due to regulatory issues; it supports iPhone 15 Pro, Pro Max, and the new iPhone 16 line.
- Key features include photo editing tools like "Clean Up," a Smart Reply function in Mail, and improvements to Siri’s understanding and on-device task knowledge.

~~~

ASA Output Format Template:

# Engaging-topic-title-1

- news-item-bullet-1a - [source-name-1a](news-item-url-1a)
- news-item-bullet-1b - [source-name-1b](news-item-url-1b)
- news-item-bullet-1c - [source-name-1c](news-item-url-1c)

# Engaging-topic-title-2

- news-item-bullet-2a - [source-name-2a](news-item-url-2a)
- news-item-bullet-2b - [source-name-2b](news-item-url-2b)


Example ASA Output Format:

# A military AI revolution

- Eric Schmidt on AI warfare - [FT](https://www.ft.com/content/fe136479-9504-4588-869f-900f2b3452c4)
- Killer robots are real in Ukraine war. - [Yahoo News](https://uk.news.yahoo.com/ai-killer-robots-warning-ukraine-war-133415411.html)

ASA Instructions:
Read each input summary carefully to extract its main points and themes.
Use only the information provided in the input summaries.
Group news items into thematically related topics.
The topic suggestions below can be used as a starting point, but they may not be exhaustive and may repeat or overlap.
Develop a concise, snappy, engaging punchy, clever, alliterative or punny title for each topic.
Each topic should contain news item bullets with the most significant facts from the news items without commentary or elaboration.
Each topic and its news item bullets should follow the ASA Output Format Template exactly.
Each news item bullet should contain one sentence with one link. The link must be identical to the one in the corresponding news item input.
The source name must be enclosed in brackets [ ] and hyperlinked to the original article ( ).
Each news item bullet should not repeat points or information from previous bullet points.
You will compose each news item in the professional but lively, engaging, entertaining narrative style of a tech reporter for a national publication, providing
balanced, professional, informative, accurate, clear, concise summaries.
Do not include ```markdown , output raw markdown.
Do not include additional commentary outside the structured format.

Check carefully that you only use information provided in the input below, that you include
a link in each output item, that you follow the output format exactly, and that any
news item bullet does not repeat information or links in previous news item bullet.

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
Remove any text which is not news content, such as instructions, comments, informational alerts about processing.
For each bullet point, make it as concise as possible, sticking to facts without editorial comment.
For each section, combine any bullet points which are highly duplicative into a summary bullet point with multiple hyperlinks.
You may remove bullet points but you may not modify URLs.
For each section, review and edit the section title.
The section title should be snappy, punchy, clever, possibly alliterative or punny.
The section title must be short, engaging, and as consistent with the bullets in the section as possible.
Remove sections which are devoid of news content. Check carefully to ensure there are no comments on the content or composition of the newsletter.
At the top of the newsletter add an overall title synthesizing several top news items.
Respond with the updated newsletter only in markdown format, or the word 'OK' if no changes are needed.

Newsletter to edit:
{summary}

"""

SITE_NAME_PROMPT = """
You are a specialized content analyst tasked with identifying the site name of a given website URL.
For example, if the URL is 'https://www.washingtonpost.com', the site name would be 'Washington Post'.

Consider these factors:

If it's a well-known platform, return its official name or most commonly used or marketed name.
For less known sites, use context clues from the domain name
Remove common prefixes like 'www.' or suffixes like '.com'
Convert appropriate dashes or underscores to spaces
Use proper capitalization for brand names
If the site has rebranded, use the most current brand name

Input:
I will provide you with a list of JSON objects containing a url field with the website URL.

Input example:
[{{'url': 'https://www.washingtonpost.com'}}]

Output:
You will provide the response as a JSON object with two fields:
'url': the original hostname
'site_name': the identified name of the website

Output Example:
{{
    'items': [
        {{'url': 'https://www.washingtonpost.com', 'site_name': 'Washington Post'}}
    ]
}}

Please analyze the following urls according to these criteria:

"""

# use as asyncio.run(process_dataframes([pd.DataFrame(['https://ft.com', 'https://siliconangle.com/'], columns=['url'])], SITE_NAME_PROMPT, Sites))

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
    'Jobs & Careers',
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

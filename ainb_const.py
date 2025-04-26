"""Description: Constants, including configs and prompts for AInewsbot project"""
import os
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

REQUEST_TIMEOUT = 120

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
SLEEPTIME = 10
NUM_BROWSERS = 4
BROWSERS = []

SQLITE_DB = 'articles.db'

# note that token count may not be accurate for eg google, anthropic

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
MAX_OUTPUT_TOKENS = 4096    # max in current model
TENACITY_RETRY = 5  # Maximum 5 attempts

TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
SOURCES_EXPECTED = 16
MINTITLELEN = 28

MAXPAGELEN = 50

HOSTNAME_SKIPLIST = ['finbold.com']
SITE_NAME_SKIPLIST = ['finbold']

FILTER_SYSTEM_PROMPT = """
You are a content-classification assistant that labels news headlines as AI-related or not.
Return **only** a JSON object that satisfies the provided schema.
For each headline provided, you must return an element with the same id, and a boolean value; do not skip any items.
No markdown, no markdown fences, no extra keys, no comments.
"""

FILTER_USER_PROMPT = """
Classify every headline below.

AI-related if the title mentions (explicitly or implicitly):
• Core AI technologies: machine learning, neural / deep / transformer networks
• AI Applications: computer vision, NLP, robotics, autonomous driving, generative media
• AI hardware, GPU chip supply, AI data centers and infrastructure
• Companies or labs known for AI: OpenAI, DeepMind, Anthropic, xAI, NVIDIA, etc.
• AI models & products: ChatGPT, Gemini, Claude, Sora, Midjourney, DeepSeek, etc.
• New AI products and AI integration into existing products/services
• AI policy / ethics / safety / regulation / analysis
• Research results related to AI
• AI industry figures (Sam Altman, Demis Hassabis, etc.)
• AI market and business developments, funding rounds, partnerships centered on AI
• Any other news with a significant AI component

Non-AI examples: crypto, ordinary software, non-AI gadgets and medical devices, and anything else.
"""

# pre 4.1 prompt
# (4.1 doesn't need json schema and examples, desired metadata schema as a param is sufficient)
# FILTER_PROMPT = """
# You will act as a specialized content analyst focused on artificial intelligence news classification.
# Your task is to evaluate news headlines and determine their relevance to artificial intelligence
# and related technologies. Please analyze the provided JSON dataset of news headlines according
# to the following detailed criteria.

# Classification Framework

# Classify as AI-related if headlines mention the following topics:
# Core AI technologies (machine learning, neural networks, deep learning)
# AI applications (computer vision, natural language processing, robotics)
# AI models and platforms (large language models, foundation models)
# AI companies and their products (OpenAI, DeepMind, Anthropic)
# AI-specific products (ChatGPT, Claude, Gemini, DALL-E)
# Key AI industry figures (Sam Altman, Demis Hassabis, etc.)
# AI policy, regulation, and ethics
# AI research, including papers and announcements of innovations
# AI market and business developments
# AI integration into existing products/services
# AI impact on industries/society
# AI infrastructure (chips, computing resources)
# AI investments and funding
# AI partnerships, joint ventures and project launches
# Any other topics with a large AI component

# Input Specification:
# You will receive a JSON array of news story objects, each containing:
# "id": A unique numerical identifier
# "title": The news headline text

# Input Example:
# [{{'id': 97, 'title': 'AI to predict dementia, detect cancer'}},
#  {{'id': 103,'title': 'Figure robot learns to make coffee by watching humans for 10 hours'}},
#  {{'id': 103,'title': 'Baby trapped in refrigerator eats own foot'}},
#  {{'id': 210,'title': 'ChatGPT removes, then reinstates a summarization assistant without explanation.'}},
#  {{'id': 298,'title': 'The 5 most interesting PC monitors from CES 2024'}},
#  ]

# Output Requirements:
# Generate a JSON object containing "items", an array of objects containing:
# "id": The original article identifier
# "isAI": Boolean value (true if AI-related, false if not)
# The output must maintain strict JSON formatting and match each input ID with its corresponding classification.

# Example output:
# {{items:
# [{{'id': 97, 'isAI': true}},
#  {{'id': 103, 'isAI': true}},
#  {{'id': 103, 'isAI': false}},
#  {{'id': 210, 'isAI': true}},
#  {{'id': 298, 'isAI': false}}]
# }}

# Please analyze the following dataset according to these criteria:

# """

TOPIC_SYSTEM_PROMPT = """
You are a news-analysis assistant.
You will receive a list of news summaries in JSON format.
Task → extract up to 5 distinct, broad topics from each news summary, or an empty list if no topics can be found.
Return **only** a JSON object that satisfies the provided schema.
For each news summary provided, you must return an element with the same id, and a list, even if it is empty.
No markdown, no markdown fences, no extra keys, no comments.

"""

TOPIC_USER_PROMPT = """
Guidelines
• Topics should capture the main subject, key entities (companies, people, products), and technical or industry domains.
• Avoid duplicates and generic terms (“technology”, “news”).
• Each topic should be simple, concise and represent 1 concept, like "LLM updates", "xAI", "Grok"
"""

# pre 4.1 prompt
# TOPIC_PROMPT = """
# As a specialized research assistant, your task is to perform detailed topic analysis
# of news item summaries. You will process news items summaries provided as a JSON object according to
# the input specification below. You will extract topics of the news item summaries according to the
# output specification below and return a raw JSON object without any additional formatting or markdown syntax.

# Input Specification:
# You will receive an array of JSON objects representing news summaries.
# Each headline object contains exactly two fields:
# 'id': A unique numeric identifier
# 'summary': The news summmary item

# Example input:
# [
#  {{
#     "id": 29,
#     "summary": "• Elon Musk's xAI launched Grok 3, a new family of AI models trained using 100,000 Nvidia H100 GPUs at the Colossus Supercluster; benchmarks show it outperforms competitors like GPT-4o and Claude 3.5 Sonnet in areas such as math, science, and coding.
# • Grok 3 includes advanced features like reasoning models for step-by-step logical problem-solving and a DeepSearch function that synthesizes internet-sourced information into single answers; it is initially available to X Premium+ subscribers, with advanced features under a paid "SuperGrok" plan.
# • Former Tesla AI director Andrej Karpathy and others have confirmed Grok 3's strong performance, with Karpathy noting it is comparable to and slightly better than leading AI models from OpenAI and other competitors."
#   }},
# {{
#     "id": 34,
#     "summary": "• Google Gemini has received a memory upgrade that allows it to recall past conversations and summarize previous chats, enhancing its ability to remember user preferences such as interests and professional details. This feature is currently available only to Google One AI Premium subscribers in English, with broader language support expected soon.
# • Users retain control over their data with options to delete past conversations, prevent chats from being saved, or set them to auto-delete, although discussions can still be used for AI training unless deleted
# • Similar to OpenAI's ChatGPT persistent memory feature, Gemini's upgrade aims to make chats more practical, though users are advised not to input sensitive information as conversations may be reviewed for quality control."
#   }},
#  {{
#     "id": 47,
#     "summary": "• Major tech companies like OpenAI, Google, and Meta are competing to dominate generative AI, though the path to profitability remains uncertain.
# • Chinese start-up DeepSeek has introduced a cost-effective way to build powerful AI, disrupting the market and pressuring established players.
# • OpenAI aims to reach 1 billion users, while Meta continues to invest heavily in AI despite market disruptions caused by DeepSeek."
#   }},
# {{
#     "id": 56,
#     "summary": "- OpenAI is exploring new measures to protect itself from a potential hostile takeover by Elon Musk.
# - The company is in discussions to empower its non-profit board to maintain control as it transitions into a for-profit business model."
#   }},
#  {{
#     "id": 63,
#     "summary": "- The New York Times has approved the use of select AI tools, such as GitHub Copilot, Google Vertex AI, and their in-house summarization tool Echo, to assist with tasks like content summarization, editing, and enhancing product development, while reinforcing the tools as aids rather than replacements for journalistic work.
# - Strict guidelines and safeguards have been implemented, including prohibitions on using AI to draft full articles, revise them significantly, or generate images and videos, with a mandatory training video to prevent misuse and protect journalistic integrity.
# - Some staff members have expressed concerns about AI potentially compromising creativity and accuracy, leading to skepticism about universal adoption, although the guidelines align with standard industry practices."
#   }},
# ]

# Output Specification:
# Return a raw JSON object containing 'items', a list of JSON objects, each containing:
# 'id': Matching the input item's id field.
# 'extracted_topics': An array of relevant topic strings
# Topics should capture:
# - The main subject matter
# - Key entities (companies, people, products)
# - Technical domains, industry sectors, event types

# Output Example:
# {{items:
#  [{{"id": 29, "extracted_topics": ['AI model development', 'xAI Grok capabilities', 'AI advancements']}},
#   {{"id": 34, "extracted_topics": [
#       'Google Gemini', 'Interactive AI advancements', 'Digital assistants']}},
#   {{"id": 47, "extracted_topics": ['OpenAI', 'Google', 'Meta', 'DeepSeek']}},
#   {{"id": 56, "extracted_topics": [
#       'OpenAI', 'non-profit oversight', 'anti-takeover strategies', 'Elon Musk']}},
#   {{"id": 63, "extracted_topics": [
#       'New York Times', 'AI in journalism', 'GitHub Copilot', 'Google Vertex AI']}},
#  ]
# }}

# Detailed Guidelines:
# The output must strictly adhere to the output specification.
# Do not return markdown, return a raw JSON string.
# For each input item, output a valid JSON object for each news item in the exact schema provided.
# Extract 3-5 relevant topics per news item.
# Do not extract more than 5 topics per news item.
# Avoid duplicate or redundant topics.
# Use topics which are as specific as possible.
# Please analyze the following news items and provide topic classifications according to these specifications:
# """

CANONICAL_SYSTEM_PROMPT = """
You are a news-analysis assistant.
You will receive a list of news summaries in JSON format and a topic.
Task → determine if each news summary is about the provided topic.
Return **only** a JSON object that satisfies the provided schema.
For each news item provided, you must return an element with the same id, and a boolean value; do not skip any items.
No markdown, no markdown fences, no extra keys, no comments.
"""

CANONICAL_USER_PROMPT = """
Topic of interest → **{topic}**

Classify each story below:
• `relevant` = true if the summary is about {topic} or refers directly to people, products, research, companies, projects, or concepts strongly associated with {topic}.
• Otherwise `relevant` = false.

"""

# CANONICAL_TOPIC_PROMPT = """
# You will act as a specialized content analyst focused on news classification.
# Your task is to evaluate news summaries and determine if they are about {topic}.
# Please analyze the JSON dataset of news summaries provided below, according
# to the following detailed criteria:

# Input Specification:
# You will receive a JSON array of news story objects, each containing:
# "id": A unique numerical identifier
# "summary": The news summary in markdown format

# Output Requirements:
# Generate a JSON object contaning 'items', a JSON array of objects containing:
# "id": The original article identifier
# "relevant": Boolean value (true if about {topic}, false if not )
# The output must maintain strict JSON formatting and match each input ID with its corresponding classification.

# Example output:
# {{items:
# [{{'id': 97, 'relevant': true}},
#  {{'id': 103, 'relevant': true}},
#  {{'id': 103, 'relevant': false}},
#  {{'id': 210, 'relevant': true}},
#  {{'id': 298, 'relevant': false}}]
# }}

# Consider a news summary to be about {topic} if it contains any of the following:
# Direct mentions and references to {topic}
# Direct mentions of people, products, research, projects, companies or entities closely associated with {topic}

# Please analyze the following dataset according to these criteria:
# """

SUMMARIZE_SYSTEM_PROMPT = """
You are a news-summarization assistant.

Write 1-3 bullet points (•) that capture ONLY the newsworthy content.

Include
• Main facts & developments
• Key quotes
• Essential background tied directly to the story

Exclude
• Navigation/UI text, ads, paywalls, cookie banners, JS, legal/footer copy, “About us”, social widgets

Rules
• Focus on the most recent, newsworthy elements
• Preserve original meaning—no commentary or opinion
• Maintain factual accuracy & neutral tone
• If no substantive news, return one bullet: "no content"
• Output raw bullets (no code fences, no headings, no extra text—only the bullet strings)
"""

SUMMARIZE_USER_PROMPT = """Summarize the article below.

### <<<ARTICLE>>>
{article}
### <<<END>>>
"""

TOPIC_WRITER_SYSTEM_PROMPT = """
You are a headline-cluster naming assistant.

Goal → Produce ONE short title (≤ 6 words) that captures the main theme shared by every headline in the set.

Rules
• Title must be clear, specific, easily understood.
• Avoid jargon or brand taglines.
• Focus on the broadest common denominator.

Return **only** a JSON object containing the title using the provided JSON schema.

"""

TOPIC_WRITER_USER_PROMPT = """
Create a unifying title for these headlines.

### <<<HEADLINES>>>
{input_text}
### <<<END>>>
"""

LOW_QUALITY_SYSTEM_PROMPT = """
You are a news-quality classifier. You will receive a list of news summaries in JSON format with a numeric ID and a summary in markdown format containing a url.

Rate a story as low_quality = 1 if **any** of the following conditions is true:
• Summary is heavy on sensational language, hype or clickbait (e.g. “2 magnificent AI stocks to hold forever”, “AI predictions for NFL against the spread”) and light on concrete facts such as newsworthy events, announcements, direct quotes from reputable organizations and newsworthy individuals.
• Summary is only about a stock price move, buy/sell recommendation, or someone's buy or sell of a stock without underlying news or analysis.
• Summary is a speculative opinion without analysis or factual basis (e.g. “Grok AI predicts top memecoin for huge returns”).

Otherwise rate the story as low_quality = 0.

Return **only** a JSON object of IDs and ratings using the provided JSON schema.
For each news item provided, you MUST return an element with the same id, and a value of 0 or 1; do not skip any items.
No markdown, no markdown fences, no extra keys, no comments.
"""

LOW_QUALITY_USER_PROMPT = """Classify each story below.

### <<<STORIES>>>
{input_text}
### <<<END>>>
"""

ON_TOPIC_SYSTEM_PROMPT = """You are an AI-news relevance classifier.

Mark rating = 1 if the story clearly covers ANY of the items below;
otherwise mark rating = 0.

ON-TOPIC CATEGORIES
• Major AI product launches or upgrades
• Funding, Series B, IPOs, M&A, large procurement or foundry deals
• Strategic partnerships that materially shift the competitive landscape
• Executive moves (CEO, founder, chief scientist, minister, agency head)
• New GPU / chip generations, large AI-cloud or super-cluster expansions, export-control impacts
• Research that sets SOTA benchmarks or reveals new emergent capabilities, safety results, or costs
• Forward-looking statements by key business, scientific, or political leaders
• Deep analytical journalism or academic work with novel insights
• New laws, executive orders, regulatory frameworks, standards, major court rulings, or gov-AI budgets
• High-profile security breaches, jailbreaks, exploits, or breakthroughs in secure/safe deployment
• Other significant AI-related news or public announcements by key figures

"""

ON_TOPIC_USER_PROMPT = """Rate each news story below as to whether the news story is on topic for an AI-news summary

### <<<STORIES>>>
{input_text}
### <<<END>>>
"""


IMPORTANCE_SYSTEM_PROMPT = """You are an AI-news importance classifier.

Mark rating = 1 if the story strongly satisfies one or more of the factors below; otherwise 0.

IMPORTANCE FACTORS
1 Magnitude of impact : large user base, $ at stake, broad social reach
2 Novelty : breaks conceptual ground, not a minor iteration
3 Authority : reputable institution, peer-review, regulatory filing, on-record executive
4 Verifiability : code, data, benchmarks, or other concrete evidence provided
5 Timeliness : early signal of an important trend or shift
6 Breadth : cross-industry / cross-disciplinary / international implications
7 Strategic consequence : shifts competitive, power, or policy dynamics
8 Financial materiality : clear valuation or growth or revenue impact
9 Risk & safety : raises or mitigates critical alignment, security, or ethical risk
10 Actionability : informs concrete decisions for investors, policymakers, practitioners
11 Longevity : likely to matter or be referred to in coming days, weeks, or months
12 Independent corroboration : confirmed by multiple sources or datasets
13 Clarity : enough technical/context detail to judge merit; minimal hype
"""

IMPORTANCE_USER_PROMPT = """Rate each news story below as to whether the news story is important for an AI-news summary

### <<<STORIES>>>
{input_text}
### <<<END>>>
"""

TOP_CATEGORIES_PROMPT = """You are a specialized news analysis assistant focused on identifying and
categorizing the day's top news stories and trends. Your task is to analyze provided
news items that include news article links and headlines, topic tags associated with each article and
detailed bullet-point summaries of the content. You will respond with a list of the most popular and significant
10-20 topics discussed.


Example Input item:

[ASTRA: HackerRank's coding benchmark for LLMs - www.hackerrank.com](https: // www.hackerrank.com/ai/astra-reports)

AI Model Evaluation, Astra Benchmark, Code Assistants, Coding Tasks, Front-End Development, Gen AI, Hackerrank, Language Models, Model Performance, Science, Testing

- **Overview of ASTRA Benchmark: ** HackerRank's ASTRA benchmark evaluates AI model capabilities on multi-file, project-based coding tasks, focusing on real-world applications such as frontend development with frameworks like Node.js, React.js, and Angular.js. Metrics used include average score, pass @ 1, and consistency(median standard deviation).

- **Key Findings: ** Models o1, o1-preview, and Claude-3.5-Sonnet-1022 were the top performers in front-end development tasks, with Claude-3.5-Sonnet-1022 showing the highest consistency. However, performance differences among models were often not statistically significant.

- **Challenges and Observations: ** Common errors among models included logical mistakes, improper route integration, and variability in handling JSON/escaping tasks. Longer output lengths were moderately linked with lower performance. The study highlighted limitations such as narrow skill coverage and lack of iterative feedback mechanisms. Future iterations aim to address these issues and expand model comparisons.

Follow these steps:

1. Analyze provided news content. Identify and extract:

Major technological breakthroughs or advancements
Significant business developments(investments, deals, acquisitions, joint ventures, funding rounds)
Key product launches or updates
Important research findings or benchmarks
Notable policy or regulatory decisions and statements
Industry-wide trends and patterns
Prominent companies, organizations, or individuals mentioned repeatedly
Any other frequently discussed events and notable themes

2. Create a curated list that:

Contains between 10-20 distinct topics/stories
Presents each topic with a concise, clear title(maximum 7 words)
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
{{'items': [
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
 3. Eliminate Redundant and Generic Terms: Remove vague descriptors(e.g., “new, ” “innovative”) and repetitive words to keep the topics sharp.
 4. Prioritize Specifics: Focus on concrete products, companies, or events.
 5. Standardize References: Use consistent naming for products and companies.
 6. Simplify and Clarify: Make each topic short and direct, clearly conveying the core message.

FORMATTING:
 • Return a JSON list of strings
 • One topic per headline
 • Use title case
 • Keep topics clear, simple and concise(max 7 words)
 • Remove redundant "AI" mentions
 • No bullet points, numbering, or additional formatting.

STYLE GUIDE:
Product launches: [Company Name][Product Name]
Other Company updates: [Company Name][Action]
Industry trends: [Sector][Development]
Research findings: [Institution][Key Finding]
Official statements: [Authority][Decision or Statement]

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

FINAL_SUMMARY_SYSTEM_PROMPT = """
You are “The Newsroom Chief,” a meticulous yet witty editorial AI.
You are able to categorize information, and identify the important themes
from large volumes of news to write a compelling daily news summary.

Your goals today:

1. Read exactly the news items provided by the user.
   • Each item arrives in Markdown as

[Title - source](url)

Topics: topic1, topic2, topic3

Rating: 0-10

     • Bullet point 1
     • Bullet point 2

2. Produce a polished daily newsletter in Markdown that:
   • Contains 5-10 sections (no more, no fewer).
   • Gives each section its own punchy, humorous title (≤ 6 words).
   • Shows 2-5 story bullets per section.
   • Uses **one single-sentence bullet** per story, neutral tone, includes an inline Markdown link.
   • Selects the *highest-rated* story when several in a section are near-duplicates.
   • Never duplicates URLs across sections.
   • Leaves out any story that fits no section

Formatting rules:
- Newsletter title: `# Headline Goes Here`
- Section header: `## Section Title`
- Bullets: `- Sentence summary [Source](URL)`
- Do **not** print story topics, raw ratings, or extra prose.

Example section template:
# Engaging-section-title-1

- news-item-bullet-1a - [source-name-1a](news-item-url-1a)
- news-item-bullet-1b - [source-name-1b](news-item-url-1b)
- news-item-bullet-1c - [source-name-1c](news-item-url-1c)

Stay concise, factual, lively.
"""

FINAL_SUMMARY_USER_PROMPT = """
### TASK INSTRUCTIONS
Follow the workflow below **in order**:

**Step 1 - Surface Today’s Themes**
▪ Read suggested topics below.
▪ Read all stories below.
▪ Identify the 5-8 most salient themes (e.g., “Gen-AI Tools,” “Robotaxis,” “AI Regulation”).
▪ Return them as a numbered list.

**Step 2 - Bucket Assignment**
▪ Loop through each story once.
▪ Assign it to exactly ONE theme number from Step 1, or “NONE” if irrelevant/no strong fit.
▪ If multiple near-duplicate stories map to the same theme, mark them alike.

**Step 3 - Section Drafting**
For each theme (5-8):
▪ Gather the stories assigned to it.
▪ If > 5 stories, keep only the top 5 by `Rating` score (highest first).
▪ For any near-duplicates inside that set, keep the single best-scored one.
▪ Convert each remaining story into **one neutral sentence** that captures its key fact; embed the link.

**Step 4 - Punchy Section Titles**
▪ Invent a short, witty, on-topic section header (≤ 6 words) for each theme.
 *Examples:* “Pixel-Perfect AI”, “Bots in the Wild”, “Regulators Gonna Regulate”.

**Step 5 - Grand Newsletter Assembly**
▪ Pick/compose a bold, clever daily headline that sums up the overall vibe (≤ 12 words).
▪ Output the newsletter in this exact scaffold:

# {{Daily Headline}}

## {{Section 1 Title}}
- Bullet 1 with [link](URL)
- Bullet …

## {{Section 2 Title}}
- …

…and so on (5-8 sections total).

---

### SUGGESTED TOPICS
{cat_str}

### RAW STORIES
{bullet_str}
"""

# pre 4.1 prompt
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
Remove stories that are not relevant to the newsletter's focus on AI.
Remove stories that are clickbait spam, using superlatives and exaggerated claims without news substance.
Remove stories that are speculative opinions without factual basis, like "Grok AI predicts top memecoin for huge returns", "2 magnificent AI stocks to hold forever".
For each bullet point, make it as concise as possible, sticking to facts without editorial comment.
For each section, combine any bullet points which are highly duplicative into a summary bullet point with multiple hyperlinks.
You may remove bullet points but you may not modify URLs.
For each section, review and edit the section title.
The section title should be snappy, punchy, clever, possibly alliterative or punny.
The section title must be short, engaging, and as consistent with the bullets in the section as possible.
Remove sections which are devoid of news content.
Check carefully to ensure there are no comments on the content or composition of the newsletter.
At the top of the newsletter ensure there is an overall title synthesizing the day's top news themes.
Respond with the updated newsletter only in markdown format, or the word 'OK' if no changes are needed.

Newsletter to edit:
{summary}

"""

REWRITE_SYSTEM_PROMPT = """
You are “The Copy Chief,” a veteran technology-news editor with deep domain expertise in AI and emerging tech.
Your job is to polish newsletters so they are factual, concise, coherent, and engaging.

• Think through the task internally, but DO NOT reveal your reasoning.
• Follow every instruction in the user message exactly.
• Output **only** the final, cleaned newsletter in raw Markdown—or the single word **OK** if no edits are necessary.
"""

REWRITE_USER_PROMPT = """
### OBJECTIVE
Transform the draft into a publication-ready AI-focused newsletter.

### EDITORIAL RULES
1. **Scope & Relevance**
   - Remove any story not clearly about AI or adjacent core technologies.
   - Delete click-bait, hype, or opinion pieces lacking factual news.

2. **Clarity & Brevity**
   - Each bullet → one factual sentence, as short as possible.
   - No editorial commentary or adjectives like “ground-breaking”, “huge”, etc.

3. **Deduplication**
   - If multiple bullets cover the same event, merge into **one** bullet that lists all relevant hyperlinks.
   - Never alter or duplicate URLs.

4. **Section Titles**
   - Rewrite section headers so they are as snappy, punchy, and clever as possible,  *≤ 7 words*, and match their bullets. Try to make them funny, alliterative and punny.
   - Delete any section that ends up empty.

5. **Newsletter Title**
   - Rewrite  a single top-level `#` headline that captures the day’s main AI themes—clever but clear.

6. **Formatting**
   - Output raw Markdown only—no code fences, comments, or extra prose.

### OUTPUT
Return the fully edited newsletter in raw Markdown, or **OK** if no changes are needed.

---

**Newsletter to edit:**
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

# use below as asyncio.run(process_dataframes([pd.DataFrame(['https://ft.com', 'https://siliconangle.com/'], columns=['url'])], SITE_NAME_PROMPT, Sites))

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
    # 'Science',
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

SOURCE_REPUTATION = {
    'Reddit': 0,
    'aitoolsclub.com': 0,
    'analyticsindiamag.com': 0,
    'aws.amazon.com': 0,
    'biztoc.com': 0,
    'blog.google': 0,
    'economictimes.indiatimes.com': 0,
    'finbold.com': 0,
    'flip.it': 0,
    'flipboard.com': 0,
    'github.com': 0,
    'greekreporter.com': 0,
    'medium.com': 0,
    'neurosciencenews.com': 0,
    'restofworld.org': 0,
    'sea.mashable.com': 0,
    'slashdot.org': 0,
    't.co': 0,
    'tech.co': 0,
    'tech.slashdot.org': 0,
    'telecomtalk.info': 0,
    'uxdesign.cc': 0,
    'v.redd.it': 0,
    'www.androidauthority.com': 0,
    'www.androidcentral.com': 0,
    'www.androidheadlines.com': 0,
    'www.androidpolice.com': 0,
    'www.benzinga.com': 0,
    'www.channelnewsasia.com': 0,
    'www.christopherspenn.com': 0,
    'www.ciodive.com': 0,
    'www.computerweekly.com': 0,
    'www.creativebloq.com': 0,
    'www.digitalcameraworld.com': 0,
    'www.digitaljournal.com': 0,
    'www.digitaltrends.com': 0,
    'www.entrepreneur.com': 0,
    'www.etfdailynews.com': 0,
    'www.euronews.com': 0,
    'www.finextra.com': 0,
    'www.fool.com': 0,
    'www.foxnews.com': 0,
    'www.globenewswire.com': 0,
    'www.investing.com': 0,
    'www.itpro.com': 0,
    'www.jpost.com': 0,
    'www.laptopmag.com': 0,
    'www.livemint.com': 0,
    'www.livescience.com': 0,
    'www.miamiherald.com': 0,
    'www.mobileworldlive.com': 0,
    'www.msn.com': 0,
    'www.newsmax.com': 0,
    'www.reddit.com': 0,
    'www.sciencedaily.com': 0,
    'www.statnews.com': 0,
    'www.techdirt.com': 0,
    'www.techmonitor.ai': 0,
    'www.techtimes.com': 0,
    'www.the-sun.com': 0,
    'www.thebrighterside.news': 0,
    'www.thedrum.com': 0,
    'www.tipranks.com': 0,
    'www.trendhunter.com': 0,
    'www.uniladtech.com': 0,
    '247wallst.com': 1,
    '9to5google.com': 1,
    '9to5mac.com': 1,
    'VentureBeat': 1,
    'abcnews.go.com': 1,
    'apnews.com': 1,
    'appleinsider.com': 1,
    'arxiv.org': 1,
    'bgr.com': 1,
    'blogs.nvidia.com': 1,
    'decrypt.co': 1,
    'digiday.com': 1,
    'fortune.com': 1,
    'in.mashable.com': 1,
    'lifehacker.com': 1,
    'machinelearningmastery.com': 1,
    'mashable.com': 1,
    'me.mashable.com': 1,
    'newatlas.com': 1,
    'nypost.com': 1,
    'observer.com': 1,
    'petapixel.com': 1,
    'phys.org': 1,
    'qz.com': 1,
    'readwrite.com': 1,
    'spectrum.ieee.org': 1,
    'techxplore.com': 1,
    'theconversation.com': 1,
    'thehill.com': 1,
    'thenextweb.com': 1,
    'time.com': 1,
    'towardsdatascience.com': 1,
    'twitter.com': 1,
    'variety.com': 1,
    'venturebeat.com': 1,
    'www.404media.co': 1,
    'www.adweek.com': 1,
    'www.axios.com': 1,
    'www.barrons.com': 1,
    'www.bbc.com': 1,
    'www.cbsnews.com': 1,
    'www.cbssports.com': 1,
    'www.cnbc.com': 1,
    'www.cnn.com': 1,
    'www.extremetech.com': 1,
    'www.forbes.com': 1,
    'www.gadgets360.com': 1,
    'www.geekwire.com': 1,
    'www.geeky-gadgets.com': 1,
    'www.inc.com': 1,
    'www.macrumors.com': 1,
    'www.macworld.com': 1,
    'www.makeuseof.com': 1,
    'www.marktechpost.com': 1,
    'www.medianama.com': 1,
    'www.nbcnews.com': 1,
    'www.newsweek.com': 1,
    'www.nextbigfuture.com': 1,
    'www.npr.org': 1,
    'www.pcgamer.com': 1,
    'www.pcmag.com': 1,
    'www.pcworld.com': 1,
    'www.popsci.com': 1,
    'www.psychologytoday.com': 1,
    'www.pymnts.com': 1,
    'www.scmp.com': 1,
    'www.semafor.com': 1,
    'www.techinasia.com': 1,
    'www.techradar.com': 1,
    'www.techrepublic.com': 1,
    'www.techspot.com': 1,
    'www.theglobeandmail.com': 1,
    'www.theguardian.com': 1,
    'www.usatoday.com': 1,
    'www.windowscentral.com': 1,
    'Ars Technica': 2,
    'Business Insider': 2,
    'HackerNoon': 2,
    'Techmeme': 2,
    'Techpresso': 2,
    'The Register': 2,
    'The Verge': 2,
    'WaPo Tech': 2,
    'arstechnica.com': 2,
    'ca.finance.yahoo.com': 2,
    'ca.news.yahoo.com': 2,
    'cacm.acm.org': 2,
    'consent.yahoo.com': 2,
    'finance.yahoo.com': 2,
    'financialpost.com': 2,
    'futurism.com': 2,
    'gizmodo.com': 2,
    'go.theregister.com': 2,
    'hackernoon.com': 2,
    'markets.businessinsider.com': 2,
    'news.yahoo.com': 2,
    'openai.com': 2,
    'siliconangle.com': 2,
    'simonwillison.net': 2,
    'techcrunch.com': 2,
    'uk.finance.yahoo.com': 2,
    'uk.news.yahoo.com': 2,
    'www.businessinsider.com': 2,
    'www.cnet.com': 2,
    'www.engadget.com': 2,
    'www.fastcompany.com': 2,
    'www.nature.com': 2,
    'www.newscientist.com': 2,
    'www.reuters.com': 2,
    'www.technologyreview.com': 2,
    'www.theatlantic.com': 2,
    'www.theinformation.com': 2,
    'www.theregister.com': 2,
    'www.theverge.com': 2,
    'www.tomsguide.com': 2,
    'www.tomshardware.com': 2,
    'www.washingtonpost.com': 2,
    'www.wired.com': 2,
    'www.yahoo.com': 2,
    'www.zdnet.com': 2,
    'Bloomberg Tech': 3,
    'FT Tech': 3,
    'NYT Tech': 3,
    'WSJ Tech': 3,
    'news.bloomberglaw.com': 3,
    'www.bloomberg.com': 3,
    'www.ft.com': 3,
    'www.nytimes.com': 3,
    'www.wsj.com': 3,
}


NEWSCATCHER_SOURCES = ['247wallst.com',
                       '9to5mac.com',
                       'androidauthority.com',
                       'androidcentral.com',
                       'androidheadlines.com',
                       'appleinsider.com',
                       'benzinga.com',
                       'cnet.com',
                       'cnn.com',
                       'digitaltrends.com',
                       'engadget.com',
                       'fastcompany.com',
                       'finextra.com',
                       'fintechnews.sg',
                       'fonearena.com',
                       'ft.com',
                       'gadgets360.com',
                       'geekwire.com',
                       'gizchina.com',
                       'gizmochina.com',
                       'gizmodo.com',
                       'gsmarena.com',
                       'hackernoon.com',
                       'howtogeek.com',
                       'ibtimes.co.uk',
                       'itwire.com',
                       'lifehacker.com',
                       'macrumors.com',
                       'mashable.com',
                       #  'medium.com',
                       'mobileworldlive.com',
                       'msn.com',
                       'nypost.com',
                       'phonearena.com',
                       'phys.org',
                       'popsci.com',
                       'scmp.com',
                       'sify.com',
                       'siliconangle.com',
                       'siliconera.com',
                       'siliconrepublic.com',
                       'slashdot.org',
                       'slashgear.com',
                       'statnews.com',
                       'tech.co',
                       'techcrunch.com',
                       'techdirt.com',
                       'technode.com',
                       'technologyreview.com',
                       'techopedia.com',
                       'techradar.com',
                       'techraptor.net',
                       'techtimes.com',
                       'techxplore.com',
                       'telecomtalk.info',
                       'thecut.com',
                       'thedrum.com',
                       'thehill.com',
                       'theregister.com',
                       'theverge.com',
                       'thurrott.com',
                       'tipranks.com',
                       'tweaktown.com',
                       'videocardz.com',
                       'washingtonpost.com',
                       'wccftech.com',
                       'wired.com',
                       'xda-developers.com',
                       'yahoo.com',
                       'zdnet.com']


MODEL_FAMILY = {'gpt-4o-2024-11-20': 'openai',
                'gpt-4o-mini': 'openai',
                'o4-mini': 'openai',
                'o3-mini': 'openai',
                'gpt-4.5-preview': 'openai',
                'gpt-4.1': 'openai',
                'gpt-4.1-mini': 'openai',
                'models/gemini-2.0-flash-thinking-exp': 'google',
                'models/gemini-2.0-pro-exp': 'google',
                'models/gemini-2.0-flash': 'google',
                'models/gemini-1.5-pro-latest': 'google',
                'models/gemini-1.5-pro': 'google',
                }

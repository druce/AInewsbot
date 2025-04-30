"""Description: Constants, including configs and prompts for AInewsbot project"""
import os
import dotenv
dotenv.load_dotenv()

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

# Path to browser app
# FIREFOX_APP_PATH = '/Applications/Firefox.app'
# Path to profile
# FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/k8k0lcjj.default-release'
FIREFOX_PROFILE_PATH = '/Users/drucev/Library/Application Support/Firefox/Profiles/j6cl7lzz.playwright'
# CHROME_PROFILE_PATH = '/Users/drucev/Library/Application Support/Google/Chrome'
# CHROME_PROFILE = 'Profile 7'
# CHROME_DRIVER_PATH = '/Users/drucev/Library/Application Support/undetected_chromedriver/undetected_chromedriver'
SLEEP_TIME = 10
# NUM_BROWSERS = 4
# BROWSERS = []

SQLITE_DB = 'articles.db'

# note that token count may not be accurate for eg google, anthropic

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
# MAX_OUTPUT_TOKENS = 4096    # max in current model
TENACITY_RETRY = 5  # Maximum 5 attempts

# TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
SOURCES_EXPECTED = 16
MIN_TITLE_LEN = 28

# MAXPAGELEN = 50

HOSTNAME_SKIPLIST = ['finbold.com']
SITE_NAME_SKIPLIST = ['finbold']


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

######################################################################

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

######################################################################

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

######################################################################

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
• `relevant` = true if the summary is directly related to {topic}.
• Otherwise `relevant` = false.

"""

######################################################################

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

######################################################################

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

######################################################################

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

######################################################################

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

######################################################################

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

######################################################################

TOP_CATEGORIES_SYSTEM_PROMPT = """
# Role & Objective
You are **“The News Pulse Analyst.”**
Your task: read a daily batch of AI-related news items and surface **10-30** short, high-impact topic titles for an executive summary.
You will receive today's AI-related news items in markdown format.
Each item will have headline, URL, topics, a rating, and bullet-point summary.
Return **10-30** distinct, high-impact topics in the supplied JSON format.

# Input Format
```markdown
[Headline - URL](URL)
Topics: topic1, topic2, ...
Rating: 0-10
- Bullet 1
- Bullet 2
...
```
"""

TOP_CATEGORIES_USER_PROMPT = """

# Response Rules

- Scope: use only the supplied bullets—no external facts.
- Title length: ≤ 5 words, Title Case.
- Count: 10 ≤ topics ≤ 30; if fewer qualify, return all.
- Priority: rank by (impact × frequency); break ties by higher Rating, then alphabetical.
- Redundancy: merge or drop overlapping stories.
- Tone: concise, neutral; no extra prose.
- Privacy: never reveal chain-of-thought.
- Output: one valid JSON object matching the schema supplied (double quotes only)

Scoring Heuristics  (internal - do not output scores)
1. Repeated entity or theme
2. Major technological breakthrough
3. Significant biz deal / funding
4. Key product launch or update
5. Important benchmark or research finding
6. Major policy or regulatory action
7. Significant statement by influential figure

Reasoning Steps  (think silently)
1. Parse each item; extract entities/themes.
2. Count their recurrence.
3. Weigh impact via the heuristics.
4. Select top 10-30 non-overlapping topics.
5. Draft ≤ 5-word titles.
6. Emit a JSON object with a list of strings using the supplied schema. *(Expose only Step 6.)*

### <<<STORIES>>>
{input_text}
### <<<END>>>

Now think step by step, then output the JSON using the supplied schema.

"""

######################################################################
TOPIC_REWRITE_SYSTEM_PROMPT = """
# Role & Objective
You are **“The Topic Optimizer.”**
Goal: Polish a set of proposed technology-focused topic lines into **10-30** unique, concise, title-case entries (≤ 5 words each) and return a JSON object using the supplied schema.

# Rewrite Rules
1. **Merge Similar**: combine lines that describe the same concept or event.
2. **Split Multi-Concept**: separate any line that mixes multiple distinct ideas.
3. **Remove Fluff**: delete vague words (“new”, “innovative”, “AI” if obvious, etc.).
4. **Be Specific**: prefer concrete products, companies, events.
5. **Standardize Names**: use official product / company names.
6. **Deduplicate**: no repeated items in final list.
7. **Clarity & Brevity**: ≤ 5 words, Title Case.

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

FORMATTING:
 • Return a JSON object containing a list of strings using the provided JSON schema
 • One topic per headline
 • Use title case
"""

TOPIC_REWRITE_USER_PROMPT = """
Edit this list of technology-focused topics.

Reasoning Steps  (think silently)
1. Parse input lines.
2. Apply merge / split logic.
3. Simplify and clarify, apply style guide.
4. Finalize ≤ 5-word titles.
5. Build JSON array (unique, title-case).
6. Output exactly the JSON schema—nothing else.

### START_TOPICS
{input_text}
### END_TOPICS

Now think step by step, then output the JSON using the supplied schema.

"""

######################################################################

FINAL_SUMMARY_SYSTEM_PROMPT = """
You are “The Newsroom Chief,” an expert AI editor, who
can identify the important themes and throughlines
from large volumes of news to write a compelling daily
news summary. You will transform raw tech-news digests
into well-structured data for downstream formatting.

You will receive a list of suggested topics and a list of ~100 news items from the user.

You will select the contents for a polished daily newsletter in the supplied JSON schema.

— Think silently; never reveal chain of thought.
— Follow every instruction from the user exactly.
— Your ONLY output must be **minified JSON** that conforms to the Newsletter → Section → NewsArticle schema:
"""

FINAL_SUMMARY_USER_PROMPT = """
#############################
##  TASK: JSON NEWSLETTER  ##
#############################

Input below:
1. A **suggested topics list** (guidance only).
2. ~100 news items in this Markdown pattern:

[Title – Source](URL)
Topics: topic1, topic2, …
Rating: 0-10
• Bullet 1
• Bullet 2
…

---------------------------------------
### TASK INSTRUCTIONS
Follow the workflow below **in order**:
---------------------------------------
1. **Section discovery**
   • Create 5-10 themed sections **plus one final catch-all** section titled **"Other News"**.
   • Provided topics may be duplicative or incomplete, so generate your own topics for the most coherent grouping and narrative.

2. **Bucket assignment**
   • Place each story in exactly one section or in **"Other News"**.

3. **Filtering & deduplication**
   • Exclude items that are **not AI/tech**, are clickbait, or pure opinion.
   • For near-duplicates keep only the highest **Rating** (tie → earliest in list).

4. **Section size**
   • 2-5 stories per themed section; unlimited in "Other News" if needed.
   • Select stories that are highly rated, relevant, to make a compelling coherent section narrative.

5. **Story summarisation**
   • For every kept story write **one neutral sentence ≤ 30 words**.
   • No hype like “ground-breaking”, “magnificent”, etc.

6. **Section titles**
   • ≤ 6 words, punchy/punny, reflect the bullets.

7. **JSON rules**
   • Return JSON in **exactly** the provided schema.
   • Do **NOT** change URLs or add new keys.
   • Output must be **minified** (no line breaks, no code fences).
   • Any deviation → downstream parsing will fail.

---------------------------------------
###  SUGGESTED TOPICS
{cat_str}

---------------------------------------
###  RAW NEWS ITEMS
{bullet_str}

### RAW STORIES

"""

######################################################################

REWRITE_SYSTEM_PROMPT = """
You are “The Copy Chief,” a veteran technology-news editor with deep domain expertise in AI and emerging tech.

**Goal** – Produce a publication-ready, AI-centric newsletter in raw Markdown.

• THINK silently; never reveal chain-of-thought.
• Follow the user’s rules **exactly**
• Output only RAW Markdown or the single word “OK”.
• Markdown must begin with one line that starts with “# ” (the newsletter headline).
"""

REWRITE_USER_PROMPT = """

**Task** POLISH THIS NEWSLETTER

-------------------------------------------------
RULES  (follow in order – no exceptions)
-------------------------------------------------
1. SCOPE  – Keep only stories clearly about AI, ML, data-center hardware for AI, robotics, or adjacent policy.
   • Delete items from low-cred sites (e.g. gossip tabloids).
   • Delete items that are clickbait, purely opinion, hype, stock tips, or lack verifiable facts.

2. DEDUPLICATION
   • If ≥2 bullets describe the same event or product launch, keep ONE bullet.
   • Merge extra hyperlinks into that bullet, comma-separated.
   • Never repeat a URL within a section or between sections.

3. BREVITY & TONE
   • Each bullet = ONE neutral factual sentence as short as possible (≤ 25 words).
   • No filler phrases (“The article states…”, “According to…”).
   • No superlatives: amazing, huge, groundbreaking, etc.

4. SECTION TITLES
   • Rewrite to be punchy, witty, **≤ 6 words**, and allude to the content. Try to make them funny, alliterative and punny.
   • *Examples*: “Chip Flip & Fab”, “Bot Battles”, “Regulation Rumble”.
   • Delete any section left empty.

5. NEWSLETTER HEADLINE
   • Write one line starting with “# ” that cleverly captures the day’s overarching AI themes (≤ 12 words).
   • Do **NOT** recycle a section title.

6. FORMATTING
   • Structure:
     ```
     # Daily Headline

     ## Section Title
     - Bullet 1 [Source](URL)
     - Bullet 2 [Source](URL)
     ```
   • Raw Markdown only—no code fences, no explanatory text.

7. FINAL CHECK
   • Must contain 5–8 sections (after deletions).
   • No bullet may exceed 25 words.
   • Every bullet has at least one clickable link.
   • Newsletter starts with “# ”, ends with a newline.

-------------------------------------------------
**Newsletter to edit ↓**

{summary}
"""

######################################################################

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

######################################################################

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

######################################################################

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

######################################################################

# NEWSCATCHER_SOURCES = ['247wallst.com',
#                        '9to5mac.com',
#                        'androidauthority.com',
#                        'androidcentral.com',
#                        'androidheadlines.com',
#                        'appleinsider.com',
#                        'benzinga.com',
#                        'cnet.com',
#                        'cnn.com',
#                        'digitaltrends.com',
#                        'engadget.com',
#                        'fastcompany.com',
#                        'finextra.com',
#                        'fintechnews.sg',
#                        'fonearena.com',
#                        'ft.com',
#                        'gadgets360.com',
#                        'geekwire.com',
#                        'gizchina.com',
#                        'gizmochina.com',
#                        'gizmodo.com',
#                        'gsmarena.com',
#                        'hackernoon.com',
#                        'howtogeek.com',
#                        'ibtimes.co.uk',
#                        'itwire.com',
#                        'lifehacker.com',
#                        'macrumors.com',
#                        'mashable.com',
#                        #  'medium.com',
#                        'mobileworldlive.com',
#                        'msn.com',
#                        'nypost.com',
#                        'phonearena.com',
#                        'phys.org',
#                        'popsci.com',
#                        'scmp.com',
#                        'sify.com',
#                        'siliconangle.com',
#                        'siliconera.com',
#                        'siliconrepublic.com',
#                        'slashdot.org',
#                        'slashgear.com',
#                        'statnews.com',
#                        'tech.co',
#                        'techcrunch.com',
#                        'techdirt.com',
#                        'technode.com',
#                        'technologyreview.com',
#                        'techopedia.com',
#                        'techradar.com',
#                        'techraptor.net',
#                        'techtimes.com',
#                        'techxplore.com',
#                        'telecomtalk.info',
#                        'thecut.com',
#                        'thedrum.com',
#                        'thehill.com',
#                        'theregister.com',
#                        'theverge.com',
#                        'thurrott.com',
#                        'tipranks.com',
#                        'tweaktown.com',
#                        'videocardz.com',
#                        'washingtonpost.com',
#                        'wccftech.com',
#                        'wired.com',
#                        'xda-developers.com',
#                        'yahoo.com',
#                        'zdnet.com']

######################################################################

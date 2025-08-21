"""Description: Constants, including configs and prompts for AInewsbot project"""
import os
import dotenv

dotenv.load_dotenv()

REQUEST_TIMEOUT = 600

DOWNLOAD_ROOT = "download"
DOWNLOAD_DIR = os.path.join(DOWNLOAD_ROOT, "sources")
PAGES_DIR = os.path.join(DOWNLOAD_ROOT, 'html')
TEXT_DIR = os.path.join(DOWNLOAD_ROOT, 'text')
SCREENSHOT_DIR = os.path.join(DOWNLOAD_ROOT, 'screenshots')

DATA_ROOT = "data"
CHROMA_DB_DIR = os.path.join(DATA_ROOT, "chromadb")
CHROMA_DB_NAME = "chroma_articles"
CHROMA_DB_PATH = os.path.join(CHROMA_DB_DIR, CHROMA_DB_NAME)
CHROMA_DB_COLLECTION = "articles"
CHROMA_DB_EMBEDDING_FUNCTION = "text-embedding-3-large"

OUTPUT_DIR = "out"
# 10% similarity
# I am storing metadata in the doc, so titles, keywords might change, need a higher threshold
COSINE_DISTANCE_THRESHOLD = .1
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

FIREFOX_PROFILE_PATH = os.getenv('FIREFOX_PROFILE_PATH')
if not FIREFOX_PROFILE_PATH:
      raise ValueError(
          "Firefox profile not found. Please:\n"
          "1. Install Firefox, and\n"
          "2. Set FIREFOX_PROFILE_PATH in .env file"
      )

if not os.path.exists(FIREFOX_PROFILE_PATH):
      raise ValueError(
          f"Firefox profile {FIREFOX_PROFILE_PATH} not found. Please:\n"
          "1. Install Firefox, and\n"
          "2. Set FIREFOX_PROFILE_PATH in .env file"
      )

# CHROME_PROFILE_PATH = '/Users/drucev/Library/Application Support/Google/Chrome'
# CHROME_PROFILE = 'Profile 7'
# CHROME_DRIVER_PATH = '/Users/drucev/Library/Application Support/undetected_chromedriver/undetected_chromedriver'
SLEEP_TIME = 10
# NUM_BROWSERS = 4
# BROWSERS = []

SQLITE_DB = os.path.join(DATA_ROOT, 'articles.db')

# note that token count may not be accurate for eg google, anthropic

MAX_INPUT_TOKENS = 8192     # includes text of all headlines
# MAX_OUTPUT_TOKENS = 4096    # max in current model
TENACITY_RETRY = 5  # Maximum 5 attempts

# TEMPERATURE = 0

SOURCECONFIG = "sources.yaml"
SOURCES_EXPECTED = 17
MIN_TITLE_LEN = 28
MINIMUM_STORY_RATING = 1
MAX_ARTICLES = 100
# MAXPAGELEN = 50

DOMAIN_SKIPLIST = ['finbold.com', 'philarchive.org']
SITE_NAME_SKIPLIST = ['finbold', 'philarchive.org']

MODEL_FAMILY = {'gpt-4o-2024-11-20': 'openai',
                'gpt-4o-mini': 'openai',
                'o4-mini': 'openai',
                'o3-mini': 'openai',
                'o3': 'openai',
                'gpt-4.5-preview': 'openai',
                'gpt-4.1': 'openai',
                'gpt-4.1-mini': 'openai',
                'gpt-5-nano': 'openai',
                'gpt-5-mini': 'openai',
                'gpt-5': 'openai',
                'models/gemini-2.0-flash-thinking-exp': 'google',
                'models/gemini-2.0-pro-exp': 'google',
                'models/gemini-2.0-flash': 'google',
                'models/gemini-1.5-pro-latest': 'google',
                'models/gemini-1.5-pro': 'google',
                'claude-sonnet-4-20250514': 'anthropic',
                'claude-sonnet-4': 'anthropic',
                'claude-opus-4-20250514': 'anthropic',
                'claude-opus-4': 'anthropic',
                'claude-3-5-haiku': 'anthropic',
                }

CANONICAL_TOPICS = [
    "Policy And Regulation",
    "Economics",
    "Governance",
    "Safety And Alignment",
    "Bias And Fairness",
    "Privacy And Surveillance",
    "Inequality",
    "Job Automation",
    'Disinformation',
    'Deepfakes',
    'Sustainability',

    "Agents",
    "Coding Assistants",

    "Virtual Assistants",
    "Chatbots",
    "Robots",
    "Autonomous Vehicles",
    "Drones",
    'Virtual And Augmented Reality',

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
    'Speech Recognition And Synthesis',

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
    "Mergers and Acquisitions",
    "Deals",
    "IPOs",
    "Ethics",
    "Legal Issues",
    "Cybersecurity",
    "AI Doom",
    'Stocks',
    'Bubble',
    'Cryptocurrency',
    'Climate',
    'Energy',
    'Nuclear',
    'Scams',
    'Privacy',
    'Intellectual Property',
    'Customer Service',
    'Military',
    'Agriculture',
    'Testing',
    'Authors And Writing',
    'Books And Publishing',
    'TV And Film And Movies',
    'Streaming',
    'Hollywood',
    'Music',
    'Art And Design',
    'Fashion',
    'Food And Drink',
    'Travel',
    'Health And Fitness',
    'Sports',
    'Gaming',
    # 'Science',
    'Politics',
    'Finance',
    'History',
    'Society And Culture',
    'Lifestyle And Travel',
    'Jobs And Careers',
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
# arbitrary, could start with e.g. Alexa rank, Google PageRank, Cloudflare Radar rank, SimilarWeb traffic, SEMrush, Ahrefs, Moz, Majestic, etc.
# don't really need 0s, can just .get() and default to 0
SOURCE_REPUTATION = {
    'redd.it': 0,
    'aitoolsclub.com': 0,
    'analyticsindiamag.com': 0,
    'amazon.com': 0,
    'biztoc.com': 0,
    'blog.google': 0,
    'indiatimes.com': 0,
    'finbold.com': 0,
    'flip.it': 0,
    'flipboard.com': 0,
    'github.com': 0,
    'greekreporter.com': 0,
    'medium.com': 0,
    'neurosciencenews.com': 0,
    'restofworld.org': 0,
    't.co': 0,
    'tech.co': 0,
    'slashdot.org': 0,
    'uxdesign.cc': 0,
    'androidauthority.com': 0,
    'androidcentral.com': 0,
    'androidheadlines.com': 0,
    'androidpolice.com': 0,
    'benzinga.com': 0,
    'channelnewsasia.com': 0,
    'christopherspenn.com': 0,
    'ciodive.com': 0,
    'computerweekly.com': 0,
    'creativebloq.com': 0,
    'digitalcameraworld.com': 0,
    'digitaljournal.com': 0,
    'digitaltrends.com': 0,
    'entrepreneur.com': 0,
    'etfdailynews.com': 0,
    'euronews.com': 0,
    'finextra.com': 0,
    'fool.com': 0,
    'foxnews.com': 0,
    'globenewswire.com': 0,
    'investing.com': 0,
    'itpro.com': 0,
    'jpost.com': 0,
    'laptopmag.com': 0,
    'livemint.com': 0,
    'livescience.com': 0,
    'miamiherald.com': 0,
    'mobileworldlive.com': 0,
    'msn.com': 0,
    'newsmax.com': 0,
    'reddit.com': 0,
    'sciencedaily.com': 0,
    'statnews.com': 0,
    'techdirt.com': 0,
    'techmonitor.ai': 0,
    'techtimes.com': 0,
    'the-sun.com': 0,
    'thebrighterside.news': 0,
    'thedrum.com': 0,
    'tipranks.com': 0,
    'trendhunter.com': 0,
    'uniladtech.com': 0,
    '247wallst.com': 1,
    '9to5google.com': 1,
    '9to5mac.com': 1,
    'go.com': 1,
    'apnews.com': 1,
    'appleinsider.com': 1,
    'arxiv.org': 1,
    'bgr.com': 1,
    'nvidia.com': 1,
    'decrypt.co': 1,
    'digiday.com': 1,
    'fortune.com': 1,
    'lifehacker.com': 1,
    'machinelearningmastery.com': 1,
    'mashable.com': 1,
    'newatlas.com': 1,
    'nypost.com': 1,
    'observer.com': 1,
    'petapixel.com': 1,
    'phys.org': 1,
    'qz.com': 1,
    'readwrite.com': 1,
    'ieee.org': 1,
    'techxplore.com': 1,
    'theconversation.com': 1,
    'thehill.com': 1,
    'thenextweb.com': 1,
    'time.com': 1,
    'towardsdatascience.com': 1,
    'twitter.com': 1,
    'variety.com': 1,
    'venturebeat.com': 1,
    '404media.co': 1,
    'adweek.com': 1,
    'axios.com': 1,
    'barrons.com': 1,
    'bbc.com': 1,
    'cbsnews.com': 1,
    'cbssports.com': 1,
    'cnbc.com': 1,
    'cnn.com': 1,
    'extremetech.com': 1,
    'forbes.com': 1,
    'gadgets360.com': 1,
    'geekwire.com': 1,
    'geeky-gadgets.com': 1,
    'inc.com': 1,
    'macrumors.com': 1,
    'macworld.com': 1,
    'makeuseof.com': 1,
    'marktechpost.com': 1,
    'medianama.com': 1,
    'nbcnews.com': 1,
    'newsweek.com': 1,
    'nextbigfuture.com': 1,
    'npr.org': 1,
    'pcgamer.com': 1,
    'pcmag.com': 1,
    'pcworld.com': 1,
    'popsci.com': 1,
    'psychologytoday.com': 1,
    'pymnts.com': 1,
    'scmp.com': 1,
    'semafor.com': 1,
    'techinasia.com': 1,
    'techradar.com': 1,
    'techrepublic.com': 1,
    'techspot.com': 1,
    'theglobeandmail.com': 1,
    'theguardian.com': 1,
    'usatoday.com': 1,
    'windowscentral.com': 1,
    'businessinsider.com': 2,
    'businessinsider.in': 2,
    'techmeme.com': 2,
    'wapo.st': 2,
    'yahoo.com': 2,
    'acm.org': 2,
    'financialpost.com': 2,
    'futurism.com': 2,
    'gizmodo.com': 2,
    'hackernoon.com': 2,
    'openai.com': 2,
    'siliconangle.com': 2,
    'simonwillison.net': 2,
    'techcrunch.com': 2,
    'cnet.com': 2,
    'engadget.com': 2,
    'fastcompany.com': 2,
    'nature.com': 2,
    'newscientist.com': 2,
    'reuters.com': 2,
    'technologyreview.com': 2,
    'theatlantic.com': 2,
    'theinformation.com': 2,
    'tomsguide.com': 2,
    'tomshardware.com': 2,
    'washingtonpost.com': 2,
    'wired.com': 2,
    'zdnet.com': 2,
    'newyorker.com': 2,
    'theverge.com': 3,
    'theregister.com': 3,
    'arstechnica.com': 3,
    'bloomberglaw.com': 4,
    'bloomberg.com': 4,
    'bloombergtax.com': 4,
    'bnnbloomberg.ca': 4,
    'ft.com': 4,
    'nytimes.com': 4,
    'wsj.com': 4,
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

"""
Web scraping utilities.

This module contains functions used for web scraping from web sites in sources.yaml and individual news stories.
"""
import asyncio
import re
import os
from urllib.parse import urljoin, urlparse
import pdb
import json

import random
import time
import datetime
from dateutil import parser as date_parser
from pathlib import Path
import tiktoken

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

import trafilatura

from .utilities import log
from .config import (DOWNLOAD_DIR, PAGES_DIR, FIREFOX_PROFILE_PATH,  # SCREENSHOT_DIR,
                     MIN_TITLE_LEN, SLEEP_TIME, TEXT_DIR, MAX_INPUT_TOKENS)

# Global state for per-domain rate limiting
_domain_locks = {}
_domain_last_access = {}
_RATE_LIMIT_SECONDS = 15


def get_og_tags(url):
    """
    Fetches Open Graph og: tags from the HEAD of a given URL and returns them as a dictionary.

    Parameters:
    url (str): The URL of the webpage to fetch the og: tags from.

    Returns:
    dict: A dictionary containing the og: tags found in the HEAD of the webpage. The keys are the property names
          of the og: tags and the values are the corresponding content values.

    Raises:
    requests.RequestException: If there is an error fetching the webpage.

    """
    result_dict = {}
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            head = soup.head
            if head:
                og_tags = head.find_all(
                    property=lambda prop: prop and prop.startswith("og:")
                )
                for tag in og_tags:
                    if "content" in tag.attrs:
                        result_dict[tag["property"]] = tag["content"]

                page_title = ""
                title_tag = soup.find("title")
                if title_tag:
                    page_title = title_tag.text
                    if page_title:
                        result_dict["title"] = page_title
        return result_dict
    except requests.RequestException as e:
        log(f"Error fetching {url}: {e}")
    return result_dict


def get_path_from_url(url):
    """
    Extracts the path following the top-level domain name from a URL.

    :param url: The URL string.
    :return: The path component of the URL.
    """
    parsed_url = urlparse(url)
    return parsed_url.path


def trimmed_href(link):
    """
    Trims everything in the link after a question mark such as a session ID.

    :param link: The input string or bs4 link.
    :return: The trimmed string.
    """
    # Find the position of the question mark
    if isinstance(link, str):
        s = link
    else:
        s = link.get("href")
    if s:
        question_mark_index = s.find("?")

        # If a question mark is found, trim the string up to that point
        if question_mark_index != -1:
            return s[:question_mark_index]
        else:
            # Return the original string if no question mark is found
            return s
    else:
        return s


def sanitize_filename(filename):
    """
    Sanitizes a filename by removing unsafe characters and ensuring it is valid.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    sep = ""
    datestr = ""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove any other unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove leading or trailing underscores
    filename = filename.strip('_')
    # filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)
    trunc_len = 255 - len(datestr) - len(sep) - len(".html") - 1
    filename = filename[:trunc_len]
    return filename


def trunc_tokens(long_prompt, model='gpt-4o', maxtokens=MAX_INPUT_TOKENS):
    """return prompt string, truncated to maxtokens"""
    # Initialize the encoding for the model you are using, e.g., 'gpt-4'
    encoding = tiktoken.encoding_for_model(model)

    # Encode the prompt into tokens, truncate, and return decoded prompt
    tokens = encoding.encode(long_prompt)
    tokens = tokens[:maxtokens]
    truncated_prompt = encoding.decode(tokens)

    return truncated_prompt


def normalize_html(path: Path | str) -> str:
    """
    Clean and extract text content from an HTML file, including titles and social media metadata.

    Args:
        path (Path | str): Path to the HTML file to process

    Returns:
        - str: Extracted and cleaned text content, or empty string if processing fails

    The function extracts:
        - Page title from <title> tag
        - Social media titles from OpenGraph and Twitter meta tags
        - Social media descriptions from OpenGraph and Twitter meta tags
        - Main content using trafilatura library

    All extracted content is concatenated and truncated to MAX_INPUT_TOKENS length.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except Exception as exc:
        log(f"Error: {str(exc)}")
        log(f"Skipping {path}")
        return ""

    # Parse the HTML content using trafilatura
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        # Try to get the title from the <title> tag
        title_tag = soup.find("title")
        title_str = "Page title: " + title_tag.string.strip() + \
            "\n" if title_tag and title_tag.string else ""
    except Exception as exc:
        title_str = ""
        log(str(exc), "clean_html page_title")

    try:
        # Try to get the title from the Open Graph meta tag
        og_title_tag = soup.find("meta", property="og:title")
        if not og_title_tag:
            og_title_tag = soup.find(
                "meta", attrs={"name": "twitter:title"})
        og_title = og_title_tag["content"].strip(
        ) + "\n" if og_title_tag and og_title_tag.get("content") else ""
        og_title = "Social card title: " + og_title if og_title else ""
    except Exception as exc:
        og_title = ""
        log(str(exc), "clean_html og_title")

    try:
        # get summary from social media cards
        og_desc_tag = soup.find("meta", property="og:description")
        if not og_desc_tag:
            # Extract the Twitter description
            og_desc_tag = soup.find(
                "meta", attrs={"name": "twitter:description"})
        og_desc = og_desc_tag.get("content").strip() + \
            "\n" if og_desc_tag else ""
        og_desc = 'Social card description: ' + og_desc if og_desc else ""
    except Exception as exc:
        og_desc = ""
        log(str(exc), "clean_html og_desc")

    # Get text and strip leading/trailing whitespace
    log(title_str + og_title + og_desc, "clean_html")
    try:
        plaintext = trafilatura.extract(html_content)
        plaintext = plaintext.strip() if plaintext else ""
    except Exception as exc:
        plaintext = html_content
        log(str(exc), "clean_html trafilatura")

    # remove special tokens, have found in artiles about tokenization
    # All OpenAI special tokens follow the pattern <|something|>
    special_token_re = re.compile(r"<\|\w+\|>")
    plaintext = special_token_re.sub("", plaintext)
    visible_text = title_str + og_title + og_desc + plaintext
    visible_text = trunc_tokens(
        visible_text, model='gpt-4o', maxtokens=MAX_INPUT_TOKENS)
    return visible_text


async def get_browser(p):
    """
    Initializes a Playwright browser instance with stealth settings.

    Args:
        p (async_playwright.Playwright): The Playwright instance.

    Returns:
        Browser: The initialized browser instance.
    """
    viewport = random.choice([
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720}
    ])

    # random device-scale-factor for additional randomization
    device_scale_factor = random.choice([1, 1.25, 1.5, 1.75, 2])

    # Random color scheme and timezone
    color_scheme = random.choice(['light', 'dark', 'no-preference'])
    timezone_id = random.choice([
        'America/New_York', 'Europe/London', 'Europe/Paris',
        'Asia/Tokyo', 'Australia/Sydney', 'America/Los_Angeles'
    ])
    locale = random.choice([
        'en-US', 'en-GB'
    ])
    extra_http_headers = {
        "Accept-Language": f"{locale.split('-')[0]},{locale};q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1" if random.choice([True, False]) else "0"
    }

    b = await p.firefox.launch_persistent_context(
        user_data_dir=FIREFOX_PROFILE_PATH,
        headless=True,  # run headless, hide splash window
        viewport=viewport,
        device_scale_factor=device_scale_factor,
        timezone_id=timezone_id,
        color_scheme=color_scheme,
        extra_http_headers=extra_http_headers,
        # removes Playwright’s default flag
        ignore_default_args=["--enable-automation"],
        args=[
            # "--disable-blink-features=AutomationControlled",  # Chrome/Blink flag analogue
            "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"
        ],
        # provide a valid realistic User-Agent string for the latest Firefox on Apple Silicon
        # match OS / browser build
        user_agent="Mozilla/5.0 (Macintosh; ARM Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
        accept_downloads=True,
    )
    await apply_stealth_script(b)
    return b


async def apply_stealth_script(context):
    """Apply stealth settings to a new page using playwright_stealth."""
    page = await context.new_page()
    await stealth_async(page)
    await page.close()


async def perform_human_like_actions(page):
    """Perform random human-like actions on the page to mimic real user behavior."""
    # Random mouse movements
    for _ in range(random.randint(3, 8)):
        # Move mouse with multiple steps to simulate human-like movement
        x = random.randint(100, 1200)
        y = random.randint(100, 700)
        steps = random.randint(5, 10)

        # Get current mouse position
        mouse_position = await page.evaluate("""() => {
            return {x: 0, y: 0}; // Default starting position
        }""")

        current_x = mouse_position.get('x', 0)
        current_y = mouse_position.get('y', 0)

        # Calculate increments for smooth movement
        for step in range(1, steps + 1):
            next_x = current_x + (x - current_x) * step / steps
            next_y = current_y + (y - current_y) * step / steps

            # Add slight randomness to path
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-5, 5)

            await page.mouse.move(next_x + jitter_x, next_y + jitter_y)
            await asyncio.sleep(random.uniform(0.01, 0.05))

    # Random scrolling behavior
    scroll_amount = random.randint(300, 700)
    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
    await asyncio.sleep(random.uniform(0.5, 2))

    # Sometimes scroll back up a bit
    if random.random() > 0.7:
        await page.evaluate(f"window.scrollBy(0, -{random.randint(100, 300)})")
        await asyncio.sleep(random.uniform(0.3, 1))

    return page


async def worker(queue, browser, results):
    """Worker function for asynchronous url processing
    fetch URLs from queue until empty,
    calling fetch_url using browser, append to results.
    multiple asynchronous workers are appending to 1 results array
    which I guess is OK but maybe each should just return results"""

    log("Launching worker")
    # for now, skip these domains since I don't want to log in and potentially get my account blocked
    ignore_list = ["www.bloomberg.com", "bloomberg.com",
                   "cnn.com", "www.cnn.com",
                   "wsj.com", "www.wsj.com"]

    while True:
        try:
            idx, url, title = await queue.get()
            log(f"from queue: {idx}, {url} , {title}")
        except asyncio.QueueEmpty:
            return
        # skip urls from domains in ignore_list, just return empty path
        if urlparse(url).hostname in ignore_list:
            log(f"Skipping fetch for {idx} {url} {title}")
            results.append((idx, url, title, ""))
            queue.task_done()
        else:
            try:
                html_path, last_updated, final_url = await fetch_url(url, title, browser, destination=PAGES_DIR)
                results.append(
                    (idx, final_url, title, html_path, last_updated))
            except Exception as exc:
                log(f"Error fetching {url}: {exc}")
            finally:
                queue.task_done()


async def fetch_queue(queue, concurrency):
    """
    Processes a queue of URLs concurrently using a specified number of workers.

    Args:
        queue (asyncio.Queue): The queue containing tuples of (index, url, title) to process.
        concurrency (int): The number of concurrent workers to use.

    Returns:
        list: A list of tuples containing (index, url, title, result) for each processed URL.
    """

    results = []
    async with async_playwright() as p:
        log("Launching browser")
        browser = await get_browser(p)

        log("Launching workers")
        tasks = [asyncio.create_task(worker(queue, browser, results))
                 for _ in range(concurrency)]
        await queue.join()
        log("Finishing and closing browser")
        for t in tasks:
            t.cancel()
        await browser.close()

    return results

# potentially switch to chromium, didn't do in the past due to chromedriver version issues but not an issue with playwright
# 1. test running a chrome.py with playwright and playwright-stealth and chromium, make a new profile, figure out what it uses
# 2. headless, add stealth options, log in to eg feedly using the profile, see that it works, where profile is
# 3. get your organic chrome user agent string and paste
# 4. migrate existing profile settings and test. now we have a good profile.
# 5. update get_browser below to use chrome and new profile. potentially ask o3 to look at your code and suggest a good stealth calling template.


async def fetch_url(url, title, browser_context=None, click_xpath=None, scrolls=0, initial_sleep=SLEEP_TIME, destination=DOWNLOAD_DIR):
    """
    Fetches a URL using a Playwright browser context.

    Args:
        url (str): The URL to fetch.
        title (str): The title for the fetched page.
        click_xpath (str): An optional XPath expression to click on before saving.
        scrolls (int): The number of times to scroll to the bottom of the page and wait for new content to load.
        browser_context (BrowserContext): The Playwright browser context to use. If not provided, a new browser context will be initialized.
        initial_sleep (float): The number of seconds to wait after the page has loaded before clicking.

    Returns:
        tuple: (html_path, last_updated_time, final_url) where:
            html_path (str): The path to the downloaded file.
            last_updated_time (str or None): The last update time of the page.
            final_url (str): The final URL after any redirects.

    # should add retry functionality, re-enable screenshots
    """
    log(f"fetch_url({url})")
    try:
        # make output directories
        if not os.path.exists(destination):
            os.makedirs(destination)

        title = sanitize_filename(title)
        html_path = os.path.join(destination, f'{title}.html')
        # check if file already exists, don't re-download
        if os.path.exists(html_path):
            log(f"File already exists: {html_path}")
            return html_path, None, url

        # if file does not exist, download
        # rate limit per domain
        domain = urlparse(url).netloc   # get domain name
        # Get or create a lock for this domain
        lock = _domain_locks.setdefault(domain, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = _domain_last_access.get(domain, 0)
            wait = _RATE_LIMIT_SECONDS - (now - last) + random.uniform(0, 20)
            if wait > 0:
                log(f"Waiting {wait} seconds to rate limit {domain} {now - last}")
                await asyncio.sleep(wait)
            # Update last access time
            _domain_last_access[domain] = time.monotonic()

        page = await browser_context.new_page()
        response = await page.goto(url, timeout=60000)
        await asyncio.sleep(initial_sleep+random.uniform(2, 5))
        await perform_human_like_actions(page)
        if click_xpath:
            await asyncio.sleep(initial_sleep+random.uniform(2, 5))
            log(f"Attempting to click on {click_xpath}")
            # click_xpath == '//*[@aria-label="Artificial intelligence"]'
            await page.wait_for_selector(f'xpath={click_xpath}')
            await page.click(f'xpath={click_xpath}')
        for i in range(scrolls):
            log(f"Scrolling {title} ({i+1}/{scrolls})")
            await asyncio.sleep(random.uniform(2, 5))  # Stealth delay
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
            # kludge for Feedly
            # should pass a div name like #feedlyFrame
            scrolldiv = "#feedlyFrame"
            await page.evaluate("""
                const el = document.querySelector('%s');
                if (el) {
                    el.scrollTop = el.scrollHeight;
                } else {
                    window.scrollTo(0, document.body.scrollHeight);
                }
            """ % scrolldiv)

        html_source = await page.content()
        if page.url != url:
            log(f"Page URL redirected from {url} to {page.url}")
        # break if page.url domain is google.com
        if urlparse(url).netloc == "news.google.com":
            log(f"Google News page: {page.url}")
        # Determine last updated time, first try meta tags
        last_updated = None
        soup_meta = BeautifulSoup(html_source, "html.parser")
        meta_selectors = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("property", "article:modified_time"),
            ("property", "og:updated_time"),
            ("name", "pubdate"),
            ("name", "publish_date"),
            ("name", "Last-Modified"),
            ("name", "lastmod"),
        ]
        for attr, val in meta_selectors:
            tag = soup_meta.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                last_updated = tag["content"]
                log(
                    f"Found last updated time from meta tag {attr}={val}: {last_updated}")
                break

        # if not last_updated:
        #     time_tag = soup_meta.find("time", datetime=True)
        #     if time_tag and time_tag.get("datetime"):
        #         last_updated = time_tag["datetime"]
        #         log(f"Found last updated time from time tag: {last_updated}")

        # for substack
        # Find all JSON-LD script blocks
        for script in soup_meta.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'NewsArticle':
                    last_updated = data.get('datePublished')
                    log(
                        f"Found script last updated time from script datePublished: {last_updated}")
                    break
            except Exception as e:
                continue

        # Check HTTP Last-Modified header
        if not last_updated:
            if response and response.headers.get("last-modified"):
                last_updated = response.headers.get("last-modified")
                log(
                    f"Found last updated time from HTTP header: {last_updated}")

        # Fallback to document.lastModified
        if not last_updated:
            try:
                last_updated = await page.evaluate("document.lastModified")
                log(
                    f"Found last updated time from document.lastModified: {last_updated}")
            except:
                last_updated = None

        # Validate and normalize last_updated to Zulu datetime
        if last_updated and isinstance(last_updated, str):
            try:
                dt = date_parser.parse(last_updated)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                dt_utc = dt.astimezone(datetime.timezone.utc)
                last_updated = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception as e:
                log(f"Could not parse last_updated '{last_updated}': {e}")
                # set to 1 day ago
                last_updated = (datetime.datetime.now(
                    datetime.timezone.utc) - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save HTML
        log(f"Saving HTML to {html_path}")
        # if the file already exists, delete it
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_source)

        # Save screenshot for video
        # screenshot_path = f"{SCREENSHOT_DIR}/{title}.png"
        # await page.screenshot(path=screenshot_path)
        # Get the final URL after any redirects
        # Try to get canonical URL from the HTML source
        canonical_tag = soup_meta.find("link", rel="canonical")
        if canonical_tag and canonical_tag.get("href"):
            final_url = canonical_tag["href"]
        else:
            final_url = page.url

        await page.close()

        return html_path, last_updated, final_url
    except Exception as exc:
        log(f"Error fetching {url}: {exc}")
        return None, None, None


async def fetch_source(source_dict, browser_context=None):
    """
    Fetches a landing page using fetch_url and parameters defined in sources.yaml.
    source_dict is the landing page parameters loaded from sources.yaml.
    Updates source_dict['latest'] with the path to the downloaded file.

    Args:
        source_dict (dict): A dictionary containing the parameters defined in sources.yaml.
        browser_context (BrowserContext, optional): The Playwright browser context to use. If not provided, a new browser context will be initialized.

    Returns:
        str: The path to the downloaded file.

    Raises:
        Exception: If there is an error during the execution of the function.

    """
    url = source_dict.get("url")
    title = source_dict["title"]
    sourcename = source_dict["sourcename"]
    click_xpath = source_dict.get("click", "")
    scrolls = source_dict.get("scroll", 0)
    initial_sleep = source_dict.get("initial_sleep", SLEEP_TIME)

    log(f"Starting fetch_source {url}, {title}")

    # Open the page and fetch the HTML
    file_path, _, _ = await fetch_url(url, title, browser_context,
                                      click_xpath, scrolls, initial_sleep)
    source_dict['latest'] = file_path
    return (sourcename, file_path)


async def fetch_source_queue(queue, concurrency):
    """
    Processes a queue of sources concurrently using a specified number of workers.

    Args:
        queue (asyncio.Queue): The queue containing tuples of (index, url, title) to process.
        concurrency (int): The number of concurrent workers to use.

    Returns:
        list: A list of tuples containing (index, url, title, result) for each processed URL.
    """
    async with async_playwright() as p:

        browser = await p.firefox.launch_persistent_context(
            user_data_dir=FIREFOX_PROFILE_PATH,
            headless=True,  # run headless, hide splash window
            viewport={"width": 1366, "height": 768},
            # removes Playwright’s default flag
            ignore_default_args=["--enable-automation"],
            args=[
                # "--disable-blink-features=AutomationControlled",  # Chrome/Blink flag analogue
                "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"
            ],
            # match OS / browser build
            # provide a valid realistic User-Agent string for the latest Firefox on Apple Silicon
            user_agent="Mozilla/5.0 (Macintosh; ARM Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
            accept_downloads=True,
        )
        sem = asyncio.Semaphore(concurrency)

        async def bounded(source):
            async with sem:
                result = await fetch_source(source, browser)
                return result

        source_array = []
        while not queue.empty():
            source = await queue.get()
            source_array.append(source)

        results = await asyncio.gather(*[bounded(source) for source in source_array])
        await browser.close()
    return results


def parse_file(source_dict):
    """
    Parse a saved HTML file and return a list of dictionaries with title, url, src for each link in the file.

    Args:
        source_dict (dict): A dictionary containing the source information.

    Returns:
        list: A list of dictionaries, where each dictionary represents a link in the HTML file.
              Each dictionary contains the following keys: 'title', 'url', 'src'.

    Raises:
        None

    """
    sourcename = source_dict['sourcename']
    title = source_dict["title"]
    filename = source_dict["latest"]
    url = source_dict.get("url")
    exclude = source_dict.get("exclude")
    include = source_dict.get("include")
    minlength = source_dict.get("minlength", MIN_TITLE_LEN)

    link_list = []
    # get contents
    with open(filename, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags
    if soup:
        links = soup.find_all("a")
    else:
        log(f"Skipping {url}, unable to parse", "parse_file")
        return
    log(f"found {len(links)} raw links", "parse_file")

    # drop empty text
    links = [link for link in links if link.get_text(strip=True)]
    # drop some ArsTechnica links that are just the number of comments and dupe the primary link
    links = [link for link in links if not re.match(
        "^(\d+)$", link.get_text(strip=True))]

    # convert relative links to absolute links using base URL if present
    base_tag = soup.find('base')
    base_url = base_tag.get('href') if base_tag else url
    for link in links:
        link["href"] = urljoin(base_url, link.get('href', ""))

    # drop empty url path, i.e. url = toplevel domain
    links = [link for link in links if len(
        get_path_from_url(trimmed_href(link))) > 1]
    # drop anything that is not http, like javascript: or mailto:
    links = [link for link in links if link.get(
        "href") and link.get("href").startswith("http")]

    if exclude:
        for pattern in exclude:
            # filter links by exclusion pattern
            links = [
                link
                for link in links
                if link.get("href") is not None and not re.match(pattern, link.get("href"))
            ]

    if include:
        for pattern in include:
            new_links = []
            for link in links:
                href = link.get("href")
                if href and re.match(pattern, href):
                    new_links.append(link)
            links = new_links

    for link in links:
        url = trimmed_href(link)
        title = link.get_text(strip=True)
        if title == "LINK":
            # try to update title
            og_dict = get_og_tags(url)
            if og_dict.get("og:title"):
                title = og_dict.get("og:title")

        # skip some low quality links that don't have full headline, like link to a Twitter or Threads account
        if len(title) <= minlength and title != "LINK":
            continue

        link_list.append({"title": title, "url": url, "src": sourcename})

    log(f"found {len(link_list)} filtered links", "parse_file")

    return link_list


# map google news headlines to redirect
# google would never show the real url, so we have to follow redirects
# but then google eventually hard blocked scraping so this is not used
# def get_google_news_redirects(orig_df):
#     redirect_dict = {}
#     for row in orig_df.itertuples():
#         parsed_url = urlparse(row.url)
#         netloc = parsed_url.netloc
#         if netloc == 'news.google.com':
#             log_str = netloc + " -> "
#             response = requests.get(row.url, allow_redirects=False, timeout=10)
#             # The URL to which it would have redirected
#             redirect_url = response.headers.get('Location')
#             redirect_dict[row.url] = redirect_url
#             parsed_url2 = urlparse(redirect_url)
#             netloc2 = parsed_url2.netloc
#             if netloc2 == 'news.google.com':
#                 #                 logstr += netloc2 + " -> "
#                 response = requests.get(redirect_url, allow_redirects=False)
#             # The URL to which it would have redirected
#                 redirect_url = response.headers.get('Location')
#                 if redirect_url:
#                     redirect_dict[row.url] = redirect_url
#                     log_str += redirect_url
#             log(log_str, "get_google_news_redirects")

#     orig_df['actual_url'] = orig_df['url'].apply(
#         lambda url: redirect_dict.get(url, url))

#     return orig_df

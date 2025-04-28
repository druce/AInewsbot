"""
Web scraping utilities.

This module contains functions used for web scraping from web sites in sources.yaml and individual news stories.
"""
import re
import os
from urllib.parse import urljoin, urlparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from playwright.async_api import async_playwright

from bs4 import BeautifulSoup
import requests

from ainb_const import (DOWNLOAD_DIR, PAGES_DIR, SCREENSHOT_DIR, FIREFOX_PROFILE_PATH,
                        MIN_TITLE_LEN, SLEEP_TIME)
from ainb_utilities import log
import asyncio


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
        response = requests.get(url)
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
        print(f"Error fetching {url}: {e}")
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


# async def get_browser_context(firefox_profile_path=FIREFOX_PROFILE_PATH):
#     """
#     Initializes a Playwright browser context with the specified Firefox profile.

#     Args:
#         firefox_profile_path (str): The path to the Firefox profile.

#     Returns:
#         BrowserContext: The initialized Playwright browser context.

#     """
#     playwright = await async_playwright().start()
#     browser_context = await playwright.firefox.launch_persistent_context(
#         user_data_dir=firefox_profile_path,
#         headless=False,
#         viewport={"width": 1600, "height": 1600},
#         # args=["--start-maximized"]
#     )
#     return browser_context


# def quit_drivers(drivers):
#     """close a list of selenium webdrivers"""
#     log(f"quitting {len(drivers)} webdrivers", "quit_drivers")
#     for driver in drivers:
#         driver.quit()


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
    trunc_len = 255 - len(datestr) - len(sep)
    filename = filename[:trunc_len]
    return filename


async def fetch_url(url, title, browser_context=None, click_xpath=None, scrolls=0, initial_sleep=SLEEP_TIME, destination=DOWNLOAD_DIR):
    """
    Fetches a URL using a Playwright browser context.

    Args:
        url (str): The URL to fetch.
        title (str): The title for the fetched page.
        click_xpath (str): An optional XPath expression to click on before saving.
        scrolls (int): The number of times to scroll to the bottom of the page and wait for new content to load.
        browser_context (BrowserContext): The Playwright browser context.
        initial_sleep (float): The number of seconds to wait after the page has loaded before clicking.

    Returns:
        str: The path to the downloaded file.

    # should add retry functionality, re-enable screenshots
    """
    log(f"fetch_url({url})")
    try:
        # make output directories
        if not os.path.exists(destination):
            os.makedirs(destination)
        # make output filename
        title = sanitize_filename(title)
        html_path = os.path.join(destination, f'{title}.html')
        # check if file already exists, don't re-download
        if os.path.exists(html_path):
            log(f"File already exists: {html_path}")
            return html_path
        # if file does not exist, download
        # delay for when you hit same site, try not to do it at same time
        await asyncio.sleep(random.uniform(0, 5))

        page = await browser_context.new_page()
        await page.goto(url, timeout=60000)
        # Stealth delay
        if click_xpath:
            await asyncio.sleep(initial_sleep+random.uniform(2, 5))
            log(f"Attempting to click on {click_xpath}")
            # click_xpath == '//*[@aria-label="Artificial intelligence"]'
            await page.wait_for_selector(f'xpath={click_xpath}')
            await page.click(f'xpath={click_xpath}')
        for _ in range(scrolls):
            await asyncio.sleep(random.uniform(2, 5))  # Stealth delay
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
        html_source = await page.content()
        # Save HTML
        log(f"Saving HTML to {html_path}")
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_source)
        await page.close()
        # Save screenshot
        # screenshot_path = f"{SCREENSHOT_DIR}/{title}.png"
        # await page.screenshot(path=screenshot_path)
        return html_path
    except Exception as exc:
        log(f"Error fetching {url}: {exc}")
        return None


async def fetch_queue(queue, concurrency):
    """
    Processes a queue of URLs concurrently using a specified number of workers.

    Args:
        queue (asyncio.Queue): The queue containing tuples of (index, url, title) to process.
        concurrency (int): The number of concurrent workers to use.

    Returns:
        list: A list of tuples containing (index, url, title, result) for each processed URL.
    """
    async with async_playwright() as p:
        log("Launching browser")
        browser = await p.firefox.launch_persistent_context(
            user_data_dir=FIREFOX_PROFILE_PATH,
            headless=False,  # run headless, hide splash window
            viewport={"width": 1366, "height": 768},
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

        async def worker(queue, browser, results):
            log("Launching worker")
            while True:
                try:
                    idx, url, title = await queue.get()
                    log(f"from queue: {idx}, {url}, {title}")
                except asyncio.QueueEmpty:
                    return
                try:
                    results.append((idx, url, title, await fetch_url(url, title, browser, destination=PAGES_DIR)))
                finally:
                    queue.task_done()

        results = []
        log("Launching workers")
        tasks = [asyncio.create_task(worker(queue, browser, results))
                 for _ in range(concurrency)]
        log("Finishing and closing browser")
        await queue.join()
        for t in tasks:
            t.cancel()
        await browser.close()

    return results


async def fetch_source(source_dict, browser_context=None):
    """
    Fetches a landing page using get_url_playwright and parameters defined in sources.yaml.
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

    log(f"Starting get_file {url}, {title}")

    # Open the page
    file_path = await fetch_url(url, title, browser_context,
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
            headless=False,  # run headless, hide splash window
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


# def process_source_queue_factory(q):
#     """creates a queue processor function closure on the queue q
#     This function expects a sourcedict in the queue
#     Used to launch parallel selenium workers on the front pages in sources.yaml

#     Args:
#         q (Queue): Multiprocessing queue containing the source dictionaries to process.
#     """
#     def process_queue(driver=None):
#         """
#         Opens a browser using Selenium driver, processes the queue until it is empty,
#         saves the file names, and then closes the browser.

#         Returns:
#             A list of tuples containing the sourcename and sourcefile for each processed item.
#         """
#         # launch browser via selenium driver
#         if not driver:
#             driver = get_browser_context()
#         saved_pages = []
#         while not q.empty():
#             sourcedict = q.get()
#             sourcename = sourcedict['sourcename']
#             log(f'Processing {sourcename}')
#             sourcefile = fetch_source(sourcedict, driver)
#             saved_pages.append((sourcename, sourcefile))
#         # Close the browser - don't quit, keep it open for more work
#         # log("Quit webdriver")
#         # driver.quit()
#         return saved_pages
#     return process_queue


# async def get_driver_async():
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, get_browser_context)


# async def get_browsers(n):
#     global BROWSERS
#     BROWSERS = await asyncio.gather(*[get_driver_async() for _ in range(n)])
#     return BROWSERS


# def process_url_queue_factory(q):
#     """creates a queue processor function closure on the queue q

#     Args:
#         q (Queue): Multiprocessing queue containing the source dictionaries to process.
#     """
#     def process_queue(driver=None):
#         """
#         Opens a browser using Selenium driver, processes the queue until it is empty,
#         saves the file names, and then closes the browser.

#         Returns:
#             A list of tuples containing the sourcename and sourcefile for each processed item.
#         """
#         # launch browser via selenium driver
#         if not driver:
#             driver = get_browser_context()
#         saved_pages = []
#         while not q.empty():
#             i, url, title = q.get()
#             log(f'Processing page {i}: {url}')
#             savefile = fetch_url(url, title, driver)
#             if savefile:
#                 saved_pages.append((i, url, title, savefile))
#             else:
#                 log(f"Error processing {url}, continuing...")
#         # Close the browser
#         log("Quit webdriver")
#         driver.quit()
#         log(f"{len(saved_pages)} pages saved")

#         return saved_pages
#     return process_queue


# def launch_drivers(n, callable):
#     """
#     Launches n threads of callable (browser scrapers) and returns the collected results.

#     Parameters:
#     callable (function): The function to be executed by each thread.
#     n (int): The number of threads to launch.

#     Returns:
#     list: A list of results collected from each thread.

#     """
#     with ThreadPoolExecutor(max_workers=n) as executor:
#         # Create a list of future objects
#         futures = [executor.submit(callable) for _ in range(n)]

#         # Collect the results (web drivers) as they complete
#         retarray = [future.result() for future in as_completed(futures)]

#     # flatten results
#     retlist = [item for retarray in retarray for item in retarray]
#     log(f"returned {len(retlist)}")

#     return retlist

# this should work with multiprocessing.Pool and be simpler but gives an error
# might not like returning a webdriver object
# def simple_parallel_download(sources):
#     num_drivers = min(3, cpu_count())  # Use 3 or the number of CPUs available, whichever is less

#     with Pool(num_drivers) as pool:
#         # Initialize drivers in parallel
#         drivers = pool.map(get_driver, range(num_drivers))

#         # Create argument list for starmap, distributing sources over drivers
#         args = [(source, drivers[i % num_drivers]) for i, source in enumerate(sources)]

#         # Download files in parallel
#         result = pool.starmap(get_file, args)

#         # Quit drivers
#         for driver in drivers:
#             driver.quit()

#     return result

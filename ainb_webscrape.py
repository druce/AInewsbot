from datetime import datetime
import time
import re
import os
# import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.common.by import By
# use firefox v. chrome b/c it updates less often, can disable updates
# recommend importing profile from Chrome for cookies, passwords
# looks less like a bot with more user cruft in the profile
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

# import bs4
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse

from ainb_const import DOWNLOAD_DIR, GECKODRIVER_PATH, FIREFOX_PROFILE_PATH, MINTITLELEN, sleeptime
from ainb_utilities import log

# get a page title from html or tags if you have not-descriptive link titles like 'link'

DRIVERS = []


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
    retdict = {}
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
                        retdict[tag["property"]] = tag["content"]

                page_title = ""
                title_tag = soup.find("title")
                if title_tag:
                    page_title = title_tag.text
                    if page_title:
                        retdict["title"] = page_title
        return retdict
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return retdict


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
    if type(link) is str:
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


def get_encoding(driver):
    """
    Retrieve the encoding of the current web page using the Selenium driver.

    Args:
        driver: The Selenium driver object.

    Returns:
        The encoding of the web page. If the encoding is not found, it defaults to utf-8.
    """
    # not used, just assumes utf-8

    # Get the current web page source
    # Retrieve the content-type meta tag from the driver of HTML meta charset tag or http-equiv tag
    encoding = None
    try:
        # retrieve the encoding from the driver by executing javascript
        encoding = driver.execute_script("return document.characterSet;")
        if encoding:
            return encoding
    except Exception:
        pass

    # not sure why above would fail, but can also check page tags
    try:
        # retrieve the encoding from the meta charset tag
        meta_tag = driver.find_element(By.XPATH, "//meta[@charset]")
        encoding = meta_tag.get_attribute("charset")
        if encoding:
            return encoding
    except Exception:
        pass

    try:
        # retrieve the encoding from the meta http-equiv tag
        meta_tag = driver.find_element(
            By.XPATH, "//meta[@http-equiv='Content-Type']")
        content_type = meta_tag.get_attribute("content")
        # Typical format is "text/html; charset=UTF-8"
        charset_start = content_type.find("charset=")
        if charset_start != -1:
            encoding = content_type[charset_start + 8:]
        if encoding:
            return encoding
    except Exception:
        pass

    return "utf-8"


def get_driver(geckodriver_path=GECKODRIVER_PATH, firefox_profile_path=FIREFOX_PROFILE_PATH):
    """
    Initializes a Selenium webdriver with the specified geckodriver and Firefox profile.

    Args:
        geckodriver_path (str): The path to the geckodriver executable.
        firefox_profile_path (str): The path to the Firefox profile.

    Returns:
        webdriver.Firefox: The initialized Firefox webdriver.

    """

    # initialize selenium driver
    # Set up Firefox options to use existing profile
    # important for some sites that need a login, also a generic profile fingerlog that looks like a bot might get blocked
    log(f"{os.getpid()} Initializing webdriver", "get_driver")

    global DRIVERS

    options = Options()
    options.profile = firefox_profile_path
    log("Initialized webdriver profile", "get_driver")

    # Create a Service object with the path
    service = Service(geckodriver_path)
    log("Initialized webdriver service", "get_driver")

    # Set up the Firefox driver
    driver = webdriver.Firefox(service=service, options=options)
    DRIVERS.append(driver)
    log("Initialized webdriver", "get_driver")
    return driver


# this doesn't work as expected, DRIVERS is empty
def quit_drivers():
    log(f"quitting {len(DRIVERS)} webdrivers", "quit_drivers")
    for driver in DRIVERS:
        driver.quit()


def get_url(url, title, driver=None):
    """
    Fetches a URL using a Selenium driver. TODO: call this within get_file

    Args:
        url (dict): A url
        driver (WebDriver, optional): The Selenium driver to use. If not provided, a new driver will be initialized.

    Returns:
        str: The path to the downloaded file.

    Raises:
        Exception: If there is an error during the execution of the function.

    """
    if not driver:
        driver = get_driver()

    log(f"starting get_url {url}", f'get_url({url})')

    try:
        # Open the page
        driver.get(url)

        # Wait for the page to load
        time.sleep(sleeptime)  # Adjust the sleep time as necessary

        # Get the HTML source of the page
        html_source = driver.page_source

        # get the page title and sanitize
        # try:
        #     title = re.sub(r'[^a-zA-Z0-9_\-]', '_', driver.title)
        #     title = title[:200]
        # except Exception as exc:
        #     print(exc)
        #     title = ''
        # if len(title) < 6:
        #     # Generate a  random UUID for title
        #     title = uuid.uuid4()

        # check encoding, default utf-8
        encoding = "utf-8"  # Default to UTF-8 if not specified
        # Retrieve the content-type meta tag from the HTML
        # try:
        #     meta_tag = driver.find_element(
        #         By.XPATH, "//meta[@http-equiv='Content-Type']")
        #     content_type = meta_tag.get_attribute("content")
        #     # Typical format is "text/html; charset=UTF-8"
        #     charset_start = content_type.find("charset=")
        #     if charset_start != -1:
        #         encoding = content_type[charset_start + 8:]
        # except Exception as err:
        #     log(str(err))

        # Save the HTML to a local file
        datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)
        trunclen = 255-len(datestr)-6
        filename = filename[:trunclen]
        outfile = f'{filename}_{datestr}.html'
        log(f"Saving {outfile} as {encoding}", f'get_url({title})')
        destpath = DOWNLOAD_DIR + "/" + outfile
        with open(destpath, 'w', encoding=encoding) as file:
            file.write(html_source)

        return destpath
    except Exception as exc:
        log(f"Error fetching {url}: {exc}")
        return None


def get_file(sourcedict, driver=None):
    """
    Fetches a URL using a Selenium driver and parameters defined in sources.yaml.
    Updates sourcedict['latest'] with the path to the downloaded file.

    Args:
        sourcedict (dict): A dictionary containing the parameters defined in sources.yaml.
        driver (WebDriver, optional): The Selenium driver to use. If not provided, a new driver will be initialized.

    Returns:
        str: The path to the downloaded file.

    Raises:
        Exception: If there is an error during the execution of the function.

    """
    if not driver:
        driver = get_driver()

    title = sourcedict["title"]
    url = sourcedict.get("url")
    scroll = sourcedict.get("scroll", 0)
    click = sourcedict.get("click")
    initial_sleep = sourcedict.get("initial_sleep")

    log(f"starting get_files {url}", f'get_files({title})')

    # Open the page
    driver.get(url)

    # Wait for the page to load
    if not initial_sleep:
        initial_sleep = sleeptime
    time.sleep(initial_sleep)  # Adjust the sleep time as necessary

    if click:
        log(f"Attempting to click on {click}", f'get_files({title})')
        button = driver.find_element(By.XPATH, click)
        if button:
            button.click()
            log("Clicked", 'get_files')

    for _ in range(scroll):
        # scroll to bottom of infinite scrolling window
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        log("Loading additional infinite scroll items",
            f'get_files({title})')
        time.sleep(sleeptime)  # wait for it to load additional items

    # Get the HTML source of the page
    html_source = driver.page_source

    # check encoding, default utf-8
    encoding = "utf-8"  # Default to UTF-8 if not specified
    # Retrieve the content-type meta tag from the HTML
    try:
        meta_tag = driver.find_element(
            By.XPATH, "//meta[@http-equiv='Content-Type']")
        content_type = meta_tag.get_attribute("content")
        # Typical format is "text/html; charset=UTF-8"
        charset_start = content_type.find("charset=")
        if charset_start != -1:
            encoding = content_type[charset_start + 8:]
    except Exception as err:
        log(str(err))

    # Save the HTML to a local file
    datestr = datetime.now().strftime("%m_%d_%Y %I_%M_%S %p")
    outfile = f'{title} ({datestr}).html'
    log(f"Saving {outfile} as {encoding}", f'get_files({title})')
    destpath = DOWNLOAD_DIR + "/" + outfile
    with open(destpath, 'w', encoding=encoding) as file:
        file.write(html_source)
    sourcedict['latest'] = destpath

    return destpath


def parse_file(sourcedict):
    """
    Parse a saved HTML file and return a list of dictionaries with title, url, src for each link in the file.

    Args:
        sourcedict (dict): A dictionary containing the source information.

    Returns:
        list: A list of dictionaries, where each dictionary represents a link in the HTML file.
              Each dictionary contains the following keys: 'title', 'url', 'src'.

    Raises:
        None

    """
    sourcename = sourcedict['sourcename']
    title = sourcedict["title"]
    filename = sourcedict["latest"]
    url = sourcedict.get("url")
    exclude = sourcedict.get("exclude")
    include = sourcedict.get("include")
    minlength = sourcedict.get("minlength", MINTITLELEN)

    retlist = []
    # get contents
    with open(filename, "r") as file:
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
            newlinks = []
            for link in links:
                href = link.get("href")
                if href and re.match(pattern, href):
                    newlinks.append(link)
            links = newlinks

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

        retlist.append({"title": title, "url": url, "src": sourcename})

    log(f"found {len(retlist)} filtered links", "parse_file")

    return retlist


def process_source_queue_factory(q):
    """creates a queue processor function closure on the queue q
    This function expects a sourcedict in the queue
    Used to launch parallel selenium workers on the front pages in sources.yaml

    Args:
        q (Queue): Multiprocessing queue containing the source dictionaries to process.
    """
    def process_queue():
        """
        Opens a browser using Selenium driver, processes the queue until it is empty,
        saves the file names, and then closes the browser.

        Returns:
            A list of tuples containing the sourcename and sourcefile for each processed item.
        """
        # launch browser via selenium driver
        driver = get_driver()
        saved_pages = []
        while not q.empty():
            sourcedict = q.get()
            sourcename = sourcedict['sourcename']
            log(f'Processing {sourcename}')
            sourcefile = get_file(sourcedict, driver)
            saved_pages.append((sourcename, sourcefile))
        # Close the browser
        log("Quit webdriver")
        driver.quit()
        return saved_pages
    return process_queue


def process_url_queue_factory(q):
    """creates a queue processor function closure on the queue q

    Args:
        q (Queue): Multiprocessing queue containing the source dictionaries to process.
    """
    def process_queue():
        """
        Opens a browser using Selenium driver, processes the queue until it is empty,
        saves the file names, and then closes the browser.

        Returns:
            A list of tuples containing the sourcename and sourcefile for each processed item.
        """
        # launch browser via selenium driver
        driver = get_driver()
        saved_pages = []
        while not q.empty():
            i, url, title = q.get()
            log(f'Processing {url}')
            savefile = get_url(url, title, driver)
            if savefile:
                saved_pages.append((i, url, title, savefile))
            else:
                log(f"Error processing {url}, continuing...")
        # Close the browser
        log("Quit webdriver")
        driver.quit()
        return saved_pages
    return process_queue


def launch_drivers(n, callable):
    """
    Launches n threads of callable (browser scrapers) and returns the collected results.

    Parameters:
    callable (function): The function to be executed by each thread.
    n (int): The number of threads to launch.

    Returns:
    list: A list of results collected from each thread.

    """
    with ThreadPoolExecutor(max_workers=n) as executor:
        # Create a list of future objects
        futures = [executor.submit(callable) for _ in range(n)]

        # Collect the results (web drivers) as they complete
        retarray = [future.result() for future in as_completed(futures)]

    # flatten results
    retlist = [item for retarray in retarray for item in retarray]
    return retlist

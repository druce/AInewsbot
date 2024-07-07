import os
import time
import json
import tiktoken
import asyncio
import aiohttp
import openai
from bs4 import BeautifulSoup
from ainb_const import (LOWCOST_MODEL, MODEL, MAX_OUTPUT_TOKENS, MAX_RETRIES, TEMPERATURE,
                        sleeptime, MAX_INPUT_TOKENS, MAXPAGELEN,
                        FILTER_PROMPT, SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT)
from ainb_utilities import log


def count_tokens(s):
    """
    Counts the number of tokens in a given string.

    Args:
        s (str): The input string.

    Returns:
        int: The number of tokens in the input string.
    """
    # no tokeniser returned yet for gpt-4o-2024-05-13
    enc = tiktoken.encoding_for_model(MODEL)
    # enc = tiktoken.get_encoding('o200k_base')
    assert enc.decode(enc.encode("hello world")) == "hello world"
    return len(enc.encode(s))


def trunc_tokens(long_prompt, model=MODEL, maxtokens=MAX_INPUT_TOKENS):

    # Initialize the encoding for the model you are using, e.g., 'gpt-4'
    encoding = tiktoken.encoding_for_model(model)

    # Encode the prompt into tokens, truncate, and return decoded prompt
    tokens = encoding.encode(long_prompt)
    tokens = tokens[:maxtokens]
    truncated_prompt = encoding.decode(tokens)

    return truncated_prompt


async def get_response_json(
    client,
    messages,
    verbose=False,
    model=LOWCOST_MODEL,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    max_retries=MAX_RETRIES,
    temperature=TEMPERATURE,
):
    """
    Calls the OpenAI client with messages and returns the response as a JSON string.
    response_format={"type": "json_object"} forces ChatGPT to return a valid JSON string.

    Args:
        client (OpenAI.ChatCompletionClient): The OpenAI client used to make the API call.
        messages (str or list): The messages to send to the chat model. If a string is provided, it will be converted to a list with a single message.
        verbose (bool, optional): Whether to log the messages to the console. Defaults to False.
        model (str, optional): The model to use for chat completion. Defaults to MODEL.
        max_output_tokens (int, optional): The maximum number of tokens in the response. Defaults to MAX_OUTPUT_TOKENS.
        max_retries (int, optional): The maximum number of retries in case of API errors. Defaults to MAX_RETRIES.
        temperature (float, optional): The temperature parameter for generating diverse responses. Defaults to TEMPERATURE.

    Returns:
        dict: The response from the OpenAI API as a JSON object.

    Raises:
        Exception: If an error occurs during the API call.

    """
    if type(messages) is not list:  # allow passing one string for convenience
        messages = [{"role": "user", "content": messages}]

    if verbose:
        log("\n".join([str(msg) for msg in messages]))

    # truncate number of tokens
    # retry loop, have received untrapped 500 errors like too busy
    for i in range(max_retries):
        if i > 0:
            log(f"Attempt {i+1}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            # no exception thrown
            return response
        except Exception as error:
            log(f"An exception occurred on attempt {i+1}:", error)
            time.sleep(sleeptime)
            continue  # try again
        # retries exceeded if you got this far
    log("Retries exceeded.")
    return None


async def fetch_response(client, prompt, p):
    """Fetches the response from the OpenAI client based on the given prompt and page.

    Args:
        client (OpenAI.Client): The OpenAI client used to make the API request.
        prompt (str): The prompt to be processed by the OpenAI client.
        p (dict): The page containing keys and values to be processed.

    Returns:
        list: The response as a list of keys and values.

    Raises:
        TypeError: If the response is not a dictionary or a list.

    """
    retlist = []
    response = await get_response_json(client, prompt + json.dumps(p))
    response_json = json.loads(response.choices[0].message.content)

    if type(response_json) is dict:
        for k, v in response_json.items():
            if type(v) is list:  # came back correctly as e.g. {'stories': []}
                retlist.extend(v)
            else:  # maybe a weird dict with keys  of id
                retlist.append(v)
        log(f"got dict with {len(retlist)} items ")
    elif type(response_json) is list:
        retlist = response_json
        log(f"got list with {len(retlist)} items ")
    else:
        raise TypeError("Error: Invalid response type")

    sent_ids = [s['id'] for s in p]
    received_ids = [r['id'] for r in response_json['stories']]
    difference = set(sent_ids) - set(received_ids)

    if difference:
        log(f"missing items, {str(difference)}")
        return []
    else:
        return retlist


def paginate_df(filtered_df, maxpagelen=MAXPAGELEN, max_input_tokens=MAX_INPUT_TOKENS):
    """
    Splits the filtered dataframe into pages based on the maximum page length in rows and maximum input tokens in titles.

    Args:
        filtered_df (pandas.DataFrame): The filtered dataframe to be paginated.
        maxpagelen (int, optional): The maximum length of each page in characters. Defaults to MAXPAGELEN.
        max_input_tokens (int, optional): The maximum number of tokens allowed in each page. Defaults to MAX_INPUT_TOKENS.

    Returns:
        list: A list of pages, where each page is a list of dictionaries representing the links.

    """
    pages = []
    current_page = []
    pagelength = 0

    for row in filtered_df.itertuples():
        curlink = {"id": row.Index, "title": row.title}
        curlength = count_tokens(json.dumps(curlink))
        # Check if adding the current string would exceed the limit
        if len(current_page) >= maxpagelen or pagelength + curlength > max_input_tokens:
            # If so, start a new page
            pages.append(current_page)
            current_page = [curlink]
            pagelength = curlength
        else:
            # Otherwise, add the string to the current page
            current_page.append(curlink)
            pagelength += curlength

    # add the last page if it's not empty
    if current_page:
        pages.append(current_page)
    return pages


async def process_pages(client, prompt, pages):
    """
    Process a list of pages by sending them individually to the OpenAI API with the given prompt.

    Args:
        client (OpenAI.Client): The OpenAI API client.
        prompt (str): The prompt template to be sent to the OpenAI API.
        pages (list): A list of pages to be processed.

    Returns:
        list: A list of enriched URLs.

    """
    client = openai.ChatCompletion.create()
    enriched_urls = []
    tasks = []
    for i, p in enumerate(pages):
        log(f"send page {i+1} of {len(pages)}, {len(p)} items ")
        task = asyncio.create_task(fetch_response(client, prompt, p))
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    for retlist in responses:
        if retlist:
            enriched_urls.extend(retlist)
        else:
            log("process_pages failed")
    return enriched_urls


# this version runs faster and hits the endpoint directly using aiohttp instead of the OpenAI Python client

API_URL = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
}


async def fetch_openai(session, payload):
    """
    Asynchronously fetches a response from the OpenAI URL using an aiohttp ClientSession.

    Parameters:
    - session (aiohttp.ClientSession): The aiohttp ClientSession object used for making HTTP requests.
    - payload (dict): The payload to be sent in the request body as JSON.

    Returns:
    - dict: The full JSON response from the OpenAI API.

    Raises:
    - aiohttp.ClientError: If there is an error during the HTTP request.

    Example usage:
    ```
    async with aiohttp.ClientSession() as session:
        response = await fetch_openai(session, payload)
        print(response)
    ```
    """
    async with session.post(API_URL, headers=headers, json=payload) as response:
        return await response.json()


async def async_filter_page(
    p,
    session,
    prompt=FILTER_PROMPT,
    model=LOWCOST_MODEL,
    max_retries=MAX_RETRIES,
    temperature=TEMPERATURE,
    verbose=False,
):
    retlist = []
    for i in range(max_retries):
        try:
            if i > 0:
                log(f"Attempt {i+1}...")

            messages = [{"role": "user",
                         "content": prompt + json.dumps(p)
                         }]

            payload = {"model":  model,
                       "response_format": {"type": "json_object"},
                       "messages": messages,
                       "temperature": temperature
                       }

            log(f"sent {len(p)} items ")

            response = await fetch_openai(session, payload)

            # valid json
            response_json = json.loads(
                response["choices"][0]["message"]["content"])

            if verbose:
                print(response_json)

            # valid dict in json
            if type(response_json) is dict:
                for k, v in response_json.items():
                    # came back correctly as e.g. {'stories': []}
                    if type(v) is list:
                        retlist.extend(v)
                    else:  # maybe a weird dict with keys  of id
                        retlist.append(v)
                log(f"got dict with {len(retlist)} items ")
            elif type(response_json) is list:
                retlist = response_json
                log(f"got list with {len(retlist)} items ")
            else:
                raise TypeError("Error: Invalid response type")

            # sent items match received items
            sent_ids = [s['id'] for s in p]
            received_ids = [r['id'] for r in retlist]
            difference = set(sent_ids) - set(received_ids)

            if verbose:
                print(difference)

            if difference:
                log(f"missing items, {str(difference)}")
                raise TypeError("Error: response mismatch")

            # success
            return retlist

        except Exception as exc:
            log(f"Error: {exc}")

    # failed all retries, return []
    log("MAX_RETRIES exceeded")
    return retlist


async def fetch_pages(pages,
                      prompt=FILTER_PROMPT,
                      model=LOWCOST_MODEL,
                      max_retries=MAX_RETRIES,
                      temperature=TEMPERATURE,
                      verbose=False,):

    tasks = []
    async with aiohttp.ClientSession() as session:
        for p in pages:
            task = asyncio.create_task(async_filter_page(p,
                                                         session,
                                                         prompt=prompt,
                                                         model=model,
                                                         max_retries=max_retries,
                                                         temperature=temperature,
                                                         verbose=verbose,
                                                         ))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
    retlist = [item for sublist in responses for item in sublist]

    log(f"Processed {len(retlist)} responses.")

    return retlist


async def fetch_all_summaries(page_df):
    tasks = []
    responses = []
    async with aiohttp.ClientSession() as session:

        for row in page_df.itertuples():

            # Read the HTML file
            try:
                with open(row.path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
            except Exception as exc:
                log(f"Error: {str(exc)}")
                log(f"Skipping {row.id} : {row.path}")
                continue

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            try:
                # Try to get the title from the <title> tag
                title_tag = soup.find("title")
                title_str = "Page title: " + title_tag.string.strip() + \
                    "\n" if title_tag and title_tag.string else ""
            except Exception as exc:
                log(str(exc), "fetch_all2 page_title")

            try:
                # Try to get the title from the Open Graph meta tag
                og_title_tag = soup.find("meta", property="og:title")
                og_title = og_title_tag["content"].strip(
                ) + "\n" if og_title_tag and og_title_tag.get("content") else ""
                if not og_title:
                    og_title_tag = soup.find(
                        "meta", attrs={"name": "twitter:title"})
                    og_title = "Social card title: " + og_title_tag["content"].strip(
                    ) + "\n" if og_title_tag and og_title_tag.get("content") else ""
            except Exception as exc:
                log(str(exc), "fetch_all2 og_title")

            try:
                # get summary from social media cards
                og_desc_tag = soup.find("meta", property="og:description")
                og_desc = f'Summary: {og_desc_tag["content"]}' + \
                    "\n" if og_desc_tag else ""
                if not og_desc:
                    # Extract the Twitter description
                    og_desc_tag = soup.find(
                        "meta", attrs={"name": "twitter:description"})
                    og_desc = f'Summary: {og_desc_tag["content"]}' + \
                        "\n" if og_desc_tag else ""
            except Exception as exc:
                log(str(exc), "fetch_all2 og_desc")

            # Filter out script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()

            # Get text and strip leading/trailing whitespace

            visible_text = title_str + og_title + og_desc + \
                soup.get_text(separator=' ', strip=True)
            visible_text = trunc_tokens(
                visible_text, model=MODEL, maxtokens=MAX_INPUT_TOKENS)

            userprompt = f"""{SUMMARIZE_USER_PROMPT}:
{visible_text}
            """

            payload = {"model":  LOWCOST_MODEL,
                       "messages": [{"role": "system",
                                     "content": SUMMARIZE_SYSTEM_PROMPT
                                     },
                                    {"role": "user",
                                     "content": userprompt
                                     }]
                       }

            task = asyncio.create_task(
                fetch_openai_summary(session, payload, row.id))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
    return responses


async def fetch_openai_summary(session, payload, i):
    """
    Asynchronously fetches a response from the OpenAI URL using an aiohttp ClientSession.
    This version returns the id as well as the response, to allow us to map the summary to the original request.

    Parameters:
    - session (aiohttp.ClientSession): The aiohttp ClientSession object used for making HTTP requests.
    - payload (dict): The payload to be sent in the request body as JSON.
    - i (int): an id to return, to allow us map summary to original request

    Returns:
    - dict: The full JSON response from the OpenAI API.

    Raises:
    - aiohttp.ClientError: If there is an error during the HTTP request.

    Example usage:
    ```
    async with aiohttp.ClientSession() as session:
        response = await fetch_openai(session, payload)
        print(response)
    ```
    """
    async with session.post(API_URL, headers=headers, json=payload) as response:
        retval = await response.json()
        return (i, retval)

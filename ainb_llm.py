import os
import time
import json
from collections import defaultdict

import aiohttp
import asyncio

import tiktoken
import openai
from bs4 import BeautifulSoup
import trafilatura

from ainb_const import (LOWCOST_MODEL, BASEMODEL, MODEL, MAX_OUTPUT_TOKENS, MAX_RETRIES, TEMPERATURE,
                        sleeptime, MAX_INPUT_TOKENS, MAXPAGELEN, CANONICAL_TOPICS,
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
    enc = tiktoken.encoding_for_model(BASEMODEL)
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
    json_schema=None,
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
            if json_schema:
                # use schema given
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                )
            else:
                # no schema given
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
    json_schema=None,
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

            if json_schema:
                response_format = {"type": "json_schema",
                                   "json_schema": json_schema}
            else:
                response_format = {"type": "json_object"}

            payload = {"model":  model,
                       "response_format": response_format,
                       "messages": messages,
                       "temperature": temperature
                       }

            log(f"sent {len(p)} items ")

            response = await fetch_openai(session, payload)
            # log(type(response))
            # log(response)
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
                      json_schema=None,
                      max_retries=MAX_RETRIES,
                      temperature=TEMPERATURE,
                      verbose=False,):

    tasks = []
    log(f"Applying prompt to {len(pages)} pages using {model}")

    async with aiohttp.ClientSession() as session:
        for p in pages:
            task = asyncio.create_task(async_filter_page(p,
                                                         session,
                                                         prompt=prompt,
                                                         model=model,
                                                         json_schema=json_schema,
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

            # Parse the HTML content using trafilatura
            soup = BeautifulSoup(html_content, 'html.parser')

            try:
                # Try to get the title from the <title> tag
                title_tag = soup.find("title")
                title_str = "Page title: " + title_tag.string.strip() + \
                    "\n" if title_tag and title_tag.string else ""
            except Exception as exc:
                log(str(exc), "fetch_all_summaries page_title")

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
                log(str(exc), "fetch_all_summaries og_title")

            try:
                # get summary from social media cards
                og_desc_tag = soup.find("meta", property="og:description")
                if not og_desc_tag:
                    # Extract the Twitter description
                    og_desc_tag = soup.find(
                        "meta", attrs={"name": "twitter:description"})
                og_desc = og_desc_tag["content"] + "\n" if og_desc_tag else ""
                og_desc = 'Social card description: ' + og_desc if og_desc else ""
            except Exception as exc:
                log(str(exc), "fetch_all_summaries og_desc")

            # Get text and strip leading/trailing whitespace
            log(title_str + og_title + og_desc, "fetch_all_summaries")
            plaintext = ""
            try:
                plaintext = trafilatura.extract(html_content)
                plaintext = plaintext.strip() if plaintext else ""
            except Exception as exc:
                log(str(exc), "fetch_all_summaries trafilatura")

            visible_text = title_str + og_title + og_desc + plaintext
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


async def categorize_headline(AIdf,
                              categories=CANONICAL_TOPICS,
                              maxpagelen=MAXPAGELEN,
                              model=LOWCOST_MODEL):

    topic_prompt_template = """
You will act as a research assistant to categorize news headlines based whether they discuss {topic}.
The input will be formatted as a list of JSON objects.
You will closely read each headline to determine if it discusses {topic}.
You will respond with a list of JSON objects.

Input Specification:
You will receive a list of news headlines formatted as JSON objects.
Each object will include an 'id' and a 'headline'.

Classification Criteria:
Classify each story based on its headline to determine whether it discusses {topic}.

Output Specification:
You will return a JSON object with the field 'headlines' containing an array of classification results.
For each headline, your output will be a JSON object containing the original 'id' and a new field 'relevant',
a boolean indicating if the story is about {topic}. You must strictly adhere to this output schema, without
modification.

Example Output Format:
{{'headlines':
  [{{'id': 97, 'relevant': true}},
   {{'id': 103, 'relevant': true}},
   {{'id': 103, 'relevant': false}},
   {{'id': 210, 'relevant': true}},
   {{'id': 298, 'relevant': false}}]
}}

Instructions:
Ensure that each output object accurately reflects the 'id' field of the corresponding input object
and that the 'relevant' field accurately represents the title's relevance to {topic}.

The list of headlines to classify is:"""

    # structured response format
    json_schema = {
        "name": "headline_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headlines": {
                    "type": "array",
                    "items": {
                        "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "number"
                                    },
                                    "relevant": {
                                        "type": "boolean"
                                    }
                                },
                        "required": ["id", "relevant"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["headlines"],
            "additionalProperties": False
        }
    }

    log("Start canonical topic classification")
    # Create a defaultdict where each element is a list
    canonical_dict = defaultdict(list)
    pages = paginate_df(AIdf, maxpagelen=maxpagelen)
    for tid, topic in enumerate(categories):
        log(f"{topic}, topic {tid+1} of {len(categories)}")
        enriched_urls = asyncio.run(fetch_pages(pages,
                                                model=model,
                                                prompt=topic_prompt_template.format(
                                                    topic=topic),
                                                json_schema=json_schema))
        relevant_ids = [d['id'] for d in enriched_urls if d['relevant']]
        for relevant_id in relevant_ids:
            canonical_dict[relevant_id].append(topic)
    log("end canonical topic classification")
    return canonical_dict


async def categorize_headline_old(headline, categories, session,
                                  model=LOWCOST_MODEL,
                                  temperature=0.5,
                                  max_retries=MAX_RETRIES):
    """Match headline to specified category(ies)"""
    # TODO: refactor to to send all headlines at once per keyword using async_filter_page

    retlist = []
    if type(categories) is not list:
        categories = [categories]

    # structured response format
    json_schema = {
        "name": "extracted_topics",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "number",
                }
            },
            "required": ["response"],
            "additionalProperties": False,
        }
    }

    for topic in categories:
        cat_prompt = f"""You are a news topic categorizaton assistant. I will provide a headline
and a topic. You will respond with a JSON object {{'response': 1}} if the news headline matches
the news topic and {{'response': 0}} if it does not. Check carefully and only return {{'response': 1}}
if the headline closely matches the topic. If the headline is not a close match or if unsure,
return {{'response': 0}}
Headline:
{headline}
Topic:
{topic}
"""
        for i in range(max_retries):
            try:
                response = None
                messages = [
                    {"role": "user", "content": cat_prompt
                     }]
                payload = {"model":  model,
                           'response_format': {"type": "json_schema",
                                               "json_schema": json_schema},                           "messages": messages,
                           "temperature": temperature,
                           }
                response = await fetch_openai(session, payload)
                response_dict = json.loads(
                    response["choices"][0]["message"]["content"])
                response_val = response_dict['response']
                if response_val == 1:
                    retlist.append(topic)
                # success
                return retlist
            except Exception as exc:
                if response:
                    log(response)
                log(f"Error: {exc}")

    return retlist


def clean_topics(row):
    # clean up free form topics
    topics = [x.title() for x in row.topics if x.lower()
              not in {"ai", "artificial intelligence"}]
    # clean up canonical topics
    assigned_topics = [x.title() for x in row.assigned_topics]
    combined = sorted(list(set(topics + assigned_topics)))
    combined = [s.replace("Ai", "AI") for s in combined]
    combined = [s.replace("Genai", "Gen AI") for s in combined]

    return ", ".join(combined)


async def categorize_df(AIdf):
    catdict = {}
    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(AIdf.itertuples()):
            tasks = []
            log(f"Categorizing headline {row.id+1} of {len(AIdf)}")
            h = row.title
            log(h)
            for c in CANONICAL_TOPICS:
                task = asyncio.create_task(categorize_headline(h, c, session))
                tasks.append(task)
            responses = await asyncio.gather(*tasks)
            catdict[row.id] = [item for l in responses for item in l]
            log(catdict[row.id])
    return catdict

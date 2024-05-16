import time
from datetime import datetime
import json
import tiktoken

from ainb_const import MODEL, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_RETRIES, MAXPAGELEN, TEMPERATURE, sleeptime
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
    enc = tiktoken.encoding_for_model('gpt-4')
    assert enc.decode(enc.encode("hello world")) == "hello world"
    return len(enc.encode(s))


def get_response_json(
    client,
    messages,
    verbose=False,
    model=MODEL,
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


def fetch_response(client, prompt, p):
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

    try:
        response = get_response_json(client, prompt + json.dumps(p))
        response_json = json.loads(response.choices[0].message.content)

        if type(response_json) is dict:
            for k, v in response_json.items():
                if type(v) is list:
                    retlist.extend(v)
                else:
                    retlist.append(v)
            log(f"{datetime.now().strftime('%H:%M:%S')} got dict with {len(retlist)} items ")
        elif type(response_json) is list:
            retlist = response_json
            log(f"{datetime.now().strftime('%H:%M:%S')} got list with {len(retlist)} items ")
        else:
            raise TypeError("Error: Invalid response type")

        sent_ids = [s['id'] for s in p]
        received_ids = [r['id'] for r in response_json['stories']]
        difference = set(sent_ids) - set(received_ids)
        if difference:
            log(f"missing items, {str(difference)}")
            return []

    except Exception as err:
        log(f"Error in fetch_response: {err}")
        return []

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


def process_pages(client, prompt, pages):
    """
    Process a list of pages by sending them individually to the OpenAI API with the given prompt.

    Args:
        client (OpenAI.Client): The OpenAI API client.
        prompt (str): The prompt template to be sent to the OpenAI API.
        pages (list): A list of pages to be processed.

    Returns:
        list: A list of enriched URLs.

    """
    enriched_urls = []
    for i, p in enumerate(pages):
        log(f"send page {i+1} of {len(pages)}, {len(p)} items ")
        for c in range(MAX_RETRIES):
            if c:
                log(f"Retrying, attempt {c+1}")
            retlist = fetch_response(client, prompt, p)
            if retlist:
                break
        if retlist:
            enriched_urls.extend(retlist)
        else:
            log(f"failed after {c+1} attempts")
    return enriched_urls

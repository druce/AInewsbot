import time
from datetime import datetime
import json
import tiktoken
from IPython.display import Markdown, display

from ainb_const import MODEL, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_RETRIES, MAXPAGELEN, TEMPERATURE, sleeptime, bb_agent_system_prompt
from ainb_utilities import log
from BB_agent_tool import BB_agent_tool

import openai
from openai import OpenAI
client = OpenAI()


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


# utility functions to support use of client-side tools when querying openai
def eval_tool(tool_call, verbose=True):
    """
    Given an OpenAI tool_call response,
    evaluates the tool function using the arguments provided by OpenAI,
    and returns the message to send back to OpenAI, including the function return value.

    Args:
        tool_call (object): The OpenAI tool_call response.

    Returns:
        dict: The message to send back to OpenAI, containing the tool_call_id, role, name, and value returned by the tool call.

    """
    try:
        function_name = tool_call.function.name
        # look up the function based in global tools on the name
        fn = BB_agent_tool.agent_registry[function_name]
        # make the tool call's json args into a dict
        kwargs = json.loads(tool_call.function.arguments)

        if verbose:
            print(f"{function_name}({str(kwargs)}) -> ", end="")
        # call function with the args and return value
        fn_value = fn(**kwargs)
        if type(fn_value) is list or type(fn_value) is dict:
            fn_value = str(fn_value)
        if verbose:
            output = str(fn_value)
            if len(output) > 100:
                output = output[:100] + "..."
            print(output)

        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": fn_value,
        }
    except Exception as exc:
        return f"Error: {exc}"


# utility function to call chatgpt
def get_response(messages, tools, model=MODEL, json_format=False):
    """
    Get a single response from ChatGPT based on a chain of messages.

    Args:
        messages (list): A list of message objects representing the conversation history.
        json_format(boolean): True if JSON response requested. (Last message must express the request for JSON response.)

    Returns:
        dict: A response object containing the generated response from ChatGPT.

    Raises:
        OpenAIError: If there is an error during the API call.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What's the weather like today?"},
        ... ]
        >>> response = get_response(messages)
    """

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # can't pass None, need to pass NotGiven
        tools=tools if tools else openai.NotGiven(),
        # tool_choice="auto",  # auto is default, but we'll be explicit
        response_format={"type": "json_object"} if json_format else None,
    )

    return response


def get_response_and_eval(messages, tools=[], json_format=False, raw=False, verbose=False):
    """
    Sends a list of messages to OpenAI and returns the response.
    If tool calls are returned, calls all the tools and sends the values back to OpenAI.
    If further tool calls returned, iterates until no more tool calls are returned and
    'stop' is returned as finish_reason, then returns the response.

    Args:
        messages (list): A list of messages to send to OpenAI.
        json_format (boolean): If the final response should be in JSON format.
        raw (boolean): after last tool is called return raw data response that enabled answering the question
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        response: The final response object returned by OpenAI.

    Raises:
        None

    """
    response = get_response(messages, tools=tools, json_format=json_format)
    choice = response.choices[0]
    response_message = choice.message
    finish_reason = choice.finish_reason

    if verbose:
        print(choice)

    while finish_reason != 'stop':
        # Extend conversation with assistant's reply
        messages.append(response_message)
        if finish_reason == 'tool_calls':
            tool_calls = response_message.tool_calls
            if verbose:
                print(tool_calls)
            # Call the tools and add all return values as messages
            for tool in tool_calls:
                messages.append(eval_tool(tool, verbose=True))
            # Get next response
            response = get_response(
                messages, tools=tools, json_format=json_format)
            choice = response.choices[0]
            response_message = choice.message
            finish_reason = choice.finish_reason
            if verbose:
                output = str(choice)
                output = output[:1000] + \
                    "..." if len(output) > 1000 else output
                print(output)
        else:
            print('finish_reason: ', finish_reason)
            break

    if raw:
        # probably want to process that message and return call signature + value
        return messages[-1]
    else:
        return response


def agent_query(user_message, raw=False, verbose=True):
    """
    Send a user message to OpenAI and retrieve the response, calling all tools until done.

    Args:
        user_message (str): The message from the user.
        raw (boolean): after last tool is called return raw data response that enabled answering the question
        verbose (bool, optional): Display intermediate tool calls and return values. Defaults to False.

    Returns:
        str: The response from the agent.


    Example:
        >>> agent_query("Hello")
        'Hello! How can I assist you today?'
    """

    # recompute system prompt, adding tool metadata i.e. descriptions of available tools to system prompt
    tool_descs = ""
    openai_tools = []
    for v in BB_agent_tool.agent_registry.values():
        t = v.tooldict
        openai_tools.append(t)
        tname = t['function']['name']
        tdesc = t['function']['description']
        tool_descs += f"{tname} : {tdesc}"
        if v.example_code:
            tool_descs += f" Usage: {v.example_code}"
        tool_descs += "\n---\n"
    # tool_descs = "\n".join([f"{tool['function']['name']} : {tool['function']['description']}" for tool in tools.values()])
    current_system_prompt = bb_agent_system_prompt + f"""

Available tools, with name, description, and calling example, delimited by ---:
{tool_descs}
    """
    # print(bb_agent_system_prompt)
    RETRIES = 3
    for retry in range(RETRIES):
        try:
            if retry:
                print(f"retrying, attempt {retry + 1}")
            messages = [{"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": user_message}]
            response = get_response_and_eval(
                messages, tools=openai_tools, json_format=False, raw=raw, verbose=verbose)
            response_str = response.choices[0].message.content
            # escape stuff that is interpreted as latex
            response_str = response_str.replace("$", "\\\$")
            display(Markdown(response_str))
            # success, exit retry loop
            break
        except Exception as exc:
            print(exc)

"""
Module for interacting with Large Language Models (LLMs) in the AInewsbot project.

Most importantly, apply a prompt to a each row in a dataframe, either sending
the entire dataframe to the model at once (paginating it) or sending each row individually,
sending all pages or rows in parallel (using asyncio), and returning structured outputs.

This module provides functions for calling LLMs to classify, extract topics, summarize, and filter
news articles. It also provides functions for calling LLMs to generate a newsletter and podcast
based on the filtered and summarized articles.

The module is designed to be used by the AInewsbot script to generate a newsletter and podcast
based on the latest AI news.
"""
# flake8: noqa: E722
# pylint: disable=W0718  # bare-except

# import pdb
import os
import re
import math
# import aiohttp
import asyncio
from typing import List, Type, Dict, Any, Callable, Optional, Union, Tuple, Awaitable

import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# import openai
# from openai import OpenAI

# import langchain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
# from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, HumanMessage, ToolMessage
# MessagesPlaceholder, PromptTemplate,
from langchain_core.prompts import (ChatPromptTemplate,)
# SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from .utilities import log
from .config import (TENACITY_RETRY)
from .prompts import (
    CANONICAL_SYSTEM_PROMPT, CANONICAL_USER_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT,
)
from .llm_output_schemas import (
    StoryRatings,
    CanonicalTopicSpecList,
    T,
)
##############################################################################
# utility functions
##############################################################################

# def count_tokens(s):
#     """
#     Counts the number of tokens in a given string.

#     Args:
#         s (str): The input string.

#     Returns:
#         int: The number of tokens in the input string.
#     """
#     # no tokeniser returned yet for gpt-4o-2024-05-13
#     enc = tiktoken.encoding_for_model('gpt-4o')
#     # enc = tiktoken.get_encoding('o200k_base')
#     assert enc.decode(enc.encode("hello world")) == "hello world"
#     return len(enc.encode(s))

def sanitize_error_for_logging(error_msg: Any) -> str:
    """Remove API keys from error messages"""
    # Remove common API key patterns
    patterns = [
        # OpenAI keys (various formats)
        r'sk-[a-zA-Z0-9]{20,}',
        r'sk-proj-[a-zA-Z0-9]{48}',

        # Anthropic keys
        r'sk-ant-[a-zA-Z0-9-]{40,}',

        # Google keys (multiple formats)
        r'AIza[a-zA-Z0-9_-]{35}',
        r'gsk_[a-zA-Z0-9]{40,}',
        r'ya29\.[a-zA-Z0-9_-]+',

        # AWS keys
        r'AKIA[0-9A-Z]{16}',
        r'aws_access_key_id=[A-Z0-9]+',

        # Generic patterns (case-insensitive)
        r'(?i)api[_-]?key\s*[=:]\s*["\']?[a-zA-Z0-9_-]{8,}["\']?',
        r'(?i)authorization\s*:\s*[^\s,]+',
        r'(?i)bearer\s+[a-zA-Z0-9_.-]+',
        r'(?i)token\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',
        r'(?i)secret\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',
        r'(?i)password\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',

        # JSON context
        r'"(?:api_key|token|secret|password)"\s*:\s*"[^"]+',

        # URL parameters
        r'[?&](?:api_key|token|secret|password)=[^&\s]+',

        # GitHub tokens
        r'ghp_[a-zA-Z0-9]{36}',
        r'gho_[a-zA-Z0-9]{36}',

        # Slack tokens
        r'xox[bpoa]-[0-9]+-[0-9]+-[a-zA-Z0-9]+',
    ]

    sanitized = str(error_msg)
    for pattern in patterns:
        sanitized = re.sub(pattern, '[API_KEY_REDACTED]', sanitized, flags=re.IGNORECASE)

    return sanitized

def paginate_df(input_df: pd.DataFrame,
                maxpagelen: int = 50) -> List[pd.DataFrame]:
    """
    Paginates a DataFrame into smaller DataFrames of at most maxpagelen rows each.

    Args:
        input_df (pd.DataFrame): Input DataFrame to be paginated
        maxpagelen (int): Maximum number of rows per paginated DataFrame (default: 1000)

    Returns:
        List[pd.DataFrame]: List of paginated DataFrames

    Example:
        >>> df = pd.DataFrame({'A': range(1500)})
        >>> pages = paginate_df(df, maxpagelen=500)
        >>> [len(page) for page in pages]
        [500, 500, 500]
    """
    # Input validation
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame")
    if not isinstance(maxpagelen, int) or maxpagelen <= 0:
        raise ValueError("maxpagelen must be a positive integer")

    # Calculate number of pages needed
    total_rows = len(input_df)
    if total_rows == 0:
        return [input_df.copy()]

    num_pages = math.ceil(total_rows / maxpagelen)

    # Create list of paginated DataFrames
    paginated_dfs = []
    for page_num in range(num_pages):
        start_idx = page_num * maxpagelen
        end_idx = min((page_num + 1) * maxpagelen, total_rows)
        paginated_dfs.append(input_df.iloc[start_idx:end_idx].copy())

    return paginated_dfs


##############################################################################
# basic langchain call async function with decorators
##############################################################################

# CALLS_PER_MINUTE = 60  # Adjust based on your quota
# MAX_CONCURRENT = 5     # Maximum concurrent requests
# sem = Semaphore(MAX_CONCURRENT) # semaphore for controlling concurrent requests

# @sleep_and_retry
# @limits(calls=CALLS_PER_MINUTE, period=60)
@retry(
    stop=stop_after_attempt(TENACITY_RETRY),  # Maximum 5 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: log(sanitize_error_for_logging(
        f"Attempt {retry_state.attempt_number}: {retry_state.outcome.exception()}, tag: {retry_state.args[1].get('tag', '')}")),
    reraise=True,  # Make sure to re-raise the final exception after all retries are exhausted
)
async def async_langchain(chain: Any, input_dict: Dict[str, Any], tag: str = "", label: str = "article", verbose: bool = False) -> Union[Tuple[Any, str, int], Tuple[Any, str]]:
    #     async with sem:
    """
    call langchain asynchronously with ainvoke
    adds retry via tenacity decorator
    also adds a reference tag so if we gather 100 async responses we can match them up with the input
    also returns the length of a chosen input item like "article" which we are summarizing (admittedly hacky)
    """
    if verbose:
        print(sanitize_error_for_logging(f"async_langchain: {tag}, {input_dict}"))
    # Call the chain asynchronously
    response = await chain.ainvoke(input_dict)

    if verbose:
        print(sanitize_error_for_logging(f"async_langchain: {tag} response: {response}"))

    if label in input_dict:
        return response, tag, len(input_dict.get(label, ""))
    else:
        return response, tag


@retry(
    stop=stop_after_attempt(TENACITY_RETRY),  # Maximum 5 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: log(sanitize_error_for_logging(
        f"Attempt {retry_state.attempt_number}: {retry_state.outcome.exception()}, tag: {retry_state.args[1].get('tag', '')}")),
    reraise=True,  # Make sure to re-raise the final exception after all retries are exhausted
)
async def async_langchain_with_probs(chain: Any, input_dict: Dict[str, Any], tag: str = "", verbose: bool = False) -> Tuple[str, float, str]:
    #     async with sem:
    """
    similar to async_langchain, but expects chain from model.bind(logprobs=True) and a prompt for 1 token only
    returns tag token and prob of token returned
    call langchain asynchronously with ainvoke
    adds retry via tenacity decorator
    also adds a reference tag so if we gather 100 async responses we can match them up with the input
    """
    if verbose:
        print(sanitize_error_for_logging(f"async_langchain_with_probs: {tag}, {input_dict}"))
    # Call the chain asynchronously
    response = await chain.ainvoke(input_dict)

    if verbose:
        print(sanitize_error_for_logging(f"async_langchain_with_probs: {tag} response: {response}"))

    content = response.content
    logprob = response.response_metadata["logprobs"]["content"][0]
    prob = 2 ** logprob['logprob']

    return content, prob, tag


##############################################################################
# functions to process dataframes
##############################################################################
@retry(
    stop=stop_after_attempt(TENACITY_RETRY),  # Maximum 8 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: log(sanitize_error_for_logging(
        f"Retrying after {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}"))
)
async def filter_page_async(
    input_df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    output_class: Type[T],
    model: ChatOpenAI,
    input_vars: Optional[Dict[str, Any]] = None,
    opening_delimiter: str = "### <<<DATASET>>>",
    closing_delimiter: str = "### <<<END>>>###\nThink carefully, then respond with the JSON only.",
) -> T:
    """
    Process a single dataframe asynchronously.
    apply input_prompt to input_df converted to JSON per output_class type schema,
    supplying additional input_vars, and returning a variable of type output_class
    TODO: call async_langchain and just use the retry decorator there, add structurd output option
    """
    # print(user_prompt)
    if opening_delimiter:
        user_prompt += "\n" + opening_delimiter + "\n"
    user_prompt += "{input_text}"
    if closing_delimiter:
        user_prompt += "\n" + closing_delimiter
    # print(user_prompt)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    # Create the chain
    chain = prompt_template | model.with_structured_output(output_class)

    # Convert DataFrame to JSON
    input_text = input_df.to_json(orient='records', indent=2)
    input_dict = {"input_text": input_text}
    if input_vars is not None:
        input_dict.update(input_vars)
    # input_prompt = prompt_template.format_messages(**input_dict)
    # print(input_prompt)
    # print(input_text)

    try:
        response = await chain.ainvoke(input_dict)
    except asyncio.TimeoutError as e:
        log(f"Timeout error in filter_page_async: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        log(f"Network/timeout error in filter_page_async: {str(e)}")
        raise
    except ValueError as e:
        log(f"Invalid data in filter_page_async: {str(e)}")
        raise
    except Exception as e:
        log(f"Unexpected error in filter_page_async: {str(e)}")
        raise

    return response  # Should be an instance of output_class


@retry(
    stop=stop_after_attempt(TENACITY_RETRY),  # Maximum 5 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: log(sanitize_error_for_logging(
        f"Retrying after {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}")),
    reraise=True,  # Make sure to re-raise the final exception
)
async def filter_page_async_id(
    input_df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    output_class: Type[T],
    model: ChatOpenAI,
    input_vars: Optional[Dict[str, Any]] = None,
    item_list_field: str = "items",
    item_id_field: str = "id",
    opening_delimiter: str = "### <<<DATASET>>>",
    closing_delimiter: str = "### <<<END>>>\nThink carefully, then respond with the JSON only.",
) -> T:
    """
    similar to filter_page_async but checks ids in the response
    Process a single dataframe asynchronously.
    apply system_prompt to input_df converted to JSON per output_class type schema,
    supplying additional input_vars, and returning a variable of type output_class
    """
    if opening_delimiter:
        user_prompt += "\n" + opening_delimiter + "\n"
    user_prompt += "{input_text}"
    if closing_delimiter:
        user_prompt += "\n" + closing_delimiter

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    # Create the chain
    chain = prompt_template | model.with_structured_output(output_class)

    # Convert DataFrame to JSON
    input_text = input_df.to_json(orient='records', indent=2)
    input_dict = {"input_text": input_text}
    if input_vars is not None:
        input_dict.update(input_vars)
    # input_prompt = prompt_template.format_messages(**input_dict)
    # print(input_prompt)
    # print(input_text)

    try:
        response = await chain.ainvoke(input_dict)
    except asyncio.TimeoutError as e:
        log(f"Timeout error in filter_page_async_id: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        log(f"Network/timeout error in filter_page_async_id: {str(e)}")
        raise
    except ValueError as e:
        log(f"Invalid data in filter_page_async_id: {str(e)}")
        raise
    except Exception as e:
        log(f"Unexpected error in filter_page_async_id: {str(e)}")
        raise

    # check ids in response
    if item_list_field:
        if hasattr(response, item_list_field):
            response_list = getattr(response, item_list_field)
            if item_id_field:
                sent_ids = input_df[item_id_field].to_list()

                received_ids = [getattr(r, item_id_field)
                                for r in response_list]
                difference = set(sent_ids) - set(received_ids)
                if difference:
                    log(sanitize_error_for_logging(f"missing items: {str(difference)}"))
                    raise ValueError(
                        f"No {item_id_field} found in the results")

    # print(response)
    return response  # instance of output_class


async def process_dataframes(dataframes: List[pd.DataFrame],
                             system_prompt: str,
                             user_prompt: str,
                             output_class: Type[T],
                             model: ChatOpenAI,
                             input_vars: Optional[Dict[str, Any]] = None,
                             item_list_field: str = "items",
                             item_id_field: str = "",
                             ) -> Union[List[Any], None]:
    """
    Process multiple dataframes asynchronously.
    if item_list_field is provided, flatten the results
    if item_id_field is provided, check returned ids match sent ids

    Args:
        dataframes: List of dataframes to process
        input_prompt: The prompt template to use
        output_class: The output class for structured output
        model: The language model to use
        batch_size: Number of concurrent tasks to run

    Returns:
        List of processed results

    will fire off dataframes without any delay but with retry
    we paginate so we don't have to retry 200 if one fails
    this may run into tokens per minute rate limits, will error and retry
    could be smarter and check rate limits first, not easily available in langchain
    ideally if it fails on rate limit, check when it resets and wait til then
    """
    if item_id_field:
        tasks = [
            filter_page_async_id(df, system_prompt, user_prompt, output_class,
                                 model, input_vars, item_list_field, item_id_field)
            for df in dataframes
        ]
    else:
        tasks = [
            filter_page_async(df, system_prompt, user_prompt,
                              output_class, model, input_vars)
            for df in dataframes
        ]

    results = await asyncio.gather(*tasks)

    # if each result is an object with items as a list of objects then flatten
    flat_list = []
    if item_list_field:
        for result in results:
            if hasattr(result, item_list_field):
                flat_list.extend(getattr(result, item_list_field))
            else:
                raise ValueError(sanitize_error_for_logging(f"No {item_list_field} found in the results"))
        if item_id_field:   # check ids if provided
            sent_ids = [
                item_id for df in dataframes for item_id in df[item_id_field].to_list()]
            received_ids = [getattr(r, item_id_field) for r in flat_list]
            difference = set(sent_ids) - set(received_ids)
            if difference:
                log(f"missing {item_id_field}, {str(difference)}")
                raise ValueError(
                    f"missing {item_id_field} items not found in the results")
        return flat_list
    else:
        log(f"no {item_list_field} in result, returning raw results")


def filter_df(aidf: pd.DataFrame,
              model: BaseLanguageModel,
              system_prompt: str, user_prompt: str,
              output_column: str, input_column: str, input_column_rename: str = "",
              output_class: Type[T] = StoryRatings,
              output_class_label: str = "rating",
              mapper_func: Optional[Callable] = None,
              batch_size: int = 50) -> pd.DataFrame:
    """
    Generic filter for a dataframe using a prompt.

    Given a dataframe of article information, create a new column with the
    output of a prompt. The dataframe is paginated and processed
    asynchronously. The entire dataframe (or page) is sent to the model at once.
    The prompt should expect a dataframe with id and input_column(_rename),
    and return a dataframe with id and output_column.
    Returns an object of type T (with a list of responses for each row)

    Args:
        aidf (pd.DataFrame): The DataFrame containing article information.
        model: The language model to use
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        output_column (str): The name of the new column to create
        input_column (str): The name of the column containing the text to apply the prompt to
        input_column_rename (str, optional): The name to give the input column. Defaults to "".
        output_class (Type[T], optional): The class to use for the output. Defaults to StoryRatings.
        mapper_func (Callable, optional): A function to apply to the input column. Defaults to None.
        batch_size (int, optional): The number of rows to process at once. Defaults to 50.

    Returns:
        pd.DataFrame: the updated aidf, a DataFrame with the output of the prompt.
    """
    log(f"Starting {output_column} filter")
    qdf = aidf[['id', input_column]].copy()
    if input_column_rename:
        qdf = qdf.rename(columns={input_column: input_column_rename})
        input_column = input_column_rename
    if mapper_func:
        qdf[input_column] = qdf[input_column].apply(
            mapper_func, axis=1)
    pages = paginate_df(qdf, maxpagelen=batch_size)
    responses = asyncio.run(process_dataframes(
        pages,
        system_prompt,
        user_prompt,
        output_class,
        model=model,
    ))
    response_dict = {resp.id: getattr(
        resp, output_class_label) for resp in responses}
    aidf[output_column] = aidf["id"].map(response_dict.get)
    return aidf


@retry(
    stop=stop_after_attempt(TENACITY_RETRY),  # Maximum 5 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: log(
        f"Retrying after {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}")
)
async def filter_df_rows(aidf: pd.DataFrame,
                         model: BaseLanguageModel,
                         system_prompt: str,
                         user_prompt: str,
                         output_column: str,
                         input_column: str,
                         input_column_rename: str = "",
                         output_class: Type[T] = StoryRatings,
                         output_class_label: str = "rating",
                         mapper_func: Optional[Callable] = None) -> pd.DataFrame:
    """
    Generic filter for a dataframe using a prompt.

    Given a dataframe of article information, create a new column with the
    output of a prompt. Rows are sent indidividually. The prompt should expect a
    string and return a single object of type T.

    this has not been tested or used yet, but initial implementation to parallel filter_df

    Args:
        aidf (pd.DataFrame): The DataFrame containing article information.
        model: The language model to use
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        output_column (str): The name of the new column to create
        input_column (str): The name of the column containing the text to apply the prompt to
        input_column_rename (str, optional): The name to give the input column. Defaults to "".
        output_class (Type[T], optional): The class to use for the output. Defaults to StoryRatings.
        mapper_func (Callable, optional): A function to apply to the input column. Defaults to None.

    Returns:
        Dict[str, int]: A dictionary mapping article id to the output of the prompt.
    """
    log(f"Starting {output_column} filter")
    qdf = aidf[['id', input_column]].copy()
    if input_column_rename:
        qdf = qdf.rename(columns={input_column: input_column_rename})
    else:
        input_column_rename = input_column
    if mapper_func:
        qdf[input_column_rename] = qdf[input_column_rename].apply(
            mapper_func, axis=1)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
         ("user", user_prompt)]
    )
    chain = prompt_template | model.with_structured_output(output_class)

    tasks = []
    for row in qdf.itertuples():
        input_str = getattr(row, input_column_rename)
        log(f"Queuing {row.id}: {input_str[:50]}...")
        task = asyncio.create_task(async_langchain(
            chain, {"input_text": input_str}, tag=row.id, verbose=False))
        tasks.append(task)

    try:
        log(f"Fetching responses for {len(tasks)} articles")
        responses = await asyncio.gather(*tasks)
        log(f"Received {len(responses)} responses")
    except Exception as e:
        log(f"Error fetching responses: {str(e)}")
    response_dict = {rowid: getattr(
        resp, output_class_label) for resp, rowid in responses}

    aidf[output_column] = aidf["id"].map(response_dict.get)

    return aidf


async def fetch_all_summaries(aidf: pd.DataFrame, model: BaseLanguageModel) -> List[Tuple[str, str, int]]:
    """
    Fetch summaries for all articles in the AIdf DataFrame.

    This function processes each row in the AIdf, extracts the article
    content using the path column, and generates a summary using the MODEL LLM.
    Summaries are fetched asynchronously.

    Args:
        AIdf (pd.DataFrame): A DataFrame containing article information. Each row should
                             have 'path' and 'id' attributes.

    Returns:
        list: A list of summaries for each article.

    Raises:
        Exception: If there is an error during the asynchronous processing of the articles.



    """

    log("Fetching summaries for all articles")
    tasks = []

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SUMMARIZE_SYSTEM_PROMPT),
            ("user", SUMMARIZE_USER_PROMPT)]
    )
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    log(f"Attempting to fetch summaries for {len(aidf)} articles")

    count_valid, count_no_path, count_no_content = 0, 0, 0
    for row in aidf.itertuples():
        text_path, rowid = row.text_path, row.id
        article_str = ""
        if not text_path:
            count_no_path += 1
            log(f"No text_path for {rowid}")
            continue

        # check if path exists
        if not os.path.exists(text_path):
            log(f"Invalid path for {rowid}")
            count_no_path += 1
            continue

        # read the file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                article_str = f.read()
        except FileNotFoundError as e:
            log(f"File not found {text_path}: {str(e)}")
            count_no_content += 1
            continue
        except PermissionError as e:
            log(f"Permission denied reading {text_path}: {str(e)}")
            count_no_content += 1
            continue
        except UnicodeDecodeError as e:
            log(f"Encoding error reading {text_path}: {str(e)}")
            count_no_content += 1
            continue

        if not article_str:
            count_no_content += 1
            continue

        count_valid += 1
        log(f"Queuing {rowid}: {article_str[:50]}...")
        task = asyncio.create_task(async_langchain(
            chain, {"article": article_str}, tag=rowid, verbose=False))
        tasks.append(task)

    log(f"{count_valid} valid articles, {count_no_path} no path, {count_no_content} no content")

    try:
        log(f"Fetching summaries for {len(tasks)} articles")
        responses = await asyncio.gather(*tasks)
        log(f"Received {len(responses)} summaries")
        for summary, rowid, article_len in responses:
            log(f"Summary for {rowid} (length {article_len}): {summary}")
    except asyncio.TimeoutError as e:
        log(f"Timeout error fetching summaries: {str(e)}")
    except ConnectionError as e:
        log(f"Network connection error fetching summaries: {str(e)}")
    except Exception as e:
        log(f"Unexpected error fetching summaries: {str(e)}")

    return responses


async def get_canonical_topic_results(pages: List[pd.DataFrame], topic: str, model_low: BaseLanguageModel) -> Tuple[str, List[Any]]:
    """call CANONICAL_TOPIC_PROMPT on pages for a single topic"""
    retval = await process_dataframes(dataframes=pages,
                                      system_prompt=CANONICAL_SYSTEM_PROMPT,
                                      user_prompt=CANONICAL_USER_PROMPT,
                                      output_class=CanonicalTopicSpecList,
                                      model=model_low,
                                      input_vars={'topic': topic})
    return topic, retval


async def get_all_canonical_topic_results(pages: List[pd.DataFrame], topics: List[str], model_medium: BaseLanguageModel) -> List[Tuple[str, List[Any]]]:
    """call all topics on pages"""
    tasks = []
    for topic in topics:
        log(f"Canonical topic {topic}")
        tasks.append(get_canonical_topic_results(pages, topic, model_medium))
    log(f"Sending prompt for {len(tasks)} canonical topics")
    results = await asyncio.gather(*tasks)
    return results


def clean_topics(row: Any) -> List[str]:
    """
    Cleans the extracted_topics and assigned_topics by removing certain common topics and combining them into a single list.

    Args:
        row (pandas.Series): The row containing the extracted_topics and assigned_topics.
        lcategories (set): The set of lowercase categories.

    Returns:
        list: The cleaned and combined list of topics.

    TODO: could send more concurrent prompts to gpt-4o-mini and not hit rate limits

    """
    extracted_topics = [x.title() for x in row.extracted_topics if x.lower() not in {
        "technology", "ai", "artificial intelligence", "gen ai", "no content"}]
    assigned_topics = [x.title()
                       for x in row.assigned_topics]
    combined = sorted(list(set(extracted_topics + assigned_topics)))
    combined = [s.replace("Genai", "Gen AI") for s in combined]
    combined = [s.replace("Openai", "OpenAI") for s in combined]
    combined = [s.replace("Ai", "AI") for s in combined]

    return combined


async def filter_df_rows_with_probability(aidf: pd.DataFrame,
                                          model: BaseChatModel,
                                          system_prompt: str,
                                          user_prompt: str,
                                          output_column: str,
                                          input_column: str,
                                          input_column_rename: str = "",
                                          mapper_func: Optional[Callable] = None) -> pd.DataFrame:
    """
    Generic filter for a dataframe using a prompt that returns 1/0 probability.

    Given a dataframe of article information, create a new column with the
    probability of "1" from a prompt. Rows are sent individually. The prompt should
    return either "1" or "0" as a single token, and this function extracts the
    probability of the "1" token.

    Args:
        aidf (pd.DataFrame): The DataFrame containing article information.
        model: The ChatAnthropic language model that supports returning probabilities
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        output_column (str): The name of the new column to create
        input_column (str): The name of the column containing the text to apply the prompt to
        input_column_rename (str, optional): The name to give the input column. Defaults to "".
        mapper_func (Callable, optional): A function to apply to the input column. Defaults to None.

    Returns:
        pd.DataFrame: The original DataFrame with a new column containing 1/0 probabilities.
    """
    log(f"Starting {output_column} probability filter")
    qdf = aidf[['id', input_column]].copy()
    if input_column_rename:
        qdf = qdf.rename(columns={input_column: input_column_rename})
    else:
        input_column_rename = input_column
    if mapper_func:
        qdf[input_column_rename] = qdf[input_column_rename].apply(
            mapper_func, axis=1)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
         ("user", user_prompt)]
    )

    # Configure model to return logprobs for probability extraction
    chain = prompt_template | model.bind(logprobs=True, top_logprobs=1)
    tasks = []
    for row in qdf.itertuples():
        input_str = getattr(row, input_column_rename)
        if not input_str:
            log(f"No input for {row.id}")
            continue
        log(f"Queuing {row.id}: {input_str[:50]}...")
        task = asyncio.create_task(async_langchain_with_probs(
            chain, {"input_text": input_str}, tag=row.id, verbose=False))
        tasks.append(task)

    try:
        log(f"Fetching responses for {len(tasks)} articles")
        responses = await asyncio.gather(*tasks)
        log(f"Received {len(responses)} responses")
    except asyncio.TimeoutError as e:
        log(f"Timeout error fetching probability responses: {str(e)}")
    except ConnectionError as e:
        log(f"Network connection error fetching probability responses: {str(e)}")
    except Exception as e:
        log(f"Unexpected error fetching probability responses: {str(e)}")

    # Extract response, probabilities from responses
    response_dict = {}
    for response, prob, rowid in responses:
        if response == "1":
            response_dict[rowid] = min(1.0, round(prob, 2))
        else:
            response_dict[rowid] = max(0.0, round(1 - prob, 2))

    aidf[output_column] = aidf["id"].map(response_dict.get)

    return aidf

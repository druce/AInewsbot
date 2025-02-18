# Description: LLM functions for AInewsbot project
# import os
# import time
# import pdb

import json
# from collections import defaultdict
import math
# import aiohttp
import asyncio
from typing import List, Type, TypeVar, Dict, Any  # , TypedDict, Annotated,
from pathlib import Path

from pydantic import BaseModel, Field
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

import tiktoken
# import openai
# from openai import OpenAI
from bs4 import BeautifulSoup
import trafilatura


# import langchain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
# from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, HumanMessage, ToolMessage
# MessagesPlaceholder, PromptTemplate,
from langchain_core.prompts import (ChatPromptTemplate,)
# SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from ainb_utilities import log
from ainb_const import (LOWCOST_MODEL, BASEMODEL, MODEL, MAX_INPUT_TOKENS,
                        CANONICAL_TOPIC_PROMPT,
                        SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_PROMPT)

##############################################################################
# pydantic classes used to get structured outputs from LLM
##############################################################################

# so we can pass types to functions
T = TypeVar('T', bound=BaseModel)


class Story(BaseModel):
    """Story class for structured output filtering"""
    id: int = Field(description="The id of the story")
    isAI: bool = Field(description="true if the story is about AI, else false")


class Stories(BaseModel):
    """Stories class for structured output filtering of a list of Story"""
    items: List[Story] = Field(description="List of Story")

# TODO: shouldn't really have to define a second class as just a list of the first
# but kind of have to do this to get a list schema to send to LangChain
# could possibly do this within the functions that take the type as argument
# class ItemList(BaseModel):
#     __root__: List[Story] = Field(description="List of Story")
# or just past List[Story] as the type to the LangChain call


class TopicSpec(BaseModel):
    """TopicSpec class for structured output of story topics"""
    id: int = Field(description="The id of the story")
    extracted_topics: List[str] = Field(
        description="List of topics covered in the story")


class TopicSpecList(BaseModel):
    """List of TopicSpec class for structured output"""
    items: List[TopicSpec] = Field(description="List of TopicSpec")


class CanonicalTopicSpec(BaseModel):
    """CanonicalTopicSpec class for structured output of canonical topics"""
    id: int = Field(description="The id of the story")
    relevant: bool = Field(
        description="True if the story is about the topic else false")


class CanonicalTopicSpecList(BaseModel):
    """List of CanonicalTopicSpec for structured output"""
    items: List[CanonicalTopicSpec] = Field(
        description="List of CanonicalTopicSpec")


class TopicHeadline(BaseModel):
    """Topic headline of a group of stories for structured output"""
    topic_title: str = Field(description="The title for the headline group")


class TopicCategoryList(BaseModel):
    """List of topics for structured output filtering"""
    items: List[str] = Field(description="List of topics")


class Site(BaseModel):
    """Site class for structured output filtering"""
    url: str = Field(description="The URL of the site")
    site_name: str = Field(description="The name of the site")


class Sites(BaseModel):
    """List of Site class for structured output filtering"""
    items: List[Site] = Field(description="List of Site")

##############################################################################
# utility functions
##############################################################################


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


def trunc_tokens(long_prompt, model=BASEMODEL, maxtokens=MAX_INPUT_TOKENS):
    """return prompt string, truncated to maxtokens"""
    # Initialize the encoding for the model you are using, e.g., 'gpt-4'
    encoding = tiktoken.encoding_for_model(model)

    # Encode the prompt into tokens, truncate, and return decoded prompt
    tokens = encoding.encode(long_prompt)
    tokens = tokens[:maxtokens]
    truncated_prompt = encoding.decode(tokens)

    return truncated_prompt


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


def should_retry_exception(exception):
    """Determine if the exception should trigger a retry. (always retry)"""
    print(type(exception))
    print(exception)
    return True


##############################################################################
# basic langchain call async function with decorators
##############################################################################

# CALLS_PER_MINUTE = 60  # Adjust based on your quota
# MAX_CONCURRENT = 5     # Maximum concurrent requests
# sem = Semaphore(MAX_CONCURRENT) # semaphore for controlling concurrent requests

# @sleep_and_retry
# @limits(calls=CALLS_PER_MINUTE, period=60)


@retry(
    stop=stop_after_attempt(8),  # Maximum 8 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(should_retry_exception),
    before_sleep=lambda retry_state: log(
        f"Retrying after {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}")
)
async def async_langchain(chain, input_dict, name=""):
    #     async with sem:
    """call langchain asynchronously with ainvoke"""
    response = await chain.ainvoke(input_dict)
    return response, name


##############################################################################
# functions to process dataframes
##############################################################################

def filter_page(input_df: pd.DataFrame,
                input_prompt: str,
                output_class: Type[T],
                model: ChatOpenAI = ChatOpenAI(model=LOWCOST_MODEL),
                input_vars: Dict[str, Any] = None
                ) -> T:
    """
    Process a single dataframe synchronously.
    apply input_prompt to input_df converted to JSON per output_class type schema,
    supplying additional input_vars, and returning output_class
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", input_prompt),
        ("user", "{input_text}")
    ])

    # Create the chain
    chain = prompt_template | model.with_structured_output(output_class)

    # Run the chain
    input_text = json.dumps(input_df.to_dict(orient='records'), indent=2)
    input_dict = {"input_text": input_text}
    if input_vars is not None:
        input_dict.update(input_vars)
    # unpack input_dict to kwargs
    response = chain.invoke(input_dict)

    return response


@retry(
    stop=stop_after_attempt(8),  # Maximum 8 attempts
    # Wait 2^x * multiplier seconds between retries
    wait=wait_exponential(multiplier=1, min=2, max=128),
    retry=retry_if_exception_type(should_retry_exception),
    before_sleep=lambda retry_state: log(
        f"Retrying after {retry_state.outcome.exception()}, attempt {retry_state.attempt_number}")
)
async def filter_page_async(
    input_df: pd.DataFrame,
    input_prompt: str,
    output_class: Type[T],
    model: ChatOpenAI = ChatOpenAI(model=LOWCOST_MODEL),
    input_vars: Dict[str, Any] = None
) -> T:
    """
    Process a single dataframe asynchronously.
    apply input_prompt to input_df converted to JSON per output_class type schema,
    supplying additional input_vars, and returning a variable of type output_class
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", input_prompt),
        ("user", "{input_text}")
    ])

    # Create the chain
    chain = prompt_template | model.with_structured_output(output_class)

    # Convert DataFrame to JSON
    input_text = json.dumps(input_df.to_dict(orient='records'), indent=2)
    input_dict = {"input_text": input_text}
    if input_vars is not None:
        input_dict.update(input_vars)

    # Call the chain asynchronously
    response = await chain.ainvoke(input_dict)

    return response  # Should be an instance of output_class


async def process_dataframes(dataframes: List[pd.DataFrame],
                             input_prompt: str,
                             output_class: Type[T],
                             model: ChatOpenAI = ChatOpenAI(
                                 model=LOWCOST_MODEL),
                             input_vars: Dict[str, Any] = None
                             ) -> T:
    """
    Process multiple dataframes asynchronously.

    Args:
        dataframes: List of dataframes to process
        input_prompt: The prompt template to use
        output_class: The output class for structured output
        model: The language model to use
        batch_size: Number of concurrent tasks to run

    Returns:
        List of processed results
    """

    tasks = [
        filter_page_async(df, input_prompt, output_class,
                          model, input_vars)
        for df in dataframes
    ]

    results = await asyncio.gather(*tasks)

    # if each result is an object with items as a list of objects then flatten
    flat_list = []
    for result in results:
        if hasattr(result, 'items'):
            flat_list.extend(result.items)
        else:
            break
    if len(flat_list) == 0:
        return results
    else:
        return flat_list


def clean_html(path: Path | str) -> str:
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
    with open(path, 'r', encoding='utf-8') as file:
        html_content = file.read()

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
        return visible_text


async def fetch_all_summaries(aidf):
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

    TODO: can make a more generic version
    pass in prompt, model, output class
    pass in df with id and value(s)
    pass in optional function dict to call on each column
    get the column names, apply function if provided, apply template using those names

    """
    tasks = []

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", SUMMARIZE_SYSTEM_PROMPT),
         ("user", SUMMARIZE_USER_PROMPT)]
    )

    openai_model = ChatOpenAI(model=MODEL)
    parser = StrOutputParser()
    chain = prompt_template | openai_model | parser

    for row in aidf.itertuples():
        path, rowid = row.path, row.id
        article_str = clean_html(path)
        task = asyncio.create_task(async_langchain(
            chain, {"article": article_str}, name=rowid))
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return responses


async def get_canonical_topic_results(pages, topic):
    """call CANONICAL_TOPIC_PROMPT on pages for a single topic"""
    retval = await process_dataframes(dataframes=pages,
                                      input_prompt=CANONICAL_TOPIC_PROMPT,
                                      output_class=CanonicalTopicSpecList,
                                      model=ChatOpenAI(model=LOWCOST_MODEL),
                                      input_vars={'topic': topic})
    return topic, retval


async def get_all_canonical_topic_results(pages, topics):
    """call all topics on pages"""
    tasks = []
    for topic in topics:
        log(f"Canonical topic {topic}")
        tasks.append(get_canonical_topic_results(pages, topic))
    log(f"Sending prompt for {len(tasks)} canonical topics")
    results = await asyncio.gather(*tasks)
    return results


def clean_topics(row, lcategories):
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
                       for x in row.assigned_topics if x.lower() in lcategories]
    combined = sorted(list(set(extracted_topics + assigned_topics)))
    combined = [s.replace("Ai", "AI") for s in combined]
    combined = [s.replace("Genai", "Gen AI") for s in combined]
    combined = [s.replace("Openai", "OpenAI") for s in combined]

    return combined

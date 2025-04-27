"""
for debugging, some synchronous versions of ainb_llm functions
can call these from top level with pdb.set_trace, and step through
unlike async versions which will spawn many threads and not allow pdb to work
"""
import pandas as pd
from typing import Type, TypeVar, Dict, Any
from langchain_openai import ChatOpenAI

T = TypeVar('T')


def sfilter_page_async(
    input_df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    output_class: Type[T],
    model: ChatOpenAI,
    input_vars: Dict[str, Any] = None,
    opening_delimiter: str = "### <<<DATASET>>>",
    closing_delimiter: str = "### <<<END>>>###\nThink silently, then respond with the JSON only.",
) -> T:
    """
    Process a single dataframe asynchronously.
    apply input_prompt to input_df converted to JSON per output_class type schema,
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
    pdb.set_trace()
    input_text = input_df.to_json(orient='records', indent=2)
    input_dict = {"input_text": input_text}
    if input_vars is not None:
        input_dict.update(input_vars)
    # input_prompt = prompt_template.format_messages(**input_dict)
    # print(input_prompt)
    # print(input_text)

    # Call the chain asynchronously
    response = chain.invoke(input_dict)
    # print(response)

    return response  # Should be an instance of output_class


def sfilter_page_async_id(
    input_df: pd.DataFrame,
    system_prompt: str,
    user_prompt: str,
    output_class: Type[T],
    model: ChatOpenAI,
    input_vars: Dict[str, Any] = None,
    item_list_field: str = "items",
    item_id_field: str = "id",
    opening_delimiter: str = "### <<<DATASET>>>",
    closing_delimiter: str = "### <<<END>>>\nThink silently, then respond with the JSON only.",
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

    # Call the chain asynchronously
    response = chain.invoke(input_dict)
    # response = chain.invoke(input_dict)

    # check ids in response
    if item_list_field:
        if hasattr(response, item_list_field):
            response_list = getattr(response, item_list_field)
            if item_id_field:
                sent_ids = [
                    item_id for item_id in input_df[item_id_field].to_list()]

                received_ids = [getattr(r, item_id_field)
                                for r in response_list]
                difference = set(sent_ids) - set(received_ids)
                if difference:
                    log(f"missing items: {str(difference)}")
                    raise ValueError(
                        f"No {item_id_field} found in the results")

    # print(response)
    return response  # instance of output_class


def sprocess_dataframes(dataframes: List[pd.DataFrame],
                        system_prompt: str,
                        user_prompt: str,
                        output_class: Type[T],
                        model: ChatOpenAI,
                        input_vars: Dict[str, Any] = None,
                        item_list_field: str = "items",
                        item_id_field: str = "",
                        ) -> T:
    """
    Process multiple dataframes asynchronously.
    if item_list_field is provided, flatten the results
    if item_id_field is proided, check returned ids match sent ids

    Args:
        dataframes: List of dataframes to process
        input_prompt: The prompt template to use
        output_class: The output class for structured output
        model: The language model to use
        batch_size: Number of concurrent tasks to run

    Returns:
        List of processed results
    """
    if item_id_field:
        tasks = [
            sfilter_page_async_id(df, system_prompt, user_prompt, output_class,
                                  model, input_vars, item_list_field, item_id_field)
            for df in dataframes
        ]
    else:
        tasks = [
            sfilter_page_async(df, system_prompt, user_prompt, output_class,
                               model, input_vars)
            for df in dataframes
        ]

    results = tasks

    # if each result is an object with items as a list of objects then flatten
    flat_list = []
    if item_list_field:
        for result in results:
            if hasattr(result, item_list_field):
                flat_list.extend(getattr(result, item_list_field))
            else:
                raise ValueError(f"No {item_list_field} found in the results")
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


def sget_canonical_topic_results(pages, topic, model_low):
    """call CANONICAL_TOPIC_PROMPT on pages for a single topic"""
    retval = sprocess_dataframes(dataframes=pages,
                                 system_prompt=CANONICAL_SYSTEM_PROMPT,
                                 user_prompt=CANONICAL_USER_PROMPT,
                                 output_class=CanonicalTopicSpecList,
                                 model=model_low,
                                 input_vars={'topic': topic})
    return topic, retval


def sget_all_canonical_topic_results(pages, topics, model_medium):
    """call all topics on pages"""
    tasks = []
    for topic in topics:
        log(f"Canonical topic {topic}")
        tasks.append(sget_canonical_topic_results(pages, topic, model_medium))
    log(f"Sending prompt for {len(tasks)} canonical topics")
    results = tasks
    # print(results)

    return results

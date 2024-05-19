# tool class that is callable and also has metadata for building the prompt
# either give it the OpenBB path from the openapi.json
# or give it a function that returns a dict with results element that contains a list of return values

# for most people I recommend langchain https://python.langchain.com/v0.1/docs/modules/agents/concepts/
# I wanted to see how this works at a granular level
# langchain abstracts some concepts and lets you switch LLMs or types of agents easily
# but it adds a layer of complexity, is a very thin abstraction, and hides what's going on

import json
from ainb_const import MODEL, bb_agent_system_prompt
from IPython.display import Markdown, display

import openai

# import openbb
from openbb import obb
# from openbb_core.app.model.obbject import OBBject


class BB_agent_tool(object):
    """
    A tool class that is callable and also has metadata for building the prompt.
    """

    # class variable keeping track of all agents created
    agent_registry = {}

    def __init__(self, name, description, openapi_path, parameters, example_parameter_values, callable=None, singular=0):
        """
        Initialize the BB_agent_tool instance.

        Parameters:
        - name (str): The name of the tool.
        - description (str): The description of the tool.
        - openapi_path (str): The OpenAPI path for the tool.
        - parameters (dict): The parameters for the tool.
        - example_parameter_values (list): The example parameter values for the tool.
        - callable (function, optional): The callable function for the tool. Defaults to None.
        - singular (int, optional): The singular value for the tool. Defaults to 0.
        """
        self.name = name
        self.description = description
        self.openapi_path = openapi_path
        self.parameters = parameters
        self.example_parameter_values = example_parameter_values
        self.singular = singular
        # needs either openapi_path or callable
        if callable:
            self.callable = callable
        else:
            # get callable via openapi_path
            self.callable = self.get_callable()
        self.tooldict = self.make_tooldict()
        self.example_code = self.make_example_code()
        # store self in agent_registry class variable that keeps track of all instances
        self.agent_registry[self.name] = self

    def __str__(self):
        """
        Return a string representation of the BB_agent_tool instance.
        """
        return (f"Tool Name:          {self.name}\n"
                f"Description:        {self.description}\n"
                f"OpenAPI Path:       {self.openapi_path}\n"
                f"Parameters:         {self.parameters}\n"
                f"Example Parameters: {self.example_parameter_values}\n"
                f"Singular:           {self.singular}\n"
                f"Tooldict:           {self.tooldict}\n"
                f"Examples:           {self.example_code}\n"
                f"Callable:           \n"
                f"{self.callable.__doc__}\n"
                )

    def __call__(self, **kwargs):
        """
        Make the instance callable and perform the default operation of calling the callable function on the kwargs.
        Return a JSON string with the return values, which can get passed back to OpenAI as part of the agent workflow.
        """
        retval = None
        try:
            # call the tool function
            obj = self.callable(**kwargs)
            # if custom tool, it returns a list
            if type(obj) is list:
                d = obj
            # OpenBB returns a rich object with some metadata, list of obb results
            elif obj and hasattr(obj, 'results'):
                # this may be a hack, obj.json() dumps to json, then loads to a dict, gets results list
                # could get results from obj which is a list of obb objects, so need to json them
                # could loop through results and run model_dump()
                d = json.loads(obj.json())['results']
            # singular should be 0 to return the full array, 1 for first element only, -1 for last element
            if self.singular:
                index = 0 if self.singular == 1 else -1
                retval = json.dumps(d[index])
            else:
                retval = json.dumps(d)
        except Exception as exc:
            print(exc)
        return str(retval)

    def get_callable(self):
        """
        Get the callable function based on the openapi_path.
        """
        op = obb
        for part in self.openapi_path.removeprefix("/api/v1/").split("/"):
            op = op.__getattribute__(part)
        return op

    def make_tooldict(self):
        """
        Create the representation of the tool that is shared with OpenAI.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    # all required for now
                    "required": [k for k in self.parameters.keys()]
                }
            }
        }

    def make_example_code(self):
        """
        Create the example code for the tool that can be included in the prompt.
        """
        examples = []
        for example in self.example_parameter_values:
            parray = []
            for k, v in example.items():
                # get type
                ptype = "string"
                if ptype == "string":
                    parray.append(f'{k}="{v}"')
                else:
                    parray.append(f'{k}={v}')
            pstr = ", ".join(parray)

            # run the example
            return_value = self(**example)
            if type(return_value) is list:  # only use 3 values if it's a long list
                return_value = str(return_value[:3])
            if type(return_value) is str:  # truncate long strings
                max_str_len = 2000
                return_value = return_value[:max_str_len] + \
                    "…" if len(return_value) > max_str_len else return_value

            examples.append(f"{self.name}({pstr}) -> {return_value}")
            return "; ".join(examples)


# utility functions to call chatgpt with tools
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


def get_response(client, messages, tools, model=MODEL, json_format=False):
    """
    Get a single response from ChatGPT based on a chain of messages.

    Args:
        client: The OpenAI client.
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


def get_response_and_eval(client, messages, tools=[], json_format=False, raw=False, verbose=False):
    """
    Sends a list of messages to OpenAI and returns the response.
    If tool calls are returned, calls all the tools and sends the values back to OpenAI.
    If further tool calls returned, iterates until no more tool calls are returned and
    'stop' is returned as finish_reason, then returns the response.

    Args:
        client: The OpenAI client.
        messages (list): A list of messages to send to OpenAI.
        json_format (boolean): If the final response should be in JSON format.
        raw (boolean): after last tool is called return raw data response that enabled answering the question
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        response: The final response object returned by OpenAI.

    Raises:
        None

    """
    response = get_response(
        client, messages, tools=tools, json_format=json_format)
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
                client, messages, tools=tools, json_format=json_format)
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


def agent_query(client, user_message, raw=False, verbose=True):
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
                client, messages, tools=openai_tools, raw=raw, verbose=verbose)
            response_str = response.choices[0].message.content
            # escape stuff that is interpreted as latex
            response_str = response_str.replace("$", "\\\$")
            display(Markdown(response_str))
            # success, exit retry loop
            break
        except Exception as exc:
            print(exc)


if __name__ == "__main__":

    # Example usage
    tool = BB_agent_tool(
        name="get_quote_json",
        description="Given a stock symbol, get the latest market data quote for the stock in JSON format.",
        openapi_path="/api/v1/equity/price/quote",
        parameters={
            "symbol": {
                "type": "string",
                "description": "The stock symbol."
            }
        },
        example_parameter_values=[{
            "symbol": "NVDA",
        }],
    )

    tools = tool.agent_registry

    tool(symbol="NVDA")

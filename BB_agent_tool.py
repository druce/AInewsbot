# tool class that is callable and also has metadata for building the prompt
# either give it the OpenBB path from the openapi.json
# or give it a function that returns a dict with results element that contains a list of return values

# for most people I recommend langchain https://python.langchain.com/v0.1/docs/modules/agents/concepts/
# I wanted to see how this works at a granular level
# langchain abstracts some concepts and lets you switch LLMs or types of agents easily
# but it adds a layer of complexity, is a very thin abstraction, and hides what's going on

import json

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
            # if custom tool, it returns a list, wrap it similarly to openbb return dict
            if type(obj) is list:
                d = obj
            # OpenBB returns objects
            elif obj and hasattr(obj, 'results'):
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

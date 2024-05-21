import os
import yaml

from langchain.tools import StructuredTool

from openbb import obb

# langchain 0.2 is still on pydantic v1
from pydantic.v1 import BaseModel, Field  # <-- Uses v1 namespace

# free API for edgar filing
import sec_parser as sp
from sec_downloader import Downloader

CONFIG_YAML = "obbtools.yaml"
tool_dict = {}


class SymbolLimitSchema(BaseModel):
    symbol: str
    limit: int


class SymbolSchema(BaseModel):
    symbol: str


class QuerySchema(BaseModel):
    query: str = Field(
        description="The search string to match to the stock symbol.")
    # limit not supported for provider=sec
    # limit: int = Field(description="The maximum number of values to return")


schema_dict = {
    "SymbolSchema": SymbolSchema,
    "QuerySchema": QuerySchema,
    "SymbolLimitSchema": SymbolLimitSchema,
}


def fn_get_10k_item1_from_symbol(symbol):
    """
    Get item 1 of the latest 10-K annual report filing for a given symbol.

    Args:
        symbol (str): The symbol of the equity.

    Returns:
        str: The item 1 of the latest 10-K annual report filing, or None if not found.

    """
    item1_text = None
    try:
        # sec needs you to identify yourself for rate limiting
        dl = Downloader(os.getenv("SEC_FIRM"), os.getenv("SEC_USER"))
        html = dl.get_filing_html(ticker=symbol, form="10-K")
        elements: list = sp.Edgar10QParser().parse(html)
        tree = sp.TreeBuilder().build(elements)
        sections = [n for n in tree.nodes if n.text.startswith("Item")]
        item1_node = sections[0]
        item1_text = "\n".join([n.text for n in item1_node.get_descendants()])
    except Exception as e:
        print(e)
        return None
    # always return a list of dicts
    return [{'item1': item1_text}]


get_10k_item1_from_symbol = StructuredTool.from_function(
    func=fn_get_10k_item1_from_symbol,
    name="get_10k_item1_from_symbol",
    description="Given a stock symbol, gets item 1 of the company's latest 10-K annual report filing.",
    args_schema=SymbolSchema
)

tool_dict["get_10k_item1_from_symbol"] = get_10k_item1_from_symbol


def obb_function_factory(fn_metadata):
    """
    Creates a function that wraps the appropriate openbb method based on a metadata dict .

    Args:
        fn_metadata (dict): The metadata of the function.

    Returns:
        function: The generated function.

    """
    # get correct obb method object based on path, e.g. /api/v1/equity/price/quote -> obb.equity.price.quote
    parts = fn_metadata["openapi_path"].removeprefix("/api/v1/").split("/")
    op = obb
    for part in parts:
        op = op.__getattribute__(part)
    singular = fn_metadata.get("singular", 0)
    default_args = fn_metadata.get("default_parameters", {})
    override_args = fn_metadata.get("override_parameters", {})

    # return a closure based on value of op etc.
    def tool_fn(**kwargs):
        """call op and return results without obb metadata"""
        # always use any override args
        for k, v in override_args.items():
            kwargs[k] = v
        # use default arg if not already present
        for k, v in default_args.items():
            if k not in kwargs:
                kwargs[k] = v
        retobj = op(**kwargs)
        retlist = [r.model_dump_json() for r in retobj.results]
        if len(retlist):
            if singular == 1:
                # return first
                return retlist[0]
            elif singular == -1:
                # return last
                return retlist[-1]
        # no retlist or not singular
        return retlist

    return tool_fn


# Load obb tool defs from YAML file
with open(CONFIG_YAML, 'r') as file:
    fn_yaml_list = yaml.safe_load(file)

# create tools from metadata
for fn_metadata in fn_yaml_list:
    tool_name = fn_metadata["name"]
    tool_desc = fn_metadata["description"]
    tool_schema_name = fn_metadata["args_schema"]
    print(f"Creating tool {tool_name}")
    # instantiate tool and add to tool_dict
    tool_dict[tool_name] = StructuredTool.from_function(
        func=obb_function_factory(fn_metadata),
        name=tool_name,
        description=tool_desc,
        args_schema=schema_dict[tool_schema_name]
    )

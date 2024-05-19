import os
from BB_agent_tool import BB_agent_tool

# free API for edgar filing
from sec_downloader import Downloader
import sec_parser as sp

from langchain.tools import StructuredTool


def make_lc_tool(f, bbtool, verbose=True):
    desc = f"{bbtool.description}"
    if bbtool.example_code:
        desc += f" Usage: {bbtool.example_code}"
    lc_tool = StructuredTool.from_function(
        func=f,
        name=bbtool.name,
        description=desc,
        handle_tool_error=True,
    )
    # truncate to max allowed desc length
    lc_tool.description = lc_tool.description[:1023]
    print(f"Name:          {lc_tool.name}")
    print(f"Description:   {lc_tool.description}")
    print(f"Return direct: {lc_tool.return_direct}")
    print(f"Args:\n{lc_tool.args}")
    return lc_tool


def get_10k_item1_from_symbol(symbol):
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
        item1_text = "\n".join([n.text for n in sections[0].get_descendants()])
    except Exception as e:
        return str(e)
    # always return a list of dicts
    return [{'item1': item1_text}]


fn_metadata = {
    "name": "get_10k_item1_from_symbol",
    "description": "Given a stock symbol, gets item 1 of the company's latest 10-K annual report filing.",
    "openapi_path": None,
    "callable": get_10k_item1_from_symbol,
    "parameters": {
        "symbol": {
            "type": "string",
            "description": "The symbol to get the 10-K item 1 for"
        }
    },
    "example_parameter_values": [{
        "symbol": "MSFT",
    }],
}

mytool1 = BB_agent_tool(**fn_metadata)


def myfunc1(symbol):
    return mytool1(symbol=symbol)


get_10k_item1_tool = make_lc_tool(myfunc1, mytool1)

fn_metadata = {
    "name": "get_equity_search_symbol",
    "description": "Given a search string, get the stock symbol of the top company whose name best matches the search string.",
    "openapi_path": '/api/v1/equity/search',
    "parameters": {
        "query": {
            "type": "string",
            "description": "The search string to match to the stock symbol."
        }, ""
        "limit": {
            "type": "integer",
            "description": "The number of results to return. Pick a small number from 1 to 10 and choose the best response."
        }
    },
    "example_parameter_values": [{
        "query": "Broadcom",
    }],
    # "singular": 1,
}

mytool2 = BB_agent_tool(**fn_metadata)


def myfunc2(query):
    return mytool2(query=query)


search_tool = make_lc_tool(myfunc2, mytool2)

fn_metadata = {
    "name": "get_equity_price_quote",
    "description": "Given a stock symbol, get latest market data including last price in JSON format.",
    "openapi_path": '/api/v1/equity/price/quote',
    "parameters": {
        "symbol": {
            "type": "string",
            "description": "The stock symbol to get quote data for."
        },
    },
    "example_parameter_values": [{
        "symbol": "AAPL",
    }],
    # "singular": 1,
}

mytool3 = BB_agent_tool(**fn_metadata)


def myfunc3(symbol):
    return mytool3(symbol=symbol)


quote_tool = make_lc_tool(myfunc3, mytool3)


get_company_profile_json = BB_agent_tool(
    name="get_company_profile_json",
    description="Given a stock symbol, get general background data about the company such as company name, industry, and sector data in JSON format",
    openapi_path='/api/v1/equity/profile',
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


def fn_get_company_profile_json(symbol):
    return get_company_profile_json(symbol=symbol)


get_company_profile_json_tool = make_lc_tool(
    fn_get_company_profile_json, get_company_profile_json)

get_equity_shorts_short_interest = BB_agent_tool(
    name="get_equity_shorts_short_interest",
    description="Given a stock symbol, get data on short volume and days to cover in JSON format.",
    openapi_path='/api/v1/equity/shorts/short_interest',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=True
)


def fn_get_equity_shorts_short_interest(symbol):
    return get_equity_shorts_short_interest(symbol=symbol)


get_equity_shorts_short_interest_tool = make_lc_tool(
    fn_get_equity_shorts_short_interest, get_equity_shorts_short_interest)

get_equity_fundamental_historical_splits = BB_agent_tool(
    name="get_equity_fundamental_historical_splits",
    description="Given a stock symbol, get the company's historical stock splits in JSON format.",
    openapi_path='/api/v1/equity/fundamental/historical_splits',
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


def fn_get_equity_fundamental_historical_splits(symbol):
    return get_equity_fundamental_historical_splits(symbol=symbol)


get_equity_fundamental_historical_splits_tool = make_lc_tool(
    fn_get_equity_fundamental_historical_splits, get_equity_fundamental_historical_splits)

get_balance_sheet_json = BB_agent_tool(
    name="get_balance_sheet_json",
    description="Given a stock symbol, get the latest balance sheet data with assets and liabilities for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/balance',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_balance_sheet_json(symbol):
    return get_balance_sheet_json(symbol=symbol)


get_balance_sheet_json_tool = make_lc_tool(
    fn_get_balance_sheet_json, get_balance_sheet_json)

get_cash_flow_json = BB_agent_tool(
    name="get_cash_flow_json",
    description="Given a stock symbol, get the latest cash flow statement data for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/cash',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_cash_flow_json(symbol):
    return get_cash_flow_json(symbol=symbol)


get_cash_flow_json_tool = make_lc_tool(
    fn_get_cash_flow_json, get_cash_flow_json)


get_income_statement_json = BB_agent_tool(
    name="get_income_statement_json",
    description="Given a stock symbol, get the latest income statement data for the company in JSON format",
    openapi_path='/api/v1/equity/fundamental/income',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_income_statement_json(symbol):
    return get_income_statement_json(symbol=symbol)


get_income_statement_json_tool = make_lc_tool(
    fn_get_income_statement_json, get_income_statement_json)


get_fundamental_metrics_json = BB_agent_tool(
    name="get_fundamental_metrics_json",
    description="Given a stock symbol, get fundamental metrics for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/metrics',
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


def fn_get_fundamental_metrics_json(symbol):
    return get_fundamental_metrics_json(symbol=symbol)


get_fundamental_metrics_json_tool = make_lc_tool(
    fn_get_fundamental_metrics_json, get_fundamental_metrics_json)


get_fundamental_ratios_json = BB_agent_tool(
    name="get_fundamental_ratios_json",
    description="Given a stock symbol, get fundamental valuation ratios for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/ratios',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_fundamental_ratios_json(symbol):
    return get_fundamental_ratios_json(symbol=symbol)


get_fundamental_ratios_json_tool = make_lc_tool(
    fn_get_fundamental_ratios_json, get_fundamental_ratios_json)


get_equity_fundamental_multiples = BB_agent_tool(
    name="get_equity_fundamental_multiples",
    description="Given a stock symbol, get fundamental valuation multiples for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/multiples',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_equity_fundamental_multiples(symbol):
    return get_equity_fundamental_multiples(symbol=symbol)


get_equity_fundamental_multiples_tool = make_lc_tool(
    fn_get_equity_fundamental_multiples, get_equity_fundamental_multiples)


get_equity_fundamental_dividend = BB_agent_tool(
    name="get_equity_fundamental_dividend",
    description="Given a stock symbol, get the latest dividend data for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/dividends',
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


def fn_get_equity_fundamental_dividend(symbol):
    return get_equity_fundamental_dividend(symbol=symbol)


get_equity_fundamental_dividend_tool = make_lc_tool(
    fn_get_equity_fundamental_dividend, get_equity_fundamental_dividend)


get_trailing_dividend_yield_json = BB_agent_tool(
    name="get_trailing_dividend_yield_json",
    description="Given a stock symbol, get the 1 year trailing dividend yield for the company over time in JSON format.",
    openapi_path='/api/v1/equity/fundamental/trailing_dividend_yield',
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


def fn_get_trailing_dividend_yield_json(symbol):
    return get_trailing_dividend_yield_json(symbol=symbol)


get_trailing_dividend_yield_json_tool = make_lc_tool(
    fn_get_trailing_dividend_yield_json, get_trailing_dividend_yield_json)


get_price_performance_json = BB_agent_tool(
    name="get_price_performance_json",
    description="Given a stock symbol, get price performance data for the stock for different time periods in JSON format.",
    openapi_path='/api/v1/equity/price/performance',
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


def fn_get_price_performance_json(symbol):
    return get_price_performance_json(symbol=symbol)


get_price_performance_json_tool = make_lc_tool(
    fn_get_price_performance_json, get_price_performance_json)


get_equity_fundamental_multiples = BB_agent_tool(
    name="get_equity_fundamental_multiples",
    description="Given a stock symbol, get fundamental valuation multiples for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/multiples',
    parameters={
        "symbol": {
            "type": "string",
            "description": "The stock symbol."
        }
    },
    example_parameter_values=[{
        "symbol": "NVDA",
    }],
    singular=1
)


def fn_get_equity_fundamental_multiples(symbol):
    return get_equity_fundamental_multiples(symbol=symbol)


get_equity_fundamental_multiples_tool = make_lc_tool(
    fn_get_equity_fundamental_multiples, get_equity_fundamental_multiples)


get_equity_fundamental_dividend = BB_agent_tool(
    name="get_equity_fundamental_dividend",
    description="Given a stock symbol, get the latest dividend data for the company in JSON format.",
    openapi_path='/api/v1/equity/fundamental/dividends',
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


def fn_get_equity_fundamental_dividend(symbol):
    return get_equity_fundamental_dividend(symbol=symbol)


get_equity_fundamental_dividend_tool = make_lc_tool(
    fn_get_equity_fundamental_dividend, get_equity_fundamental_dividend)


get_trailing_dividend_yield_json = BB_agent_tool(
    name="get_trailing_dividend_yield_json",
    description="Given a stock symbol, get the 1 year trailing dividend yield for the company over time in JSON format.",
    openapi_path='/api/v1/equity/fundamental/trailing_dividend_yield',
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


def fn_get_trailing_dividend_yield_json(symbol):
    return get_trailing_dividend_yield_json(symbol=symbol)


get_trailing_dividend_yield_json_tool = make_lc_tool(
    fn_get_trailing_dividend_yield_json, get_trailing_dividend_yield_json)


get_price_performance_json = BB_agent_tool(
    name="get_price_performance_json",
    description="Given a stock symbol, get price performance data for the stock for different time periods in JSON format.",
    openapi_path='/api/v1/equity/price/performance',
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


def fn_get_price_performance_json(symbol):
    return get_price_performance_json(symbol=symbol)


get_price_performance_json_tool = make_lc_tool(
    fn_get_price_performance_json, get_price_performance_json)


# this might exceed token context making it unreliable, anyway XLV is the wrong answer
get_etf_equity_exposure_json = BB_agent_tool(
    name="get_etf_equity_exposure_json",
    description="Given a stock symbol, get the exposure of ETFs to the stock in JSON format.",
    openapi_path='/api/v1/etf/equity_exposure',
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


def fn_get_etf_equity_exposure_json(symbol):
    return get_etf_equity_exposure_json(symbol=symbol)


get_etf_equity_exposure_json_tool = make_lc_tool(
    fn_get_etf_equity_exposure_json, get_etf_equity_exposure_json)


tool_list = [get_10k_item1_tool, search_tool,
             quote_tool, get_company_profile_json_tool, get_equity_shorts_short_interest_tool,
             get_equity_fundamental_historical_splits_tool, get_balance_sheet_json_tool,
             get_cash_flow_json_tool, get_income_statement_json_tool, get_fundamental_metrics_json_tool,
             get_fundamental_ratios_json_tool, get_equity_fundamental_multiples_tool,
             get_equity_fundamental_dividend_tool, get_trailing_dividend_yield_json_tool,
             get_price_performance_json_tool, get_equity_fundamental_multiples_tool,
             get_equity_fundamental_dividend_tool, get_trailing_dividend_yield_json_tool,
             get_price_performance_json_tool, get_etf_equity_exposure_json_tool]

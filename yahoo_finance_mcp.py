#!/usr/bin/env python3
"""
Yahoo Finance Model Context Protocol (MCP) for LLaMA 3.2 3B
This module provides real-time financial data from Yahoo Finance for LLaMA.
"""

import json
import requests
import yfinance as yf
import pandas as pd
import datetime
import time
import re
import logging
from typing import Dict, Any, List, Optional, Union

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yahoo_finance_mcp.log')
    ]
)
logger = logging.getLogger(__name__)

class YahooFinanceMCP:
    """
    Model Context Protocol for integrating Yahoo Finance API with LLaMA 3.2 3B.
    This MCP allows the model to retrieve financial data, stock information,
    and market data through Yahoo Finance.
    """
    
    # Symbol mappings for companies that have changed their ticker
    SYMBOL_MAPPINGS = {
        "FB": "META",    # Facebook -> Meta
        "TWTR": "X",     # Twitter -> X
        "FACEBOOK": "META",  # Common name -> Meta
        "TWITTER": "X",    # Common name -> X
        # Add more mappings as needed
    }
    
    def __init__(self):
        """Initialize the Yahoo Finance MCP."""
        self.name = "yahoo_finance"
        self.description = "Provides access to financial data through Yahoo Finance API"
        self.version = "1.0.0"
        
        # Cache to store recent data and avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        # Cache TTL in seconds
        self.PRICE_CACHE_TTL = 30  # 30 seconds for prices (balance between freshness and stability)
        self.INFO_CACHE_TTL = 3600  # 1 hour for company info
        self.NEWS_CACHE_TTL = 300  # 5 minutes for news
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize stock symbols to their current form.
        For example, converts FB to META.
        
        Args:
            symbol: The stock symbol to normalize
            
        Returns:
            The normalized symbol
        """
        return self.SYMBOL_MAPPINGS.get(symbol.upper(), symbol.upper())

    def get_schema(self) -> Dict[str, Any]:
        """
        Returns the schema defining the capabilities of this MCP.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "functions": [
                {
                    # Get basic information about a stock
                    "name": "get_stock_info",
                    "description": "Get basic information about a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The stock symbol (ticker) to look up"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    # Get the current price of a stock
                    "name": "get_stock_price",
                    "description": "Get the current price of a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The stock symbol (ticker) to look up"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    # Get prices for multiple stocks
                    "name": "get_multiple_stock_prices",
                    "description": "Get current prices for multiple stocks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of stock symbols to look up"
                            }
                        },
                        "required": ["symbols"]
                    }
                },
                {
                    # Get historical stock data
                    "name": "get_stock_history",
                    "description": "Get historical stock data for a specified time period",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The stock symbol (ticker) to look up"
                            },
                            "period": {
                                "type": "string",
                                "description": "Time period to retrieve data for (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')",
                                "default": "1mo"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format (if using date range)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format (if using date range)"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    # Get latest market news
                    "name": "get_market_news",
                    "description": "Get latest market news",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Optional stock symbol for company-specific news"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of news items to return",
                                "default": 5
                            }
                        }
                    }
                }
            ]
        }
    
    def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Yahoo Finance function based on the function name and parameters.
        
        Args:
            function_name: The function to execute
            parameters: The parameters for the function
            
        Returns:
            The result of the function
        """
        logger.info(f"Executing function: {function_name} with parameters: {parameters}")
        
        try:
            if function_name == "get_stock_price":
                return self.get_stock_price(parameters.get("symbol"))
            
            elif function_name == "get_multiple_stock_prices":
                symbols = parameters.get("symbols", [])
                return self.get_multiple_stock_prices(symbols)
            
            elif function_name == "get_stock_info":
                return self.get_stock_info(parameters.get("symbol"))
            
            elif function_name == "get_stock_history":
                # Check if we have start_date and end_date for absolute periods
                if "start_date" in parameters and "end_date" in parameters:
                    return self.get_stock_history_range(
                        parameters.get("symbol"),
                        parameters.get("start_date"),
                        parameters.get("end_date")
                    )
                else:
                    return self.get_stock_history(
                        parameters.get("symbol"),
                        parameters.get("period", "1mo")
                    )
            
            elif function_name == "get_market_news":
                symbol = parameters.get("symbol")
                limit = parameters.get("limit", 5)
                return self.get_market_news(symbol, limit)
            
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            logger.error(f"Error executing {function_name}: {str(e)}")
            return {"error": f"Error: {str(e)}"}
    
    def _get_cached_or_fetch(self, key: str, fetch_func, ttl: int) -> Any:
        """
        Get data from cache or fetch it if not available or expired.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch the data if not in cache
            ttl: Time to live for cache entry in seconds
            
        Returns:
            The cached or freshly fetched data
        """
        current_time = time.time()
        
        # Check if we have it in cache and if it's still valid
        if key in self.cache and key in self.cache_expiry:
            if current_time < self.cache_expiry[key]:
                logger.info(f"Cache hit for {key}")
                return self.cache[key]
        
        # Fetch fresh data
        logger.info(f"Cache miss for {key}, fetching fresh data")
        data = fetch_func()
        
        # Only cache if there's no error
        if not isinstance(data, dict) or "error" not in data:
            # Update cache
            self.cache[key] = data
            self.cache_expiry[key] = current_time + ttl
        
        return data
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current stock price data for a specific symbol.
        
        Args:
            symbol: The stock symbol to look up
            
        Returns:
            Current price data
        """
        # Normalize the symbol first
        symbol = self.normalize_symbol(symbol)
        logger.info(f"\n{'='*80}\nGetting stock price for {symbol}")
        
        # Create a stable cache key (don't include timestamp to avoid excessive fetches)
        cache_key = f"price_{symbol}"
        
        def fetch_price():
            try:
                # Method 1: Direct Yahoo Finance API
                try:
                    # Get real-time data directly from Yahoo Finance API
                    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    # Log the API request
                    log_api_request("GET", url, headers=headers)
                    
                    response = requests.get(url, headers=headers, timeout=5)
                    
                    # Log the API response
                    log_api_response(response)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'quoteResponse' in data and 'result' in data['quoteResponse'] and len(data['quoteResponse']['result']) > 0:
                            quote = data['quoteResponse']['result'][0]
                            if 'regularMarketPrice' in quote:
                                logger.info(f"Successfully retrieved price via direct API: ${quote['regularMarketPrice']}")
                                return {
                                    "symbol": symbol,
                                    "price": float(quote['regularMarketPrice']),
                                    "open": float(quote['regularMarketOpen']) if 'regularMarketOpen' in quote else 0.0,
                                    "high": float(quote['regularMarketDayHigh']) if 'regularMarketDayHigh' in quote else 0.0,
                                    "low": float(quote['regularMarketDayLow']) if 'regularMarketDayLow' in quote else 0.0,
                                    "volume": int(quote['regularMarketVolume']) if 'regularMarketVolume' in quote else 0,
                                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "method": "direct_api",
                                    "company_name": quote.get('shortName', quote.get('longName', 'N/A')),
                                    "llm_response_template": f"The current stock price for {quote.get('shortName', quote.get('longName', symbol))} ({symbol}) is ${float(quote['regularMarketPrice']):.2f} per share."
                                }
                except Exception as e:
                    logger.warning(f"Direct API method failed for {symbol}: {e}")
                    if isinstance(e, requests.RequestException):
                        log_api_response(None, error=e)
                
                # Method 2: yfinance library
                logger.info(f"\nTrying yfinance library for {symbol}")
                ticker = yf.Ticker(symbol)
                
                # Try method 2a: Get live quotes
                try:
                    logger.info("Attempting to get live quotes...")
                    # For indices and many stocks, this method works well
                    quote = ticker.quotes
                    logger.info(f"Raw quotes response: {json.dumps(quote, indent=2)}")
                    
                    if symbol in quote and quote[symbol] and 'regularMarketPrice' in quote[symbol] and quote[symbol]['regularMarketPrice']:
                        logger.info(f"Successfully retrieved price via quotes: ${quote[symbol]['regularMarketPrice']}")
                        return {
                            "symbol": symbol,
                            "price": float(quote[symbol]['regularMarketPrice']),
                            "open": float(quote[symbol]['regularMarketOpen']) if 'regularMarketOpen' in quote[symbol] else 0.0,
                            "high": float(quote[symbol]['regularMarketDayHigh']) if 'regularMarketDayHigh' in quote[symbol] else 0.0,
                            "low": float(quote[symbol]['regularMarketDayLow']) if 'regularMarketDayLow' in quote[symbol] else 0.0,
                            "volume": int(quote[symbol]['regularMarketVolume']) if 'regularMarketVolume' in quote[symbol] else 0,
                            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "method": "quotes",
                            "company_name": quote[symbol].get('shortName', quote[symbol].get('longName', 'N/A')),
                            "llm_response_template": f"The current stock price for {quote[symbol].get('shortName', quote[symbol].get('longName', symbol))} ({symbol}) is ${float(quote[symbol]['regularMarketPrice']):.2f} per share."
                        }
                except Exception as e:
                    logger.warning(f"Quotes method failed for {symbol}: {e}")
                
                # Method 2b: Get info
                try:
                    logger.info("Attempting to get info...")
                    # This works for many stocks
                    info = ticker.info
                    logger.info(f"Raw info response: {json.dumps(info, indent=2)}")
                    
                    if info and 'regularMarketPrice' in info and info['regularMarketPrice']:
                        logger.info(f"Successfully retrieved price via info: ${info['regularMarketPrice']}")
                        return {
                            "symbol": symbol,
                            "price": float(info['regularMarketPrice']),
                            "open": float(info['regularMarketOpen']) if 'regularMarketOpen' in info else 0.0,
                            "high": float(info['regularMarketDayHigh']) if 'regularMarketDayHigh' in info else 0.0,
                            "low": float(info['regularMarketDayLow']) if 'regularMarketDayLow' in info else 0.0,
                            "volume": int(info['regularMarketVolume']) if 'regularMarketVolume' in info else 0,
                            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "method": "info",
                            "company_name": info.get('shortName', info.get('longName', 'N/A')),
                            "llm_response_template": f"The current stock price for {info.get('shortName', info.get('longName', symbol))} ({symbol}) is ${float(info['regularMarketPrice']):.2f} per share."
                        }
                except Exception as e:
                    logger.warning(f"Info method failed for {symbol}: {e}")
                
                # Method 2c: Get history
                try:
                    logger.info("Attempting to get history...")
                    # Last resort - most compatible but not always up to date
                    data = ticker.history(period="1d", interval="1m", auto_adjust=True)
                    
                    if not data.empty:
                        # Get the latest row (most recent data point)
                        latest = data.iloc[-1]
                        logger.info(f"Raw history data for latest point: {latest.to_dict()}")
                        
                        # Format the date
                        date_str = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                        logger.info(f"Successfully retrieved price via history: ${latest['Close']}")
                        
                        # Try to get company name from info
                        try:
                            company_name = ticker.info.get('shortName', ticker.info.get('longName', 'N/A'))
                        except:
                            company_name = 'N/A'
                        
                        return {
                            "symbol": symbol,
                            "price": float(latest["Close"]),
                            "open": float(latest["Open"]),
                            "high": float(latest["High"]),
                            "low": float(latest["Low"]),
                            "volume": int(latest["Volume"]),
                            "date": date_str,
                            "method": "history",
                            "company_name": company_name,
                            "llm_response_template": f"The current stock price for {company_name} ({symbol}) is ${float(latest['Close']):.2f} per share."
                        }
                except Exception as e:
                    logger.warning(f"History method failed for {symbol}: {e}")
                
                # All methods failed
                logger.error(f"All methods failed for {symbol}")
                return {"error": f"Could not retrieve stock price for {symbol} after trying all methods"}
                
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve stock price for {symbol}: {str(e)}"}
        
        # Use caching with reasonable TTL
        result = self._get_cached_or_fetch(cache_key, fetch_price, self.PRICE_CACHE_TTL)
        logger.info(f"Final result for {symbol}: {json.dumps(result, indent=2)}")
        logger.info(f"{'='*80}\n")
        return result
    
    def get_multiple_stock_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get current stock prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols to look up
            
        Returns:
            Dictionary with price data for each symbol
        """
        logger.info(f"Getting stock prices for multiple symbols: {symbols}")
        
        if not symbols:
            return {"error": "No symbols provided"}
        
        # Normalize symbols and remove duplicates
        normalized_symbols = list(set(self.normalize_symbol(symbol) for symbol in symbols))
        
        # Limit to a reasonable number of symbols
        if len(normalized_symbols) > 20:
            logger.warning(f"Too many symbols requested ({len(normalized_symbols)}), limiting to 20")
            normalized_symbols = normalized_symbols[:20]
        
        # Simplified approach - just get prices one by one
        # This is more reliable than trying to batch, which often fails
        results = {"prices": []}
        
        for symbol in normalized_symbols:
            price_data = self.get_stock_price(symbol)
            if "error" not in price_data:
                results["prices"].append(price_data)
        
        return results
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed company information for a stock symbol.
        
        Args:
            symbol: The stock symbol to look up
            
        Returns:
            Detailed company information
        """
        logger.info(f"Getting company info for {symbol}")
        
        cache_key = f"info_{symbol}"
        
        def fetch_info():
            try:
                # Get company information from Yahoo Finance
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info:
                    return {"error": f"No information found for symbol: {symbol}"}
                
                # Extract relevant information and handle missing fields gracefully
                result = {
                    "symbol": symbol,
                    "name": info.get("shortName", info.get("longName", "N/A")),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "marketCap": info.get("marketCap", "N/A"),
                    "trailingPE": info.get("trailingPE", "N/A"),
                    "dividendYield": info.get("dividendYield", "N/A") * 100 if info.get("dividendYield") is not None else "N/A",
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "N/A"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
                    "website": info.get("website", "N/A"),
                    "longBusinessSummary": info.get("longBusinessSummary", "No business summary available.")
                }
                
                return result
            except Exception as e:
                logger.error(f"Error fetching info for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve company information for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_info, self.INFO_CACHE_TTL)
    
    def get_stock_history(self, symbol: str, period: str) -> Dict[str, Any]:
        """
        Get historical stock data for a relative time period.
        
        Args:
            symbol: The stock symbol to look up
            period: Time period (1d, 1wk, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            
        Returns:
            Historical stock data
        """
        logger.info(f"Getting historical data for {symbol} over period {period}")
        
        cache_key = f"history_{symbol}_{period}"
        
        def fetch_history():
            try:
                # Set interval based on period for appropriate data granularity
                interval = "1d"  # default
                if period in ["1d", "5d"]:
                    interval = "5m"  # 5 minutes for short periods
                elif period in ["1wk", "1mo"]:
                    interval = "1h"  # hourly for medium periods
                
                # Get historical data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    return {"error": f"No historical data found for {symbol} with period {period}"}
                
                # Convert to list of dictionaries for easier processing
                data_points = []
                for date, row in hist.iterrows():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                        "close": float(row["Close"]),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "volume": int(row["Volume"])
                    })
                
                return {
                    "symbol": symbol,
                    "period": period,
                    "data": data_points
                }
            except Exception as e:
                logger.error(f"Error fetching history for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve historical data for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_history, self.INFO_CACHE_TTL)
    
    def get_stock_history_range(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical stock data for a specific date range.
        
        Args:
            symbol: The stock symbol to look up
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Historical stock data for the specified range
        """
        logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
        
        cache_key = f"history_range_{symbol}_{start_date}_{end_date}"
        
        def fetch_history_range():
            try:
                # Parse dates
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                # Calculate appropriate interval based on date range
                days_diff = (end - start).days
                if days_diff <= 7:
                    interval = "1h"  # hourly for <= 1 week
                elif days_diff <= 30:
                    interval = "1d"  # daily for <= 1 month
                elif days_diff <= 90:
                    interval = "1d"  # daily for <= 3 months
                else:
                    interval = "1wk"  # weekly for > 3 months
                
                # Get historical data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if hist.empty:
                    return {"error": f"No historical data found for {symbol} from {start_date} to {end_date}"}
                
                # Convert to list of dictionaries for easier processing
                data_points = []
                for date, row in hist.iterrows():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "close": float(row["Close"]),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "volume": int(row["Volume"])
                    })
                
                # Get the period description
                period_desc = f"{start_date} to {end_date}"
                
                return {
                    "symbol": symbol,
                    "period": period_desc,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data": data_points
                }
            except Exception as e:
                logger.error(f"Error fetching history range for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve historical data for {symbol} from {start_date} to {end_date}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_history_range, self.INFO_CACHE_TTL)
    
    def get_market_news(self, symbol: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Get latest market news or company-specific news.
        
        Args:
            symbol: Optional symbol for company-specific news
            limit: Maximum number of news items to return
            
        Returns:
            Latest market news
        """
        logger.info(f"Getting market news for {'general market' if symbol is None else symbol}")
        
        cache_key = f"news_{symbol if symbol else 'market'}"
        
        def fetch_news():
            try:
                if symbol:
                    # Get company-specific news
                    ticker = yf.Ticker(symbol)
                    news_items = ticker.news
                    
                    if not news_items:
                        # Fallback to general market news if no company news
                        ticker_sp = yf.Ticker("^GSPC")  # S&P 500 as backup
                        news_items = ticker_sp.news
                else:
                    # Get general market news
                    ticker = yf.Ticker("^GSPC")  # S&P 500
                    news_items = ticker.news
                
                if not news_items:
                    return {"error": "No market news found"}
                
                # Process and format news items
                processed_news = []
                for item in news_items[:limit]:
                    # Format the date if available
                    publish_date = "Unknown"
                    if "providerPublishTime" in item:
                        timestamp = item["providerPublishTime"]
                        publish_date = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    
                    processed_news.append({
                        "title": item.get("title", "No title"),
                        "publisher": item.get("publisher", "Unknown"),
                        "link": item.get("link", "#"),
                        "published": publish_date
                    })
                
                return {
                    "symbol": symbol,
                    "news": processed_news
                }
            except Exception as e:
                logger.error(f"Error fetching news: {str(e)}")
                return {"error": f"Failed to retrieve market news: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_news, self.NEWS_CACHE_TTL)


# Register with LLaMA function (for standalone testing)
def register_with_llama(mcp: YahooFinanceMCP):
    """Helper function to register the MCP with LLaMA (for testing)"""
    schema = mcp.get_schema()
    print(f"Registering MCP: {schema['name']} v{schema['version']}")
    print(f"Description: {schema['description']}")
    print(f"Functions: {', '.join(f['name'] for f in schema['functions'])}")
    return {
        "status": "success",
        "model": "LLaMA 3.2 3B",
        "mcp_registered": schema['name'],
        "functions_available": len(schema['functions'])
    }


if __name__ == "__main__":
    # Simple test script
    mcp = YahooFinanceMCP()
    
    # Test different stock price retrieval methods
    print("\nTesting stock price reliability:")
    test_symbols = [
        "AAPL",  # Apple
        "META",  # Meta
        "^GSPC", # S&P 500
        "^IXIC", # NASDAQ
        "^DJI",  # Dow Jones
        "UBER",  # Uber
        "LYFT",  # Lyft
        "SPOT",  # Spotify
        "DIS",   # Disney
        "AMZN",  # Amazon
        "IBM",   # IBM
        "WMT",   # Walmart
        "INTC"   # Intel
    ]
    
    print("\nTesting individual stock prices:")
    for symbol in test_symbols:
        result = mcp.get_stock_price(symbol)
        if "error" in result:
            print(f"{symbol}: ERROR - {result['error']}")
        else:
            print(f"{symbol} ({result.get('company_name', 'N/A')}): ${result['price']} (using method: {result.get('method', 'unknown')})")
    
    print("\nTesting multiple stock prices:")
    prices = mcp.get_multiple_stock_prices(test_symbols)
    for price in prices.get("prices", []):
        print(f"{price['symbol']} ({price.get('company_name', 'N/A')}): ${price['price']} (using method: {price.get('method', 'unknown')})")
    
    print("\nTesting company info:")
    print(mcp.get_stock_info("MSFT")) 
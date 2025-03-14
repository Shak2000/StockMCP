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

# Configure logging to only show CRITICAL level logs
logging.basicConfig(
    level=logging.CRITICAL,  # Only show CRITICAL logs
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
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
                },
                {
                    # Get a specific field from a stock's information
                    "name": "get_stock_field",
                    "description": "Get a specific field from a stock's information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The stock symbol to look up"
                            },
                            "field": {
                                "type": "string",
                                "description": "The specific field to retrieve"
                            }
                        },
                        "required": ["symbol", "field"]
                    }
                },
                {
                    # Get multiple fields from a stock's information
                    "name": "get_multiple_stock_fields",
                    "description": "Get multiple fields from a stock's information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The stock symbol to look up"
                            },
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of fields to retrieve"
                            }
                        },
                        "required": ["symbol", "fields"]
                    }
                }
            ]
        }
    
    def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function with the given parameters."""
        try:
            # Get the function from this class
            func = getattr(self, function_name)
            
            # Execute the function with parameters
            return func(**parameters)
            
        except Exception as e:
            logger.critical(f"Error executing {function_name}: {str(e)}")
            return {"error": str(e)}

    def _get_cached_or_fetch(self, key: str, fetch_func: callable, ttl: int) -> Dict[str, Any]:
        """Get data from cache or fetch if not present/expired."""
        now = time.time()
        
        # Check cache
        if key in self.cache and now - self.cache_expiry[key] < ttl:
            return self.cache[key]
            
        # Fetch fresh data
        data = fetch_func()
        
        # Cache the result
        self.cache[key] = data
        self.cache_expiry[key] = now + ttl
        
        return data
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current stock price for a symbol.
        
        Args:
            symbol: The stock symbol to look up
            
        Returns:
            Dictionary with price data
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"price_{symbol}"
        
        def fetch_price():
            try:
                # Method 1: Direct API call to Yahoo Finance
                url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                try:
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'quoteResponse' in data and 'result' in data['quoteResponse'] and len(data['quoteResponse']['result']) > 0:
                            quote = data['quoteResponse']['result'][0]
                            if 'regularMarketPrice' in quote:
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
                    logger.critical(f"Direct API method failed for {symbol}: {e}")
                
                # Method 2: yfinance library
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Try quotes first
                    try:
                        quote = ticker.quotes
                        if quote and symbol in quote and 'regularMarketPrice' in quote[symbol]:
                            return {
                                "symbol": symbol,
                                "price": float(quote[symbol]['regularMarketPrice']),
                                "open": float(quote[symbol]['regularMarketOpen']) if 'regularMarketOpen' in quote[symbol] else 0.0,
                                "high": float(quote[symbol]['regularMarketDayHigh']) if 'regularMarketDayHigh' in quote[symbol] else 0.0,
                                "low": float(quote[symbol]['regularMarketDayLow']) if 'regularMarketDayLow' in quote[symbol] else 0.0,
                                "volume": int(quote[symbol]['regularMarketVolume']) if 'regularMarketVolume' in quote[symbol] else 0,
                                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "method": "yfinance_quotes",
                                "company_name": quote[symbol].get('shortName', quote[symbol].get('longName', 'N/A')),
                                "llm_response_template": f"The current stock price for {quote[symbol].get('shortName', quote[symbol].get('longName', symbol))} ({symbol}) is ${float(quote[symbol]['regularMarketPrice']):.2f} per share."
                            }
                    except Exception as e:
                        logger.critical(f"Quotes method failed for {symbol}: {e}")
                    
                    # Try info next
                    try:
                        info = ticker.info
                        if info and 'regularMarketPrice' in info:
                            return {
                                "symbol": symbol,
                                "price": float(info['regularMarketPrice']),
                                "open": float(info['regularMarketOpen']) if 'regularMarketOpen' in info else 0.0,
                                "high": float(info['regularMarketDayHigh']) if 'regularMarketDayHigh' in info else 0.0,
                                "low": float(info['regularMarketDayLow']) if 'regularMarketDayLow' in info else 0.0,
                                "volume": int(info['regularMarketVolume']) if 'regularMarketVolume' in info else 0,
                                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "method": "yfinance_info",
                                "company_name": info.get('shortName', info.get('longName', 'N/A')),
                                "llm_response_template": f"The current stock price for {info.get('shortName', info.get('longName', symbol))} ({symbol}) is ${float(info['regularMarketPrice']):.2f} per share."
                            }
                    except Exception as e:
                        logger.critical(f"Info method failed for {symbol}: {e}")
                    
                    # Try history as last resort
                    try:
                        hist = ticker.history(period="1d", interval="1m")
                        if not hist.empty:
                            latest = hist.iloc[-1]
                            return {
                                "symbol": symbol,
                                "price": float(latest['Close']),
                                "open": float(latest['Open']),
                                "high": float(latest['High']),
                                "low": float(latest['Low']),
                                "volume": int(latest['Volume']),
                                "date": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
                                "method": "yfinance_history",
                                "company_name": symbol,  # Basic fallback
                                "llm_response_template": f"The current stock price for {symbol} is ${float(latest['Close']):.2f} per share."
                            }
                    except Exception as e:
                        logger.critical(f"History method failed for {symbol}: {e}")
                
                except Exception as e:
                    logger.critical(f"All yfinance methods failed for {symbol}: {e}")
                
                # All methods failed
                return {"error": f"Could not retrieve stock price for {symbol} after trying all methods"}
                
            except Exception as e:
                logger.critical(f"Error fetching price for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve stock price for {symbol}: {str(e)}"}
        
        # Use caching with reasonable TTL
        return self._get_cached_or_fetch(cache_key, fetch_price, self.PRICE_CACHE_TTL)
    
    def get_multiple_stock_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get current stock prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols to look up
            
        Returns:
            Dictionary with price data for each symbol
        """
        if not symbols:
            return {"error": "No symbols provided"}
        
        # Normalize symbols and remove duplicates
        normalized_symbols = list(set(self.normalize_symbol(symbol) for symbol in symbols))
        
        # Limit to a reasonable number of symbols
        if len(normalized_symbols) > 20:
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
        Get company information for a symbol.
        
        Args:
            symbol: The stock symbol to look up
            
        Returns:
            Dictionary with company information
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"info_{symbol}"
        
        def fetch_info():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    return {
                        "symbol": symbol,
                        "info": info,
                        "company_name": info.get('shortName', info.get('longName', 'N/A')),
                        "llm_response_template": f"Here is the company information for {info.get('shortName', info.get('longName', symbol))} ({symbol}): {json.dumps(info, indent=2)}"
                    }
                else:
                    return {"error": f"No information found for {symbol}"}
                    
            except Exception as e:
                logger.critical(f"Error fetching info for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve company information for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_info, self.INFO_CACHE_TTL)

    def get_stock_field(self, symbol: str, field: str) -> Dict[str, Any]:
        """
        Get a specific field from company information.
        
        Args:
            symbol: The stock symbol to look up
            field: The field to retrieve
            
        Returns:
            Dictionary with the field value
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"field_{symbol}_{field}"
        
        def fetch_field():
            try:
                # Get company info
                info = self.get_stock_info(symbol)
                
                if "error" in info:
                    return info
                
                if field in info["info"]:
                    value = info["info"][field]
                    
                    # Format the field name for display
                    field_description = " ".join(re.findall('[A-Z][^A-Z]*', field)).lower()
                    if not field_description:
                        field_description = field.lower()
                    
                    return {
                        "symbol": symbol,
                        "field": field,
                        "field_description": field_description,
                        "value": value,
                        "company_name": info["company_name"],
                        "llm_response_template": f"The {field_description} for {info['company_name']} ({symbol}) is {value}"
                    }
                else:
                    return {"error": f"Field '{field}' not found for {symbol}"}
                    
            except Exception as e:
                logger.critical(f"Error fetching field '{field}' for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve field '{field}' for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_field, self.INFO_CACHE_TTL)

    def get_multiple_stock_fields(self, symbol: str, fields: List[str]) -> Dict[str, Any]:
        """
        Get multiple fields from company information.
        
        Args:
            symbol: The stock symbol to look up
            fields: List of fields to retrieve
            
        Returns:
            Dictionary with field values
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"fields_{symbol}_{'_'.join(fields)}"
        
        def fetch_fields():
            try:
                results = []
                for field in fields:
                    field_data = self.get_stock_field(symbol, field)
                    if "error" not in field_data:
                        results.append(field_data)
                
                if results:
                    # Get company name from first result
                    company_name = results[0]["company_name"]
                    
                    # Create response template
                    field_values = []
                    for result in results:
                        field_values.append(f"{result['field_description']}: {result['value']}")
                    
                    return {
                        "symbol": symbol,
                        "fields": results,
                        "company_name": company_name,
                        "llm_response_template": f"For {company_name} ({symbol}):\n" + "\n".join([f"- {value}" for value in field_values])
                    }
                else:
                    return {"error": f"No fields could be retrieved for {symbol}"}
                    
            except Exception as e:
                logger.critical(f"Error fetching fields for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve fields for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_fields, self.INFO_CACHE_TTL)

    def get_stock_history(self, symbol: str, period: str) -> Dict[str, Any]:
        """
        Get historical stock data for a relative time period.
        
        Args:
            symbol: The stock symbol to look up
            period: Time period (1d, 1wk, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            
        Returns:
            Historical stock data
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
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
                logger.critical(f"Error fetching history for {symbol}: {str(e)}")
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
            Historical stock data
        """
        # Normalize the symbol
        symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"history_range_{symbol}_{start_date}_{end_date}"
        
        def fetch_history_range():
            try:
                # Convert dates to datetime objects
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                # Calculate date difference to determine appropriate interval
                date_diff = (end - start).days
                
                if date_diff <= 7:
                    interval = "5m"  # 5-minute intervals for <= 7 days
                elif date_diff <= 30:
                    interval = "1h"  # Hourly intervals for <= 30 days
                else:
                    interval = "1d"  # Daily intervals for > 30 days
                
                # Get historical data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if hist.empty:
                    return {"error": f"No historical data found for {symbol} between {start_date} and {end_date}"}
                
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
                    "start_date": start_date,
                    "end_date": end_date,
                    "data": data_points
                }
                    
            except Exception as e:
                logger.critical(f"Error fetching history range for {symbol}: {str(e)}")
                return {"error": f"Failed to retrieve historical data for {symbol}: {str(e)}"}
        
        return self._get_cached_or_fetch(cache_key, fetch_history_range, self.INFO_CACHE_TTL)

    def get_market_news(self, symbol: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Get market news for a symbol or general market news.
        
        Args:
            symbol: Optional stock symbol to get news for
            limit: Maximum number of news items to return
            
        Returns:
            Dictionary with news items
        """
        # Normalize the symbol if provided
        if symbol:
            symbol = self.normalize_symbol(symbol)
        
        # Generate cache key
        cache_key = f"news_{symbol if symbol else 'market'}"
        
        def fetch_news():
            try:
                if symbol:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                else:
                    # For general market news, use ^GSPC (S&P 500)
                    ticker = yf.Ticker("^GSPC")
                    news = ticker.news
                
                if not news:
                    return {"error": f"No news found for {symbol if symbol else 'the market'}"}
                
                # Process news items
                news_items = []
                for item in news[:limit]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", ""),
                        "published": datetime.datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                return {
                    "symbol": symbol if symbol else "market",
                    "news": news_items,
                    "llm_response_template": f"Here are the latest news items for {symbol if symbol else 'the market'}:\n" + 
                                          "\n".join([f"{i+1}. {item['title']} - {item['publisher']}" for i, item in enumerate(news_items)])
                }
                    
            except Exception as e:
                logger.critical(f"Error fetching news: {str(e)}")
                return {"error": f"Failed to retrieve news: {str(e)}"}
        
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
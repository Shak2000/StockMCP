import json
import requests
import yfinance as yf
from typing import Dict, List, Any, Optional, Union

class YahooFinanceMCP:
    """
    Model Context Protocol (MCP) for integrating Yahoo Finance API with LLaMA 3.2 3B.
    This MCP allows the model to retrieve financial data, stock information,
    and market data through Yahoo Finance.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance MCP"""
        self.name = "yahoo_finance"
        self.description = "Provides access to financial data through Yahoo Finance API"
        self.version = "1.0.0"
        
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
                    # Get historical stock data for a specified time period
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
                                "description": "Time period to retrieve data for (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')",
                                "default": "1mo"
                            },
                            "interval": {
                                "type": "string",
                                "description": "Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')",
                                "default": "1d"
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

    def execute_function(self, function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function of the MCP with the provided parameters.
        
        Args:
            function_name: Name of the function to execute
            params: Parameters for the function
            
        Returns:
            Dictionary containing the execution results
        """
        try:
            if function_name == "get_stock_info":
                return self._get_stock_info(params["symbol"])
            elif function_name == "get_stock_price":
                return self._get_stock_price(params["symbol"])
            elif function_name == "get_stock_history":
                period = params.get("period", "1mo")
                interval = params.get("interval", "1d")
                return self._get_stock_history(params["symbol"], period, interval)
            elif function_name == "get_market_news":
                limit = params.get("limit", 5)
                return self._get_market_news(limit)
            else:
                return {"error": f"Unknown function: {function_name}"}
        except Exception as e:
            return {"error": str(e)}

    def _get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic information about a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Return a subset of the most relevant information
            return {
                "symbol": symbol,
                "name": info.get("shortName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "trailingPE": info.get("trailingPE", "N/A"),
                "dividendYield": info.get("dividendYield", "N/A") * 100 if info.get("dividendYield") else "N/A",
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "N/A"),
                "website": info.get("website", "N/A"),
                "longBusinessSummary": info.get("longBusinessSummary", "N/A")
            }
        except Exception as e:
            return {"error": f"Failed to get stock info for {symbol}: {str(e)}"}

    def _get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get the current price of a stock"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if data.empty:
                return {"error": f"No price data available for {symbol}"}
                
            last_row = data.iloc[-1]
            return {
                "symbol": symbol,
                "price": last_row["Close"],
                "open": last_row["Open"],
                "high": last_row["High"],
                "low": last_row["Low"],
                "volume": last_row["Volume"],
                "date": last_row.name.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": f"Failed to get stock price for {symbol}: {str(e)}"}

    def _get_stock_history(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Get historical stock data for a specified time period"""
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                return {"error": f"No historical data available for {symbol}"}
                
            # Convert to list of dictionaries for easier JSON serialization
            history_data = []
            for date, row in history.iterrows():
                history_data.append({
                    "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })
            
            # Convert to JSON string
            history_json = json.dumps(history_data)
            
            return {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": history_json
            }
        except Exception as e:
            return {"error": f"Failed to get stock history for {symbol}: {str(e)}"}

    def _get_market_news(self, limit: int = 5) -> Dict[str, Any]:
        """Get latest market news"""
        try:
            # Use a general market index to get relevant news
            ticker = yf.Ticker("^GSPC")  # S&P 500
            news = ticker.news
            
            if not news:
                return {"error": "No market news available"}
                
            # Limit and format the news items
            news_items = []
            for item in news[:limit]:
                news_items.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": item.get("providerPublishTime", "")
                })
                
            return {
                "news": news_items
            }
        except Exception as e:
            return {"error": f"Failed to get market news: {str(e)}"}


# Integration with LLaMA 3.2 3B
def register_with_llama(mcp: YahooFinanceMCP):
    """
    Register the Yahoo Finance MCP with the LLaMA 3.2 3B model.
    This function would be called to integrate the MCP with the model.
    In a real implementation, this would connect to the model's API
    or inference server to register the MCP.
    """
    # This is a placeholder for the actual registration process with LLaMA 3.2 3B
    # In a real implementation, you would:
    # 1. Connect to the LLaMA 3.2 3B inference server
    # 2. Register the MCP schema
    # 3. Set up a callback mechanism for function execution
    
    schema = mcp.get_schema()
    print(f"Registering MCP: {schema['name']} v{schema['version']}")
    print(f"Description: {schema['description']}")
    print(f"Functions: {', '.join(f['name'] for f in schema['functions'])}")
    
    # Return the registration information (mock)
    return {
        "status": "success",
        "model": "LLaMA 3.2 3B",
        "mcp_registered": schema['name'],
        "functions_available": len(schema['functions'])
    }


if __name__ == "__main__":
    # Create and initialize the MCP
    yahoo_finance_mcp = YahooFinanceMCP()
    
    # Register the MCP with LLaMA 3.2 3B
    registration_result = register_with_llama(yahoo_finance_mcp)
    print(json.dumps(registration_result, indent=2))
    
    # Example usage
    print("\nExample: Get stock info for Apple (AAPL)")
    result = yahoo_finance_mcp.execute_function("get_stock_info", {"symbol": "AAPL"})
    print(json.dumps(result, indent=2)) 
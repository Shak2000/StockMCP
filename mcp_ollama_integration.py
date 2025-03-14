#!/usr/bin/env python3
"""
Model Context Protocol (MCP) integration for LLaMA 3.2 3B via Ollama with Yahoo Finance API.
This implementation connects to the real LLaMA 3.2 3B model through Ollama
and extends its capabilities with real-time financial data from Yahoo Finance.
"""

import os
import re
import sys
import json
import time
import ollama
import argparse
from typing import Dict, Any, List, Tuple, Optional
import datetime

# Import our Yahoo Finance MCP
from yahoo_finance_mcp import YahooFinanceMCP

# Regular expressions for detecting financial queries
STOCK_PRICE_PATTERN = r'(price|worth|value|cost|trading at).*?(of|for)?\s+([a-z]+\s+)?(stock|share|ticker)'
COMPANY_INFO_PATTERN = r'(about|info|information|details|tell me about).*?(company|business|corporation)'
STOCK_HISTORY_PATTERN = r'(history|historical|performance|trend|movement|chart|change).*?(stock|share|price)'
NEWS_PATTERN = r'(news|headline|recent development|market update)'

# Time period patterns
YEAR_PATTERN = r'(?:in|during|for|of)\s+(?:the\s+year\s+)?(\d{4})'
QUARTER_PATTERN = r'(?:in|during|for)\s+(?:the\s+)?(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(\d{4})'
HALF_PATTERN = r'(?:in|during|for)\s+(?:the\s+)?(first|second|1st|2nd)\s+half\s+(?:of\s+)?(\d{4})'
MONTH_YEAR_PATTERN = r'(?:in|during|for)\s+(?:the\s+month\s+of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'

# Company names and their ticker symbols
COMPANIES = [
    ("apple", "AAPL"), ("microsoft", "MSFT"), ("google", "GOOGL"), 
    ("alphabet", "GOOGL"), ("amazon", "AMZN"), ("tesla", "TSLA"), 
    ("meta", "META"), ("facebook", "META"), ("nvidia", "NVDA"), 
    ("netflix", "NFLX"), ("boeing", "BA"), ("ford", "F"), 
    ("general motors", "GM"), ("gm", "GM"), ("walmart", "WMT"), 
    ("target", "TGT"), ("nike", "NKE"), ("coca cola", "KO"), 
    ("coke", "KO"), ("pepsi", "PEP"), ("pepsico", "PEP"), 
    ("starbucks", "SBUX"), ("disney", "DIS"), ("amd", "AMD"), 
    ("intel", "INTC"), ("ibm", "IBM"), ("oracle", "ORCL"),
    ("salesforce", "CRM"), ("adobe", "ADBE"), ("spotify", "SPOT"),
    ("uber", "UBER"), ("lyft", "LYFT"), ("airbnb", "ABNB"),
    ("s&p 500", "^GSPC"), ("dow jones", "^DJI"), ("nasdaq", "^IXIC")
]

class MCPOllamaIntegration:
    """
    Model Context Protocol (MCP) integration for LLaMA 3.2 3B through Ollama.
    This class integrates the Yahoo Finance MCP with LLaMA 3.2 3B via Ollama.
    """
    
    # Regular expressions for detecting financial queries
    STOCK_PRICE_PATTERN = r'(price|worth|value|cost|trading at).*?(of|for)?\s+([a-z]+\s+)?(stock|share|ticker)'
    COMPANY_INFO_PATTERN = r'(about|info|information|details|tell me about).*?(company|business|corporation)'
    STOCK_HISTORY_PATTERN = r'(history|historical|performance|trend|movement|chart|change).*?(stock|share|price)'
    NEWS_PATTERN = r'(news|headline|recent development|market update)'

    # Time period patterns
    YEAR_PATTERN = r'(?:in|during|for|of)\s+(?:the\s+year\s+)?(\d{4})'
    QUARTER_PATTERN = r'(?:in|during|for)\s+(?:the\s+)?(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(\d{4})'
    HALF_PATTERN = r'(?:in|during|for)\s+(?:the\s+)?(first|second|1st|2nd)\s+half\s+(?:of\s+)?(\d{4})'
    MONTH_YEAR_PATTERN = r'(?:in|during|for)\s+(?:the\s+month\s+of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize the MCP Ollama integration.
        
        Args:
            model: The Ollama model to use (default: llama3.2:3b)
        """
        self.model = model
        self.mcp = YahooFinanceMCP()
        self.conversation_history = []
        
        # Print initialization information
        print(f"Initializing Yahoo Finance MCP for {self.model} via Ollama...")
        print("Model Context Protocol ready!")
        print(f"Model: {self.model}")
        print("MCP: Yahoo Finance API (Stock prices, company info, history, and market news)")
        print("\nYou can now ask financial questions, and the model will use real-time Yahoo Finance data.")
        print("For example:")
        print("- What is the current price of Apple stock?")
        print("- Tell me about Tesla as a company")
        print("- How has Microsoft's stock performed over the past month?")
        print("- What are the latest market news headlines?")
        print("\nYou can also ask any other questions as normal.")
        print("Enter 'exit', 'quit', or 'q' to quit.")
        print("-" * 70)
    
    def _extract_companies(self, query: str) -> List[str]:
        """
        Extract company symbols from a query.
        
        Args:
            query: The query to extract companies from
            
        Returns:
            List of company symbols found in the query
        """
        query_lower = query.lower()
        companies_found = []
        
        # First try exact matches
        for company, symbol in COMPANIES:
            # Look for company name as a whole word
            company_pattern = r'\b' + re.escape(company.lower()) + r'\b'
            symbol_pattern = r'\b' + re.escape(symbol.lower()) + r'\b'
            
            if re.search(company_pattern, query_lower) or re.search(symbol_pattern, query_lower):
                companies_found.append(symbol)
        
        # If no exact matches, try fuzzy matching for company names
        if not companies_found:
            for company, symbol in COMPANIES:
                if any(word in company.lower() for word in query_lower.split()):
                    companies_found.append(symbol)
        
        return companies_found

    def _is_future_date_query(self, query_lower: str) -> bool:
        """
        Check if the query is asking about a future date.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            True if the query is about a future date, False otherwise
        """
        current_year = datetime.datetime.now().year
        
        # Check for future year
        year_match = re.search(self.YEAR_PATTERN, query_lower)
        if year_match:
            year = int(year_match.group(1))
            if year > current_year:
                return True
        
        # Check for future quarter
        quarter_match = re.search(self.QUARTER_PATTERN, query_lower)
        if quarter_match:
            year = int(quarter_match.group(2))
            if year > current_year:
                return True
        
        # Check for future half
        half_match = re.search(self.HALF_PATTERN, query_lower)
        if half_match:
            year = int(half_match.group(2))
            if year > current_year:
                return True
        
        # Check for future month and year
        month_year_match = re.search(self.MONTH_YEAR_PATTERN, query_lower)
        if month_year_match:
            month_name = month_year_match.group(1)
            year = int(month_year_match.group(2))
            if year > current_year:
                return True
            elif year == current_year:
                current_month = datetime.datetime.now().month
                month_num = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4,
                    'may': 5, 'june': 6, 'july': 7, 'august': 8,
                    'september': 9, 'october': 10, 'november': 11, 'december': 12
                }[month_name]
                if month_num > current_month:
                    return True
        
        return False
    
    def _parse_historical_period(self, query_lower: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse historical time periods from the query.
        
        Returns:
            Tuple of (start_date, end_date, period_description)
        """
        from dateutil.relativedelta import relativedelta
        
        # Check for specific year
        year_match = re.search(MCPOllamaIntegration.YEAR_PATTERN, query_lower)
        if year_match:
            year = int(year_match.group(1))
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            return start_date, end_date, f"year {year}"
        
        # Check for specific quarter
        quarter_match = re.search(MCPOllamaIntegration.QUARTER_PATTERN, query_lower)
        if quarter_match:
            quarter = quarter_match.group(1).lower()
            year = int(quarter_match.group(2))
            quarter_num = {
                'first': 1, '1st': 1,
                'second': 2, '2nd': 2,
                'third': 3, '3rd': 3,
                'fourth': 4, '4th': 4
            }[quarter]
            start_month = (quarter_num - 1) * 3 + 1
            end_month = quarter_num * 3
            start_date = f"{year}-{start_month:02d}-01"
            end_date = f"{year}-{end_month:02d}-{31 if end_month in [3,12] else 30}"
            return start_date, end_date, f"Q{quarter_num} {year}"
        
        # Check for specific half
        half_match = re.search(MCPOllamaIntegration.HALF_PATTERN, query_lower)
        if half_match:
            half = half_match.group(1).lower()
            year = int(half_match.group(2))
            if half in ['first', '1st']:
                start_date = f"{year}-01-01"
                end_date = f"{year}-06-30"
                period_desc = f"first half of {year}"
            else:
                start_date = f"{year}-07-01"
                end_date = f"{year}-12-31"
                period_desc = f"second half of {year}"
            return start_date, end_date, period_desc
        
        # Check for specific month and year
        month_year_match = re.search(MCPOllamaIntegration.MONTH_YEAR_PATTERN, query_lower)
        if month_year_match:
            month_name = month_year_match.group(1)
            year = int(month_year_match.group(2))
            month_num = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }[month_name]
            start_date = f"{year}-{month_num:02d}-01"
            
            # Calculate last day of month
            if month_num in [4, 6, 9, 11]:
                last_day = 30
            elif month_num == 2:
                last_day = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
            else:
                last_day = 31
            
            end_date = f"{year}-{month_num:02d}-{last_day}"
            return start_date, end_date, f"{month_name.title()} {year}"
        
        # Handle relative time periods (existing logic)
        if "day" in query_lower or "24 hour" in query_lower or "today" in query_lower:
            return None, None, "1d"
        elif "week" in query_lower:
            return None, None, "1wk"
        elif "month" in query_lower:
            if "six" in query_lower or "6" in query_lower:
                return None, None, "6mo"
            elif "three" in query_lower or "3" in query_lower:
                return None, None, "3mo"
            else:
                return None, None, "1mo"
        elif "year" in query_lower:
            if "five" in query_lower or "5" in query_lower:
                return None, None, "5y"
            elif "two" in query_lower or "2" in query_lower:
                return None, None, "2y"
            else:
                return None, None, "1y"
        
        return None, None, "1mo"  # default period
    
    def detect_financial_query(self, query: str) -> Dict[str, Any]:
        """Detect if the query is asking for financial data and what type."""
        # Normalize query to lowercase for pattern matching
        query_lower = query.lower()
        
        # First, check if this is a future date query
        if self._is_future_date_query(query_lower):
            return {
                "function": "handle_future_date_query",
                "parameters": {"query": query}
            }
        
        # Initialize parameters
        parameters = {}
        
        # Extract company names/symbols
        companies = self._extract_companies(query)
        if not companies:
            return None
            
        # Normalize company symbols
        companies = [self.mcp.normalize_symbol(company) for company in companies]
        # Remove duplicates while preserving order
        companies = list(dict.fromkeys(companies))
        
        # Check for price query patterns
        price_patterns = [
            r"(what( is|'s)|get|show|tell me|how much is) .* (stock |share )?price",
            r"(what( is|'s)|get|show|tell me|how much) .* trading at",
            r"(what( is|'s)|get|show|tell me) .* stock",
            r"how (much|many) (does|do) .* (cost|trade for)",
            r"current (price|value|stock) of .*"
        ]
        
        if any(re.search(pattern, query_lower) for pattern in price_patterns):
            if len(companies) == 1:
                return {
                    "function": "get_stock_price",
                    "parameters": {
                        "symbol": companies[0],
                        "response_instruction": "IMPORTANT: Please use the exact price from the API response's llm_response_template field. Do not use any other source for the price."
                    }
                }
            else:
                return {
                    "function": "get_multiple_stock_prices",
                    "parameters": {
                        "symbols": companies,
                        "response_instruction": "IMPORTANT: Please use the exact prices from the API response's llm_response_template fields. Do not use any other source for the prices."
                    }
                }
        
        # Company information query
        elif any(re.search(pattern, query_lower) for pattern in [COMPANY_INFO_PATTERN, r'(about|info|information|details|tell me about).*?(company|business|corporation)']):
            if len(companies) == 1:
                return {
                    "function": "get_stock_info",
                    "parameters": {
                        "symbol": companies[0],
                        "response_instruction": "IMPORTANT: Please use the exact information from the API response's llm_response_template field. Do not use any other source for the information."
                    }
                }
            else:
                return {
                    "function": "get_stock_info",
                    "parameters": {
                        "symbols": companies,
                        "response_instruction": "IMPORTANT: Please use the exact information from the API response's llm_response_template fields. Do not use any other source for the information."
                    }
                }
        
        # Stock history query
        elif any(re.search(pattern, query_lower) for pattern in [STOCK_HISTORY_PATTERN, r'(history|historical|performance|trend|movement|chart|change).*?(stock|share|price)']):
            if len(companies) == 1:
                # Parse the historical period
                start_date, end_date, period = self._parse_historical_period(query_lower)
                
                if start_date and end_date:
                    return {
                        "function": "get_stock_history",
                        "parameters": {
                            "symbol": companies[0],
                            "start_date": start_date,
                            "end_date": end_date,
                            "response_instruction": "IMPORTANT: Please use the exact historical data from the API response's llm_response_template field. Do not use any other source for the data."
                        }
                    }
                else:
                    return {
                        "function": "get_stock_history",
                        "parameters": {
                            "symbol": companies[0],
                            "period": period,
                            "response_instruction": "IMPORTANT: Please use the exact historical data from the API response's llm_response_template field. Do not use any other source for the data."
                        }
                    }
            else:
                return {
                    "function": "get_stock_history",
                    "parameters": {
                        "symbols": companies,
                        "response_instruction": "IMPORTANT: Please use the exact historical data from the API response's llm_response_template fields. Do not use any other source for the data."
                    }
                }
        
        # News query
        elif re.search(NEWS_PATTERN, query_lower):
            if companies:
                return {
                    "function": "get_market_news",
                    "parameters": {
                        "symbols": companies,
                        "limit": 5,
                        "response_instruction": "IMPORTANT: Please use the exact news from the API response's llm_response_template field. Do not use any other source for the news."
                    }
                }
            elif any(term in query_lower for term in ["market", "stock", "financial", "business", "economy"]):
                return {
                    "function": "get_market_news",
                    "parameters": {
                        "limit": 5,
                        "response_instruction": "IMPORTANT: Please use the exact market news from the API response's llm_response_template field. Do not use any other source for the news."
                    }
                }
        
        # Not a financial query
        return None
    
    def format_financial_data(self, function_name: str, result: Dict[str, Any]) -> str:
        """
        Format the financial data for presentation to the LLM.
        
        Args:
            function_name: The function that was called
            result: The result data
            
        Returns:
            Formatted data as a string
        """
        formatted_data = ""
        
        if "error" in result:
            return f"Error retrieving financial data: {result['error']}"
        
        if function_name == "get_stock_price":
            # Format the price
            formatted_data = (
                f"Yahoo Finance Data:\n"
                f"- Latest price for {result['symbol']}: ${result['price']:.2f} as of {result['date']}\n"
                f"- Today's range: ${result['low']:.2f} - ${result['high']:.2f}\n"
                f"- Open: ${result['open']:.2f}\n"
                f"- Volume: {result['volume']:,}"
            )
        
        elif function_name == "get_stock_info":
            # Format market cap
            if result['marketCap'] != "N/A":
                try:
                    market_cap = f"${result['marketCap']:,} USD"
                except:
                    market_cap = f"{result['marketCap']} USD"
            else:
                market_cap = "N/A"
            
            # Format dividend yield
            if result['dividendYield'] != "N/A" and isinstance(result['dividendYield'], (int, float)):
                dividend_yield = f"{result['dividendYield']:.2f}%"
            else:
                dividend_yield = "N/A"
            
            formatted_data = (
                f"Yahoo Finance Data for {result['name']} ({result['symbol']}):\n"
                f"- Sector: {result['sector']}\n"
                f"- Industry: {result['industry']}\n"
                f"- Market Cap: {market_cap}\n"
                f"- P/E Ratio: {result['trailingPE']}\n"
                f"- Dividend Yield: {dividend_yield}\n"
                f"- 52-Week Range: ${result['fiftyTwoWeekLow']} - ${result['fiftyTwoWeekHigh']}\n\n"
                f"Business Summary:\n{result['longBusinessSummary'][:500]}..."
            )
        
        elif function_name == "get_stock_history":
            try:
                # Handle both DataFrame and dict formats from yfinance
                if isinstance(result, dict):
                    if 'data' in result:
                        data_points = result['data']
                    else:
                        # Handle direct DataFrame conversion
                        data_points = []
                        for date, row in result.items():
                            data_points.append({
                                'date': date,
                                'close': float(row['Close']),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'volume': int(row['Volume'])
                            })
                else:
                    # Handle DataFrame format directly
                    data_points = []
                    for date, row in result.iterrows():
                        data_points.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'close': float(row['Close']),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'volume': int(row['Volume'])
                        })
                
                if data_points:
                    # Sort data points by date
                    data_points.sort(key=lambda x: x['date'])
                    
                    # Show first and last data points
                    first_point = data_points[0]
                    last_point = data_points[-1]
                    
                    # Calculate change
                    change = (last_point['close'] - first_point['close']) / first_point['close'] * 100
                    trend = "increased" if change > 0 else "decreased"
                    
                    formatted_data = (
                        f"Yahoo Finance Historical Data for {result.get('symbol', 'Unknown')} over {result.get('period', 'Unknown')}:\n"
                        f"- Starting point ({first_point['date']}): ${first_point['close']:.2f}\n"
                        f"- Ending point ({last_point['date']}): ${last_point['close']:.2f}\n"
                        f"- Change: {change:.2f}% ({trend})\n"
                        f"- Highest price: ${max(p['high'] for p in data_points):.2f}\n"
                        f"- Lowest price: ${min(p['low'] for p in data_points):.2f}\n"
                        f"- Average volume: {sum(p['volume'] for p in data_points) // len(data_points):,}\n"
                    )
                    
                    # Add additional points for context if there are enough data points
                    if len(data_points) >= 5:
                        formatted_data += "\nSelected data points:\n"
                        # Select a few points to show the trend (not too many)
                        sample_size = min(5, len(data_points))
                        step = len(data_points) // sample_size
                        for i in range(0, len(data_points), step):
                            if i < len(data_points) and i != 0 and i != len(data_points) - 1:  # Skip first and last (already shown)
                                point = data_points[i]
                                formatted_data += f"- {point['date']}: ${point['close']:.2f}\n"
            except Exception as e:
                formatted_data = f"Error formatting historical data: {str(e)}"
        
        elif function_name == "get_market_news":
            # Format the news
            news_items = result.get("news", [])
            if news_items:
                formatted_data = "Latest Market News from Yahoo Finance:\n"
                for i, news in enumerate(news_items, 1):
                    formatted_data += f"{i}. {news['title']} - {news['publisher']}\n"
        
        return formatted_data
    
    def process_with_mcp(self, query: str) -> str:
        """
        Process a user query, potentially using the MCP for financial data.
        
        Args:
            query: The user query
            
        Returns:
            The model's response
        """
        # Detect if this is a financial query
        detected_query = self.detect_financial_query(query)
        
        if detected_query:
            function_name = detected_query["function"]
            parameters = detected_query["parameters"]
            response_instruction = parameters.get("response_instruction", "")
            
            # Get financial data using the MCP
            financial_data = self.mcp.execute_function(function_name, parameters)
            
            # Check if we have a template response in the data
            template_response = None
            if isinstance(financial_data, dict):
                if "llm_response_template" in financial_data:
                    template_response = financial_data["llm_response_template"]
                elif "prices" in financial_data and financial_data["prices"]:
                    # Handle multiple stock prices
                    template_response = "\n".join(
                        price["llm_response_template"]
                        for price in financial_data["prices"]
                        if "llm_response_template" in price
                    )
            
            # Format the data for the LLM
            formatted_data = self.format_financial_data(function_name, financial_data)
            
            # Add context to the query for the model
            context_prompt = (
                f"The user asked: '{query}'\n\n"
                f"Here is real-time financial data from Yahoo Finance to help answer this question:\n"
                f"{formatted_data}\n\n"
            )
            
            # If we have a template response, add it to the context
            if template_response:
                context_prompt += (
                    f"Please use this exact response format:\n"
                    f"{template_response}\n\n"
                )
            
            context_prompt += (
                f"{response_instruction}\n"
                f"Respond as if you had this real-time financial information available to you."
            )
            
            # Call Ollama with the enhanced context
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant with access to real-time financial data from Yahoo Finance. "
                            "When providing financial data, you must use ONLY the exact values provided in the API response. "
                            "Do not use any other source for financial data."
                        )
                    },
                    *self.conversation_history,
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            # Update conversation history with original user query and model response
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response['message']['content']})
            
            return response['message']['content']
        else:
            # Regular query, just pass to Ollama
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *self.conversation_history,
                    {"role": "user", "content": query}
                ]
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response['message']['content']})
            
            return response['message']['content']

def run_interactive_session():
    """Run an interactive session with the MCP-enhanced LLaMA model."""
    parser = argparse.ArgumentParser(description="Yahoo Finance MCP integration with LLaMA via Ollama")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use (default: llama3.2:3b)")
    args = parser.parse_args()
    
    try:
        # Check if Ollama is available by making a simple API call
        ollama.list()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is installed and running.")
        print("You can install Ollama from: https://ollama.ai/")
        print("Then run: ollama pull llama3.2:3b")
        sys.exit(1)
    
    # Initialize the MCP integration
    mcp_integration = MCPOllamaIntegration(model=args.model)
    
    # Interactive chat loop
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            # Process the query with potential MCP enhancement
            start_time = time.time()
            response = mcp_integration.process_with_mcp(query)
            end_time = time.time()
            
            # Print the response
            print(f"\n{args.model}: {response}")
            print(f"\n[Response generated in {end_time - start_time:.2f} seconds]")
            
        except KeyboardInterrupt:
            print("\nSession terminated by user.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_interactive_session() 
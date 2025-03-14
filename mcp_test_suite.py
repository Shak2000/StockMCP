#!/usr/bin/env python3
"""
Test suite for Yahoo Finance MCP functionality.
This script runs 12 specific financial queries to test the MCP's ability to retrieve
various types of financial data from Yahoo Finance, displaying responses in a
conversational format similar to interacting with an LLM.
"""

import sys
import os
import io
import logging
import warnings

# Create a custom logger class that does nothing
class SilentLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
    
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass

# Register our custom logger class
logging.setLoggerClass(SilentLogger)

# Completely disable logging
logging.disable(logging.CRITICAL)

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logging - MUST be done before any imports that might configure logging
logging.basicConfig(
    level=logging.NOTSET,  # Setting to NOTSET but we've disabled logging anyway
    format='',  # Empty format to minimize any accidental output
    handlers=[]  # No handlers
)
logging.getLogger().setLevel(logging.NOTSET)
# Disable all loggers that might exist or be created later
logging.Logger.manager.loggerDict = {}

# Monkey patch the critical logging function to do nothing
original_critical = logging.Logger.critical
def silent_critical(self, *args, **kwargs):
    pass
logging.Logger.critical = silent_critical

# Redirect stdout during imports to suppress any print statements
original_stdout = sys.stdout
sys.stdout = io.StringIO()

# Now import modules that might produce output
import json
from yahoo_finance_mcp import YahooFinanceMCP

# After importing, replace the module's logger with our silent version
import yahoo_finance_mcp
yahoo_finance_mcp.logger = SilentLogger("yahoo_finance_mcp")
# Also replace any existing loggers
for name in logging.Logger.manager.loggerDict:
    logging.Logger.manager.loggerDict[name] = SilentLogger(name)

# Restore stdout
sys.stdout = original_stdout

# Define a NullHandler that does nothing with log records
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Add null handler to the root logger to prevent "No handlers found" warnings
logging.getLogger().addHandler(NullHandler())

def silence_function(mcp_function):
    """Decorator to silence any function call by redirecting stdout/stderr temporarily."""
    def wrapper(*args, **kwargs):
        # Save original configuration
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_log_disable = logging.disable
        
        # Redirect stdout/stderr to capture any print statements
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Create a custom logging filter to block all messages
        class BlockAllFilter(logging.Filter):
            def filter(self, record):
                return False
        
        # Apply the filter to all loggers
        block_filter = BlockAllFilter()
        root_logger = logging.getLogger()
        root_logger.addFilter(block_filter)
        
        # Disable all logging
        logging.disable(logging.CRITICAL)
        
        try:
            # Call the original function
            result = mcp_function(*args, **kwargs)
            return result
        finally:
            # Restore settings no matter what
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Remove our filter
            root_logger.removeFilter(block_filter)
    
    return wrapper

def print_conversational_result(query, result):
    """Print the result of a query in a conversational way, like an LLM would respond."""
    print(f"\nUser: {query}")
    
    if isinstance(result, dict) and "error" in result:
        # Filter out any critical log details from error messages
        error_msg = str(result["error"])
        # Remove anything after the first occurrence of CRITICAL
        if "CRITICAL" in error_msg:
            error_msg = error_msg.split("CRITICAL")[0].strip()
        # Simplify error message
        error_msg = "Could not retrieve the requested information."
        print(f"Assistant: I'm sorry, I couldn't retrieve that information. {error_msg}")
        return

    # Try to get the llm_response_template if available
    template = result.get("llm_response_template", None)
    if template:
        print(f"Assistant: {template}")
    else:
        # Fallback formatting if template is not available
        if "price" in result:
            # Stock price query
            print(f"Assistant: The current stock price for {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}) is ${result.get('price', 'N/A'):.2f} per share.")
        elif "value" in result:
            # Single field query
            print(f"Assistant: {result.get('field_description', result.get('field', 'Value'))} for {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}): {result.get('value', 'N/A')}")
        elif "fields" in result and isinstance(result["fields"], list):
            # Multiple fields query
            print(f"Assistant: For {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}):")
            for field in result["fields"]:
                print(f"- {field.get('field_description', field.get('field', 'Value'))}: {field.get('value', 'N/A')}")
        else:
            # Generic fallback without raw data
            print(f"Assistant: Here's the information for {result.get('symbol', 'the requested stock')}.")

def run_test_suite():
    """Run the test suite with the specified queries."""
    print("Yahoo Finance MCP Test Suite - Conversational Responses\n")
    
    # Ensure all loggers are using our silent logger
    for name in logging.Logger.manager.loggerDict:
        logging.Logger.manager.loggerDict[name] = SilentLogger(name)
    
    # Replace the root logger
    logging.root = SilentLogger("root")
    
    # Ensure yahoo_finance_mcp logger is silent
    import yahoo_finance_mcp
    yahoo_finance_mcp.logger = SilentLogger("yahoo_finance_mcp")
    
    # Initialize the MCP and silence its methods
    mcp = YahooFinanceMCP()
    mcp.get_stock_price = silence_function(mcp.get_stock_price)
    mcp.get_stock_field = silence_function(mcp.get_stock_field)
    mcp.get_multiple_stock_fields = silence_function(mcp.get_multiple_stock_fields)
    mcp.get_stock_info = silence_function(mcp.get_stock_info)
    mcp.get_stock_history = silence_function(mcp.get_stock_history)
    
    # Test 1: What is Intel's current stock price?
    query = "What is Intel's current stock price?"
    result = mcp.get_stock_price("INTC")
    print_conversational_result(query, result)
    
    # Test 2: How much does one Spotify share cost?
    query = "How much does one Spotify share cost?"
    result = mcp.get_stock_price("SPOT")
    print_conversational_result(query, result)
    
    # Test 3: What is the beta of Meta?
    query = "What is the beta of Meta?"
    result = mcp.get_stock_field("META", "beta")
    print_conversational_result(query, result)
    
    # Test 4: What is Walmart's P/E ratio?
    query = "What is Walmart's P/E ratio?"
    result = mcp.get_stock_field("WMT", "trailingPE")
    print_conversational_result(query, result)
    
    # Test 5: What is Ford's market capitalization?
    query = "What is Ford's market capitalization?"
    result = mcp.get_stock_field("F", "marketCap")
    print_conversational_result(query, result)
    
    # Test 6: What are Amazon's P/E ratio and market capitalization?
    query = "What are Amazon's P/E ratio and market capitalization?"
    result = mcp.get_multiple_stock_fields("AMZN", ["trailingPE", "marketCap"])
    print_conversational_result(query, result)
    
    # Test 7: What is Nike's 52-week low stock price?
    query = "What is Nike's 52-week low stock price?"
    result = mcp.get_stock_field("NKE", "fiftyTwoWeekLow")
    print_conversational_result(query, result)
    
    # Test 8: What is Google's 52-week high stock price?
    query = "What is Google's 52-week high stock price?"
    result = mcp.get_stock_field("GOOGL", "fiftyTwoWeekHigh")
    print_conversational_result(query, result)
    
    # Test 9: What are Disney's 52-week low stock price and 52-week high stock price?
    query = "What are Disney's 52-week low stock price and 52-week high stock price?"
    result = mcp.get_multiple_stock_fields("DIS", ["fiftyTwoWeekLow", "fiftyTwoWeekHigh"])
    print_conversational_result(query, result)
    
    # Test 10: What is the Coca Cola dividend yield?
    query = "What is the Coca Cola dividend yield?"
    result = mcp.get_stock_field("KO", "dividendYield")
    print_conversational_result(query, result)
    
    # Test 11: What is Apple's EPS?
    query = "What is Apple's EPS?"
    # Fix field name for Apple's EPS - try each possible field name
    try:
        # First try trailingEps
        result = mcp.get_stock_field("AAPL", "trailingEps")
        if "error" in result:
            # If error, try different field name: eps
            result = mcp.get_stock_field("AAPL", "eps")
        if "error" in result:
            # If still error, try different field name: epsTrailingTwelveMonths
            result = mcp.get_stock_field("AAPL", "epsTrailingTwelveMonths")
    except Exception as e:
        result = {"error": f"Failed to get Apple's EPS: {str(e)}"}
    print_conversational_result(query, result)
    
    # Test 12: What are the stock price and the beta of General Motors?
    query = "What are the stock price and the beta of General Motors?"
    # For GM's price and beta, first get current price and then beta separately if needed
    try:
        # Get stock price
        price_result = mcp.get_stock_price("GM")
        
        # Get beta
        beta_result = mcp.get_stock_field("GM", "beta")
        
        # Combine results
        combined_result = {
            "symbol": "GM",
            "company_name": price_result.get("company_name", "General Motors"),
            "llm_response_template": f"The current stock price for {price_result.get('company_name', 'General Motors')} (GM) is ${price_result.get('price', 'N/A'):.2f} per share. The beta value, which measures volatility relative to the market, is {beta_result.get('value', 'N/A')}."
        }
        result = combined_result
    except Exception as e:
        result = {"error": f"Failed to get General Motors data: {str(e)}"}
    print_conversational_result(query, result)

if __name__ == "__main__":
    try:
        # Silence all logging before we begin
        logging.disable(logging.CRITICAL)
        
        # Create a stream handler that captures stderr
        stderr_capture = io.StringIO()
        stderr_handler = logging.StreamHandler(stderr_capture)
        stderr_handler.setLevel(logging.NOTSET)
        
        # Add a filter to prevent any logging
        class BlockAllFilter(logging.Filter):
            def filter(self, record):
                return False
        
        stderr_handler.addFilter(BlockAllFilter())
        
        # Add the handler to the root logger
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        root_logger.addHandler(stderr_handler)
        
        # Run with all output suppressed except our explicit prints
        run_test_suite()
    except Exception as e:
        # Only print the error message without any log details
        error_msg = str(e)
        # Filter out any log-like content
        if ": CRITICAL :" in error_msg:
            error_msg = error_msg.split(": CRITICAL :")[0]
        print(f"Error: {error_msg}")
        sys.exit(1) 
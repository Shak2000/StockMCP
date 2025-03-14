#!/usr/bin/env python3
"""
Test suite for Yahoo Finance MCP functionality.
This script runs 10 specific financial queries to test the MCP's ability to retrieve
various types of financial data from Yahoo Finance.
"""

import sys
import json
from yahoo_finance_mcp import YahooFinanceMCP

def print_result(query, result):
    """Print the result of a query in a formatted way."""
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    
    if isinstance(result, dict) and "error" in result:
        print(f"ERROR: {result['error']}")
        return

    # Try to get the llm_response_template if available
    template = result.get("llm_response_template", None)
    if template:
        print(f"RESPONSE: {template}")
    else:
        # Fallback formatting if template is not available
        if "price" in result:
            # Stock price query
            print(f"RESPONSE: The current stock price for {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}) is ${result.get('price', 'N/A'):.2f} per share.")
        elif "value" in result:
            # Single field query
            print(f"RESPONSE: {result.get('field_description', result.get('field', 'Value'))} for {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}): {result.get('value', 'N/A')}")
        elif "fields" in result and isinstance(result["fields"], list):
            # Multiple fields query
            print(f"RESPONSE: For {result.get('company_name', result.get('symbol'))} ({result.get('symbol')}):")
            for field in result["fields"]:
                print(f"- {field.get('field_description', field.get('field', 'Value'))}: {field.get('value', 'N/A')}")
        else:
            # Generic fallback without showing raw data
            print(f"RESPONSE: Retrieved information for {result.get('symbol', 'the requested stock')}.")
    
    print(f"{'=' * 80}\n")

def run_test_suite():
    """Run the test suite with the 10 specified queries."""
    print("\nYahoo Finance MCP Test Suite")
    print("Running 10 specified financial queries...")
    
    # Initialize the MCP
    mcp = YahooFinanceMCP()
    
    # Test 1: What is Intel's current stock price?
    query = "What is Intel's current stock price?"
    result = mcp.get_stock_price("INTC")
    print_result(query, result)
    
    # Test 2: How much does one Spotify share cost?
    query = "How much does one Spotify share cost?"
    result = mcp.get_stock_price("SPOT")
    print_result(query, result)
    
    # Test 3: What is the beta of Meta?
    query = "What is the beta of Meta?"
    result = mcp.get_stock_field("META", "beta")
    print_result(query, result)
    
    # Test 4: What is Walmart's P/E ratio?
    query = "What is Walmart's P/E ratio?"
    result = mcp.get_stock_field("WMT", "trailingPE")
    print_result(query, result)
    
    # Test 5: What is Walmart's market capitalization?
    query = "What is Walmart's market capitalization?"
    result = mcp.get_stock_field("WMT", "marketCap")
    print_result(query, result)
    
    # Test 6: What is Amazon's P/E ratio and market capitalization?
    query = "What is Amazon's P/E ratio and market capitalization?"
    result = mcp.get_multiple_stock_fields("AMZN", ["trailingPE", "marketCap"])
    print_result(query, result)
    
    # Test 7: What is Nike's 52-week low stock price?
    query = "What is Nike's 52-week low stock price?"
    result = mcp.get_stock_field("NKE", "fiftyTwoWeekLow")
    print_result(query, result)
    
    # Test 8: What is Google's 52-week high stock price?
    query = "What is Google's 52-week high stock price?"
    result = mcp.get_stock_field("GOOGL", "fiftyTwoWeekHigh")
    print_result(query, result)
    
    # Test 9: What are Disney's 52-week low stock price and 52-week high stock price?
    query = "What are Disney's 52-week low stock price and 52-week high stock price?"
    result = mcp.get_multiple_stock_fields("DIS", ["fiftyTwoWeekLow", "fiftyTwoWeekHigh"])
    print_result(query, result)
    
    # Test 10: What is the Coca Cola dividend yield?
    query = "What is the Coca Cola dividend yield?"
    result = mcp.get_stock_field("KO", "dividendYield")
    print_result(query, result)
    
    print("\nTest suite completed.")

if __name__ == "__main__":
    try:
        run_test_suite()
    except Exception as e:
        print(f"Error running test suite: {e}")
        sys.exit(1) 
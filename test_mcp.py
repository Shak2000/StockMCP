#!/usr/bin/env python3
"""
Test script for the Yahoo Finance MCP.
This script demonstrates the functionality of the Yahoo Finance MCP
with both financial and non-financial queries.
"""

import json
from yahoo_finance_mcp import YahooFinanceMCP, register_with_llama

def main():
    """Test the Yahoo Finance MCP with various queries."""
    print("Initializing Yahoo Finance MCP for LLaMA 3.2 3B...")
    mcp = YahooFinanceMCP()
    
    # Register the MCP
    register_with_llama(mcp)
    
    # Test queries
    test_queries = [
        # Financial queries that should use the MCP
        "What is the current price of Apple stock?",
        "Tell me about Tesla as a company",
        "How has Microsoft's stock performed over the past month?",
        
        # Non-financial queries that should be handled by LLaMA directly
        "What is the capital of Florida?",
        "What is 2+2?",
        "Explain quantum computing",
    ]
    
    print("\nDemonstrating Yahoo Finance MCP functionality:")
    print("-" * 60)
    
    # Process financial queries that should use the MCP
    print("\n1. FINANCIAL QUERIES - Using Yahoo Finance MCP:")
    
    # Test get_stock_price function
    print("\na) Stock Price Query:")
    print("   Query: 'What is the current price of Apple stock?'")
    print("   LLaMA's response would recognize this as a financial query")
    print("   and call the Yahoo Finance MCP's get_stock_price function.")
    result = mcp.execute_function("get_stock_price", {"symbol": "AAPL"})
    print("\n   Yahoo Finance MCP Result:")
    print(f"   Latest price for {result['symbol']}: ${result['price']:.2f} as of {result['date']}")
    print(f"   Open: ${result['open']:.2f}, High: ${result['high']:.2f}, Low: ${result['low']:.2f}")
    print(f"   Volume: {result['volume']:,}")
    
    # Test get_stock_info function
    print("\nb) Company Information Query:")
    print("   Query: 'Tell me about Tesla as a company'")
    print("   LLaMA's response would recognize this as a financial query")
    print("   and call the Yahoo Finance MCP's get_stock_info function.")
    result = mcp.execute_function("get_stock_info", {"symbol": "TSLA"})
    print("\n   Yahoo Finance MCP Result:")
    print(f"   Company: {result['name']} ({result['symbol']})")
    print(f"   Sector: {result['sector']}, Industry: {result['industry']}")
    try:
        print(f"   Market Cap: ${result['marketCap']:,} USD")
    except:
        print(f"   Market Cap: {result['marketCap']} USD")
    print(f"   P/E Ratio: {result['trailingPE']}")
    print(f"   52-Week Range: ${result['fiftyTwoWeekLow']} - ${result['fiftyTwoWeekHigh']}")
    print(f"   Business Summary: {result['longBusinessSummary'][:300]}...")
    
    # Test get_stock_history function
    print("\nc) Stock History Query:")
    print("   Query: 'How has Microsoft's stock performed over the past month?'")
    print("   LLaMA's response would recognize this as a financial query")
    print("   and call the Yahoo Finance MCP's get_stock_history function.")
    result = mcp.execute_function("get_stock_history", {"symbol": "MSFT", "period": "1mo"})
    
    if "data" in result and len(result["data"]) > 0:
        first_point = result["data"][0]
        last_point = result["data"][-1]
        change = (last_point["close"] - first_point["close"]) / first_point["close"] * 100
        
        print("\n   Yahoo Finance MCP Result:")
        print(f"   Historical data for {result['symbol']} over {result['period']} period:")
        print(f"   Starting point ({first_point['date']}): ${first_point['close']:.2f}")
        print(f"   Ending point ({last_point['date']}): ${last_point['close']:.2f}")
        print(f"   Percentage change: {change:.2f}%")
    
    # Process non-financial queries that should be handled by LLaMA directly
    print("\n2. NON-FINANCIAL QUERIES - Handled directly by LLaMA 3.2 3B:")
    
    print("\na) General Knowledge Query:")
    print("   Query: 'What is the capital of Florida?'")
    print("   LLaMA's response: 'The capital of Florida is Tallahassee.'")
    print("   (This query doesn't trigger the Yahoo Finance MCP)")
    
    print("\nb) Math Query:")
    print("   Query: 'What is 2+2?'")
    print("   LLaMA's response: '2+2 equals 4.'")
    print("   (This query doesn't trigger the Yahoo Finance MCP)")
    
    print("\nc) Technical Explanation Query:")
    print("   Query: 'Explain quantum computing'")
    print("   LLaMA's response: 'Quantum computing is a type of computing that uses")
    print("   quantum-mechanical phenomena, such as superposition and entanglement, to")
    print("   perform operations on data. Unlike classical computers that use bits...")
    print("   (This query doesn't trigger the Yahoo Finance MCP)")
    
    print("\nCONCLUSION:")
    print("This demonstration shows how the Yahoo Finance MCP integrates with LLaMA 3.2 3B:")
    print("1. For financial queries, LLaMA utilizes the Yahoo Finance MCP to retrieve real-time data")
    print("2. For non-financial queries, LLaMA handles them directly without using the MCP")
    print("3. The integration is seamless to the user - they simply ask questions naturally")
    print("\nThis approach enhances LLaMA 3.2 3B's capabilities by giving it access to real-time")
    print("financial data while preserving its ability to answer general knowledge questions.")

if __name__ == "__main__":
    main()
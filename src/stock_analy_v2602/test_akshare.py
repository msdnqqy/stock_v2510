
import akshare as ak
import pandas as pd

print(f"akshare version: {ak.__version__}")
try:
    # Try to fetch some data that doesn't require database
    # For example, stock list
    print("Fetching stock list...")
    stock_list = ak.stock_zh_a_spot_em()
    print(f"Successfully fetched {len(stock_list)} stocks.")
    print(stock_list.head())
except Exception as e:
    print(f"Failed to fetch data: {e}")

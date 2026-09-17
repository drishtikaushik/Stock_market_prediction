"""
=========================================================
FILE NAME : fetch_data.py

PURPOSE:
--------
This file downloads historical stock market data from
Yahoo Finance and stores it as a CSV file.

Later, our machine learning model will use this CSV
to learn patterns in stock prices.

Think of this file as the "Data Collection" phase.
=========================================================
"""

# -------------------------------------------------------
# Import the yfinance library.
#
# We do not write code to connect with Yahoo ourselves.
# yfinance already provides functions that download
# stock market data for us.
#
# "yf" is simply a shorter nickname for yfinance.
# -------------------------------------------------------

import yfinance as yf
import pandas as pd


# -------------------------------------------------------
# STEP 1
#
# Store the stock symbol.
#
# Every company has a unique ticker symbol.
#
# Apple  -> AAPL
# Google -> GOOGL
# Microsoft -> MSFT
# Tesla -> TSLA
#
# Instead of writing "AAPL" everywhere,
# we store it inside a variable.
#
# This makes our code easier to modify later.
# -------------------------------------------------------

stock_symbol = "AAPL"


# -------------------------------------------------------
# STEP 2
#
# Download historical stock data.
#
# yf.download() contacts Yahoo Finance servers.
#
# Parameters:
#
# stock_symbol
#     Which company's stock should be downloaded?
#
# start
#     Starting date.
#
# end
#     Ending date.
#
# The function returns a pandas DataFrame.
#
# A DataFrame is simply a table.
# -------------------------------------------------------

stock_data = yf.download(
    stock_symbol,
    start="2022-01-01",
    end="2025-01-01"
)


# -------------------------------------------------------
# STEP 3
#
# Display the first five rows.
#
# head() is one of the most commonly used pandas
# functions.
#
# Why only five rows?
#
# Imagine a table with 700 rows.
#
# Printing everything would flood the terminal.
#
# So head() gives us a quick preview.
# -------------------------------------------------------

print("\nFirst Five Rows\n")
print(stock_data.head())


# -------------------------------------------------------
# STEP 4
#
# Display basic information.
#
# info() tells us:
#
# • Number of rows
# • Number of columns
# • Data types
# • Missing values
#
# This is one of the first things every data analyst
# checks after loading a dataset.
# -------------------------------------------------------

print("\nDataset Information\n")
print(stock_data.info())


# -------------------------------------------------------
# STEP 5
#
# Save the downloaded data.
#
# Machine Learning models usually read data
# from CSV files.
#
# CSV stands for
#
# Comma Separated Values
#
# It can be opened in:
#
# • Excel
# • Google Sheets
# • Pandas
#
# index=True saves the Date column.
#
# Without the date, our stock data loses
# an important piece of information.
# -------------------------------------------------------

stock_data.to_csv(
    "data/stock_data.csv",
    index=True
)


# -------------------------------------------------------
# STEP 6
#
# Inform the user that everything worked.
#
# Good software always tells the user
# what happened.
# -------------------------------------------------------

print("\nData downloaded successfully!")
print("CSV file saved inside the 'data' folder.")
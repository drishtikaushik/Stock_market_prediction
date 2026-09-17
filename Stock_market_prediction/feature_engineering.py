"""
=========================================================
FILE : feature_engineering.py

PURPOSE
--------
This file creates useful features for our Machine
Learning model.

Raw data is often not enough.

Instead of giving the model only stock prices,
we create additional information that helps the
model understand market trends.

This process is called Feature Engineering.
=========================================================
"""

# -----------------------------------------
# Import pandas
# -----------------------------------------

import pandas as pd


# -----------------------------------------
# STEP 1
#
# Read the cleaned dataset.
#
# We created this file in preprocess.py
# -----------------------------------------

stock_data = pd.read_csv("data/clean_stock_data.csv")


# -----------------------------------------
# STEP 2
#
# Display first five rows.
#
# This is only to verify that everything
# loaded correctly.
# -----------------------------------------

print(stock_data.head())


# -----------------------------------------
# STEP 3
#
# Create Tomorrow's Closing Price.
#
# shift(-1)
#
# moves every value upward by one row.
#
# Example
#
# Before
#
# Close
# -----
# 100
# 105
# 110
#
# After
#
# Close   Target
# 100      105
# 105      110
# 110      NaN
#
# Why?
#
# Because our model should learn
#
# Today's Price
#
# -------->
#
# Tomorrow's Price
#
# -----------------------------------------

stock_data["Target"] = stock_data["Close"].shift(-1)


# -----------------------------------------
# STEP 4
#
# Create Moving Average (5 Days)
#
# rolling(window=5)
#
# looks at the previous 5 rows.
#
# mean()
#
# calculates the average.
#
# Example
#
# Prices
#
# 100
# 105
# 110
# 108
# 115
#
# Average
#
# =
#
# (100+105+110+108+115)/5
#
# This helps the model understand
# short-term trend.
# -----------------------------------------

stock_data["MA5"] = stock_data["Close"].rolling(window=5).mean()


# -----------------------------------------
# STEP 5
#
# Create Moving Average (10 Days)
#
# This captures a slightly longer trend.
#
# Financial analysts frequently compare
# MA5 and MA10.
# -----------------------------------------

stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()


# -----------------------------------------
# STEP 6
#
# Create Previous Day Close.
#
# shift(1)
#
# moves data downward.
#
# Example
#
# Close
#
# 100
# 105
# 110
#
# Previous Close
#
# NaN
# 100
# 105
#
# This tells the model what happened
# one day earlier.
# -----------------------------------------

stock_data["Previous_Close"] = stock_data["Close"].shift(1)


# -----------------------------------------
# STEP 7
#
# Remove rows containing NaN.
#
# Why are NaN values created?
#
# MA5 needs five days.
#
# MA10 needs ten days.
#
# shift() also creates one empty row.
#
# Machine Learning cannot train using
# missing values.
# -----------------------------------------

stock_data.dropna(inplace=True)


# -----------------------------------------
# STEP 8
#
# Display the updated dataset.
# -----------------------------------------

print(stock_data.head())


# -----------------------------------------
# STEP 9
#
# Save the new dataset.
#
# This file will be used for training.
# -----------------------------------------

stock_data.to_csv(
    "data/features_stock_data.csv",
    index=False
)


print("\nFeature Engineering Completed Successfully.")
"""
=========================================================
FILE NAME : preprocess.py

PURPOSE
-------
This file loads the stock dataset that we downloaded
earlier and checks whether the data is suitable for
machine learning.

Remember:

Machine Learning models expect clean, organized data.

If the data contains missing values, duplicate rows,
or incorrect data types, the model may perform poorly.

Therefore, preprocessing is one of the most important
steps in every Machine Learning project.
=========================================================
"""

# ------------------------------------------------------
# Import the pandas library.
#
# Pandas helps us read, modify and analyze tables.
#
# We use the alias 'pd' because it is the standard
# convention followed by Python developers.
# ------------------------------------------------------

import pandas as pd


# ------------------------------------------------------
# STEP 1
#
# Read the CSV file that we created in fetch_data.py
#
# read_csv() loads the CSV file into a DataFrame.
#
# Think of it like:
#
# CSV File
#      ↓
# Pandas DataFrame
# ------------------------------------------------------

stock_data = pd.read_csv("data/stock_data.csv")


# ------------------------------------------------------
# STEP 2
#
# Display the first five rows.
#
# This is simply a quick inspection to verify that the
# dataset was loaded correctly.
# ------------------------------------------------------

print("\n========== FIRST FIVE ROWS ==========\n")
print(stock_data.head())


# ------------------------------------------------------
# STEP 3
#
# Check the size of the dataset.
#
# shape returns:
#
# (number_of_rows, number_of_columns)
#
# Example:
#
# (755, 7)
#
# means
#
# 755 rows
# 7 columns
# ------------------------------------------------------

print("\n========== DATASET SHAPE ==========\n")
print(stock_data.shape)


# ------------------------------------------------------
# STEP 4
#
# Display all column names.
#
# Sometimes datasets contain unexpected spaces,
# spelling mistakes or extra columns.
#
# Before working on data we should always know
# exactly which columns are available.
# ------------------------------------------------------

print("\n========== COLUMN NAMES ==========\n")
print(stock_data.columns)


# ------------------------------------------------------
# STEP 5
#
# Display detailed dataset information.
#
# info() tells us:
#
# • Number of rows
# • Number of columns
# • Data type of every column
# • Missing values
#
# This is one of the most frequently used commands
# in Data Science.
# ------------------------------------------------------

print("\n========== DATASET INFORMATION ==========\n")
stock_data.info()


# ------------------------------------------------------
# STEP 6
#
# Display statistical summary.
#
# describe() calculates statistics only for
# numerical columns.
#
# It gives:
#
# count
# mean
# std
# min
# max
# quartiles
#
# This helps us understand how the data is distributed.
# ------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========\n")
print(stock_data.describe())


# ------------------------------------------------------
# STEP 7
#
# Check for missing values.
#
# Missing values are empty cells inside the dataset.
#
# Example
#
# Close
# ------
# 175
# NaN
# 178
#
# NaN means "Not a Number", which represents
# missing data.
#
# isnull()
#
# Returns True wherever a value is missing.
#
# sum()
#
# Counts how many True values are present
# in each column.
# ------------------------------------------------------

print("\n========== MISSING VALUES ==========\n")
print(stock_data.isnull().sum())


# ------------------------------------------------------
# STEP 8
#
# Check for duplicate rows.
#
# Duplicate rows may appear because of
# incorrect data collection or merging datasets.
#
# duplicated()
#
# Returns True for duplicated rows.
#
# sum()
#
# Counts how many duplicate rows exist.
# ------------------------------------------------------

duplicate_rows = stock_data.duplicated().sum()

print("\n========== DUPLICATE ROWS ==========\n")
print(duplicate_rows)


# ------------------------------------------------------
# STEP 9
#
# Remove duplicate rows if any exist.
#
# inplace=True means:
#
# Modify the existing DataFrame directly.
#
# Without inplace=True,
# pandas would create a new DataFrame instead.
# ------------------------------------------------------

stock_data.drop_duplicates(inplace=True)


# ------------------------------------------------------
# STEP 10
#
# Save the cleaned dataset.
#
# This allows the next stages of our project
# to use clean data instead of raw data.
# ------------------------------------------------------

stock_data.to_csv(
    "data/clean_stock_data.csv",
    index=False
)


# ------------------------------------------------------
# STEP 11
#
# Inform the user that preprocessing is complete.
# ------------------------------------------------------

print("\nPreprocessing completed successfully.")
print("Clean dataset saved as clean_stock_data.csv")
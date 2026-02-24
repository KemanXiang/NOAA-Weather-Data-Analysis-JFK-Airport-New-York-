# NOAA-Weather-Data-Analysis-JFK-Airport-New-York-
This notebook focuses on analyzing and forecasting weather patterns using the NOAA Weather Dataset collected from JFK Airport in New York. The dataset comprises 114,546 hourly observations of 12 key climatological variables, including temperature, wind speed, humidity, and pressure. 
This notebook teaches the user to extract, clean and analyze sample weather data and predict weather trends to help airports schedule better flight times.

The notebook is organized into three main parts:

Part 1: Data Cleaning
In this section, we prepare the raw data for analysis by:

Removing unnecessary or redundant columns to retain only relevant numerical features
Converting data types and cleaning inconsistencies
Handling missing values with appropriate filling strategies
Encoding categorical weather features for downstream analysis

Part 2: Exploratory Data Analysis (EDA)
Here, we perform visual and statistical exploration of the cleaned dataset:

Load the cleaned data
Generate insightful visualizations of key variables
Identify trends, patterns, and seasonal effects in the time-series data

Part 3: Time Series Forecasting
This section focuses on predicting future temperatures using time-series models:

Load the cleaned and preprocessed data
Establish baseline forecasting models
Train and evaluate advanced statistical forecasting techniques

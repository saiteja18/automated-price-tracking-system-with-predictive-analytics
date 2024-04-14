# automated-price-tracking-system-with-predictive-analytics

## Introduction
In today's digital era, online shopping has become an integral part of our lives. With a plethora of options available, finding the best deals often involves navigating through multiple websites, which can be both time-consuming and overwhelming. Many consumers tend to stick to well-known platforms like Amazon or BestBuy, sacrificing potential savings due to the sheer effort required to compare prices across different sites.

What if there was a solution that could streamline this process, offering users a comprehensive comparison of product prices and details from various online retailers in one convenient location? Introducing our Multi-Site Price Comparison Tool – a revolutionary platform designed to simplify your online shopping experience.

Our tool aims to gather real-time information about product prices from a wide range of websites, enabling users to make informed decisions by comparing prices, features, and reviews effortlessly. Beyond just comparing current prices, our advanced algorithms predict future price trends, helping users anticipate the best time to make a purchase and maximize savings.

## Flow of the Project
1. **Web Scraping**: Extracting product data from various e-commerce websites using Python.
2. **Database Storag**e: Storing the scraped data in a structured database for easy retrieval and analysis.
3. **Price Prediction**: Implementing predictive models to forecast future price trends.
4. **Visualization**: Designing an intuitive website to visualize and compare product prices across platforms.

## Installation
### Clone the Repository
git clone https://github.com/saiteja18/automated-price-tracking-system-with-predictive-analytics.git
### Navigate to the Porject Directory
cd automated_price_tracking_system_with_predictive_analytics
### Install Dependencies
pip install -r requirements.txt

## Usage
### Run the web scraper and store in database
Running Final_Scraping_Scheduled.ipynb  will initiate the web scraping process to gather product data from various e-commerce and will store the scraped data in the database for future analysis.
### Predict Price Trends
The main.py implement predictive models to forecast future price trends based on the stored data.
### Visualize Data
Static and templates folders have all the required documents to build a web page and show the above analysis in a graphical format.


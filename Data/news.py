# import requests
# import pandas as pd
# import yfinance as yf

# # Fetch news data from NewsAPI
# def fetch_news(api_key, symbol, start_date, end_date):
#     url = f'https://newsapi.org/v2/everything'
#     params = {
#         'q': symbol,
#         'from': start_date,
#         'to': end_date,
#         'apiKey': api_key,
#         'language': 'en',
#         'sortBy': 'publishedAt'
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
    
#     articles = []
#     for article in data['articles']:
#         articles.append({
#             'Date': article['publishedAt'],
#             'Title': article['title'],
#             'Description': article['description'],
#             'Content': article['content']
#         })
    
#     return pd.DataFrame(articles)

# # Replace with your NewsAPI key
# api_key = 'e9532cf0833045ee8c48d929ccb9a728'
# symbol = 'VEDL'
# start_date = '2020-01-01'
# end_date = '2024-09-18'

# # Fetch and save news data
# news_data = fetch_news(api_key, symbol, start_date, end_date)
# news_data.to_csv('news_data.csv', index=False)

import requests
import pandas as pd

API_KEY = 'e9532cf0833045ee8c48d929ccb9a728'
API_KEY = '4dbc17e007ab436fb66416009dfb59a8'

url = 'https://newsapi.org/v2/everything'

# Set the parameters for the request
params = {
    'q': 'VEDL',             # Search for stock-related news about VEDL (Vedanta Limited)
    'language': 'en',        # News in English
    'sortBy': 'publishedAt', # Sort by latest news
    'apiKey': API_KEY,
    'from': '2024-01-01',
    'to': '2024-09-18',
}

# Make the request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    news_data = response.json()
    articles = news_data.get('articles', [])

    # Prepare data to save to CSV
    article_list = []
    for article in articles:
        article_list.append({
            'Title': article['title'],
            'Source': article['source']['name'],
            'PublishedAt': article['publishedAt'],
            'Description': article['description'],
            'URL': article['url']  # Add the URL of the article
        })

    df = pd.DataFrame(article_list)

    df.to_csv('vedanta_news.csv', index=False)

    print("News articles have been saved to 'vedanta_news.csv'.")

else:
    print(f"Failed to fetch news: {response.status_code}")
    print(f"Rzn : {response.text}")
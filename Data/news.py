import requests
import pandas as pd
import yfinance as yf

# Fetch news data from NewsAPI
def fetch_news(api_key, symbol, start_date, end_date):
    url = f'https://newsapi.org/v2/everything'
    params = {
        'q': symbol,
        'from': start_date,
        'to': end_date,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    articles = []
    for article in data['articles']:
        articles.append({
            'Date': article['publishedAt'],
            'Title': article['title'],
            'Description': article['description'],
            'Content': article['content']
        })
    
    return pd.DataFrame(articles)

# Replace with your NewsAPI key
api_key = 'YOUR_NEWSAPI_KEY'
symbol = 'VEDL'
start_date = '2020-01-01'
end_date = '2024-09-13'

# Fetch and save news data
news_data = fetch_news(api_key, symbol, start_date, end_date)
news_data.to_csv('news_data.csv', index=False)

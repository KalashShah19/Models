import json
import requests
from bs4 import BeautifulSoup

data = '''
{
    "companies": [
        {
            "id": "s0003041",
            "name": "vedanta"
        },
        {
            "name": "hindzinc",
            "id": "s0003378"
        },
        {
            "name": "iocl",
            "id": "s0003643"
        },
        {
            "name": "ntpc",
            "id": "s0003057"
        },
        {
            "id": "s0003044",
            "name": "coal"
        },
        {
            "id": "s0005375",
            "name": "kfin"
        },
        {
            "id": "s0003018",
            "name": "reliance"
        }
    ]
}
'''

json_data = json.loads(data)

def get_company_id(company_name):
    for company in json_data["companies"]:
        if company["name"].lower() == company_name.lower():
            return company["id"]
    return None 

# Function to fetch news data from Livemint
def fetch_news(stock):
    id = get_company_id(stock)
    url = f"https://www.livemint.com/{stock}/news/companyid-{id}"

    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return
    
    # Parse the webpage content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all news items by their class name
    news_items = soup.find_all('div', class_='jsx-abde3f138b2ad481 listtostoryList clearfix')

     # Open a file to save the output
    with open('news_data.txt', 'w', encoding='utf-8') as file:
        # Iterate through each news item and extract details
        for item in news_items:
            # Extract the headline and URL
            headline_tag = item.find('h2', class_='headline')
            if headline_tag:
                headline = headline_tag.text.strip()
                link = headline_tag.find('a')['href']
                full_link = f'https://www.livemint.com{link}' if link.startswith('/') else link
                
                # Write the news details to the file
                file.write(f"Headline: {headline}\n")
                file.write(f"URL: {full_link}\n")
                file.write('-' * 80 + '\n')
    
    print("News data has been saved to 'news.txt'")

stock_name = input("Enter Stock Name : ")
fetch_news(stock_name)

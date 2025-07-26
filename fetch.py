import requests
from dotenv import load_dotenv
load_dotenv() 
import os
def fetch_erc20_transfers(address):
    url = f"https://deep-index.moralis.io/api/v2.2/{address}/erc20/transfers?chain=eth"
    headers = {
        "accept": "application/json",
        "X-API-Key": os.getenv("API_KEY")
    
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('result', [])
    else:
        print(f"[{response.status_code}] Failed to fetch for {address}")
        return []

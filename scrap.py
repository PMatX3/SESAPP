import requests
import pandas as pd
import time
from requests.exceptions import ChunkedEncodingError, ConnectionError

def fetch_and_store_data(sort='DESC', sort_field='createdOn', page_size=100, max_retries=3, retry_delay=5):
    api_key = 'ALSA866971F4CF55E8E64A4EBD34D218D9F272CA'
    base_url = 'https://api.recruitly.io/api/candidate/list'
    all_data = []
    page_number = 0
    total_pages = None
    keys_to_extract = [
        "title", "gender", "reference", "fullName", "firstName", "surname", "jobTitle",
        "address", "profileImageUrl", "email", "alternateEmail", "mobile",
        "timeZone", "skype", "facebook", "linkedIn", "twitter",
        "dateOfBirth", "overview", "languages", "employer", "skills", "rating",
        "nationalities"
    ]
    address_keys_to_extract = ["cityName", "country", "countryCode","postCode"]

    while total_pages is None or page_number < total_pages:
        url = f"{base_url}?apiKey={api_key}&pageNumber={page_number}&pageSize={page_size}&sort={sort}&sortField={sort_field}"
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                data = response.json()
                if 'data' in data:
                    filtered_data = [
                        {**{k: item.get(k, None) for k in keys_to_extract},
                         "address": {ak: item['address'].get(ak, None) for ak in address_keys_to_extract if 'address' in item and item['address']}
                        } for item in data['data']
                    ]
                    all_data.extend(filtered_data)
                    total_count = data.get('totalCount', 0)
                    total_pages = (total_count + page_size - 1) // page_size
                    page_number += 1
                    print(f"Page {page_number}/{total_pages} processed.")
                    break  # If successful, break out of the retry loop
                else:
                    print("No data found in response.")
                    break
            except (ChunkedEncodingError, ConnectionError, requests.exceptions.RequestException) as e:
                retries += 1
                print(f"Error fetching page {page_number + 1}: {e}. Retrying in {retry_delay} seconds (attempt {retries}/{max_retries})...")
                time.sleep(retry_delay)
        else:
            print(f"Failed to fetch page {page_number + 1} after {max_retries} retries. Skipping to the next page.")
            page_number += 1  # Move to the next page to avoid getting stuck

    df = pd.DataFrame(all_data)
    df.to_json('compant_data/SES.json', orient='records')
    print("Filtered data has been saved to 'SES.json'.")

if __name__ == "__main__":
    fetch_and_store_data()
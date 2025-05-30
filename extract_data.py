import requests
import pandas as pd
import json

def fetch_and_store_data( page_size=100):
    api_key = 'ALSA866971F4CF55E8E64A4EBD34D218D9F272CA'
    base_url = 'https://api.recruitly.io/api/candidate'
    all_data = []
    page_number = 0
    total_pages = None
    # Define the keys you want to extract
    keys_to_extract = [
        "jobTitle", "gender", "reference", "fullName", "firstName", "surname", "address", "profileImageUrl",
        "email", "alternateEmail", "mobile", "workPhone", "timeZone", "linkedIn", "twitter", "facebook", "skype",
        "createdOn", "dateOfBirth", "employer", "hasCv", "cvId", "preferredLocations", "preferredTitles", "availability", "languages",
        "skills", "nationalities", "status", "educationHistory","overview"
    ]

    # Add specific keys for the address
    address_keys_to_extract = ["cityName", "regionName", "countryName", "country", "countryCode", "postCode"]

    while total_pages is None or page_number <= total_pages: # Changed the condition here
        url = f"{base_url}?apiKey={api_key}&pageNumber={page_number}&pageSize={page_size}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                # Filter each item to only include the keys we're interested in
                filtered_data = [
                    {**{k: item.get(k, None) for k in keys_to_extract},
                     "address": {ak: item['address'].get(ak, None) for ak in address_keys_to_extract if
                                 'address' in item and item['address']}
                     } for item in data['data']
                ]
                all_data.extend(filtered_data)
                total_count = data.get('totalCount', 0)
                total_pages = (total_count + page_size - 1) // page_size  # Calculate total pages needed
                page_number += 1
                print(f"Page {page_number}/{total_pages} processed.")
            else:
                print("No data found in response.")
                break
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    # Convert the list of dictionaries to a DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    df.to_json('compant_data/SES.json', orient='records')
    print("Filtered data has been saved to 'SES.json'.")
    print(f"Total pages: {total_pages}") # Print the total pages
    print(f"Total records: {len(all_data)}") # Print the total number of records

if __name__ == "__main__":
    # Example usage
    fetch_and_store_data()

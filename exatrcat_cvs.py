import requests
import pandas as pd

def fetch_and_store_data(sort='DESC', sort_field='createdOn', page_size=100):
    api_key = 'ALSA866971F4CF55E8E64A4EBD34D218D9F272CA'
    base_url = 'https://api.recruitly.io/api/candidate/list'
    all_data = []
    page_number = 0
    total_pages = None
    # Define the keys you want to extract
    keys_to_extract = [
        "title", "gender", "reference", "fullName", "firstName", "surname", "jobTitle", 
        "address", "profileImageUrl", "type", "email", "alternateEmail", "mobile", 
        "timeZone", "skype", "facebook", "linkedIn", "twitter", "hasInterview", 
        "dateOfBirth", "overview", "languages", "employer", "skills", "rating", 
        "nationalities"
    ]

    while total_pages is None or page_number < total_pages:
        url = f"{base_url}?apiKey={api_key}&pageNumber={page_number}&pageSize={page_size}&sort={sort}&sortField={sort_field}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                # Filter each item to only include the keys we're interested in
                filtered_data = [{k: item.get(k, None) for k in keys_to_extract} for item in data['data']]
                all_data.extend(filtered_data)
                total_count = data.get('totalCount', 0)
                total_pages = (total_count + page_size - 1) // page_size  # Calculate total pages needed
                page_number+=1
                print(f"Page {page_number}/{total_pages} processed.")
            else:
                print("No data found in response.")
                break
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    # Convert the list of dictionaries to a DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    # df.to_csv('candidates_data.csv', index=False)
    # print("Filtered data has been saved to 'candidates_data.csv'.")

    df.to_json('candidates_data.json', orient='records')
    print("Filtered data has been saved to 'candidates_data.json'.")

if __name__ == "__main__":
    # Example usage
    fetch_and_store_data()
import requests
from datetime import datetime
import time
from mongo_connection import get_mongo_client
from pymongo import UpdateOne, MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

def fetch_with_retry(url, params, max_retries=3, initial_delay=30):
    """
    Fetch data from API with retry logic and exponential backoff
    """
    delay = initial_delay
    # Loop for the number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"accept": "application/json"},
                timeout=10
            )
            print(f"Attempt {attempt + 1}: Fetching {url} with params {params}")
            
            if response.status_code == 200:
                print(f"Successfully fetched data from {url}")
                return response.json()
            
            # Handle rate limiting
            if response.status_code in (429, 503):  # Too Many Requests or Service Unavailable
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    print(f"Rate limit hit. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                continue
            print(f"Unexpected status code: {response.status_code}. Response: {response.text}")
            # Handle other errors
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
    
    return None

def compute_total_experience(employment_history):
    total_days = 0
    today = datetime.now()
    
    for job in employment_history:
        start_date_str = job.get("startDate", "")
        if not start_date_str:
            continue
            
        try:
            # Handle the year format issue (00XX -> 20XX)
            parts = start_date_str.split("/")
            if len(parts) == 3:
                day, month, year = parts
                # If year starts with "00", replace with "20"
                if year.startswith("00"):
                    year = "20" + year[2:]
                corrected_start_date = f"{day}/{month}/{year}"
                sd = datetime.strptime(corrected_start_date, "%d/%m/%Y")
            else:
                sd = datetime.strptime(start_date_str, "%d/%m/%Y")
        except (ValueError, IndexError):
            continue
        
        # Handle end date
        end_date_str = job.get("endDate")
        if not end_date_str:
            # If no end date, assume current job
            ed = today
        else:
            try:
                # Apply same year correction to end date
                parts = end_date_str.split("/")
                if len(parts) == 3:
                    day, month, year = parts
                    if year.startswith("00"):
                        year = "20" + year[2:]
                    corrected_end_date = f"{day}/{month}/{year}"
                    ed = datetime.strptime(corrected_end_date, "%d/%m/%Y")
                else:
                    ed = datetime.strptime(end_date_str, "%d/%m/%Y")
            except (ValueError, IndexError):
                ed = today
        
        # Calculate experience for this job
        if ed > sd:
            total_days += (ed - sd).days
    
    # Convert to years and round to 2 decimal places
    return round(total_days / 365.25, 2)

# Load environment variables
MONGO_USERNAME = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')

# Connect to MongoDB
mongo_uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.m2e1jl3.mongodb.net/user_db"
m_client = MongoClient(mongo_uri)
db = m_client["user_db"]
collection = db["test_parsed"]

# Settings
BATCH_SIZE = 500
updates = []
count = 0

API_KEY = "ALSA866971F4CF55E8E64A4EBD34D218D9F272CA"
BASE_URL = "https://api.recruitly.io/api/candidate"

cursor = collection.find(
    { "totalYearOfExperience": { "$gt": 10 } },
    { "_id": 1, "candidateId": 1 }
)

count = collection.count_documents({ "totalYearOfExperience": { "$gt": 10 } })
print(count)

for doc in cursor:
    try:
        if doc.get("candidateId"):
            try:
                # Fetch data from API
                api_data = fetch_with_retry(
                    f"{BASE_URL}/{doc['candidateId']}", 
                    params={"apiKey": API_KEY}
                )
                
                if api_data and "employmentHistory" in api_data:
                    # Get employment history from API
                    employment_history = api_data["employmentHistory"]
                    
                    # Calculate experience based on API employment history
                    total_exp = compute_total_experience(employment_history)
                    
                    print(f"Processing document {doc['_id']}: Found {len(employment_history)} jobs, total experience: {total_exp} years")
                else:
                    print(f"No employment history found in API for candidateId {doc['candidateId']}")
                    continue
            except Exception as api_error:
                print(f"API error for candidateId {doc['candidateId']}: {str(api_error)}")
                continue
        else:
            print(f"Skipping document {doc['_id']}: No candidateId found")
            continue
            
        # Update both employment history and total experience
        
        updates.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {
                    "employmentHistory": employment_history,
                    "totalYearOfExperience": total_exp,
                    "lastUpdated": datetime.now()
                }}
            )
        )

        if len(updates) >= BATCH_SIZE:
            collection.bulk_write(updates)
            count += len(updates)
            print(f"Updated {count} docs...")
            updates = []
            
    except Exception as e:
        print(f"Error processing document {doc['_id']}: {str(e)}")
        continue

# Final batch
if updates:
    collection.bulk_write(updates)
    count += len(updates)
    print(f"Updated total: {count} docs.")

m_client.close()

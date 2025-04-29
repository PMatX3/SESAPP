import requests
import pandas as pd
import schedule
import time, json
from extract_data import fetch_and_store_data
from cromadbTest import load_json_data

def job():
    print("Running scheduled job...")
    fetch_and_store_data()

# Schedule the job every day
schedule.every().day.at("00:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)  # wait one minute
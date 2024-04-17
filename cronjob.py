import requests
import pandas as pd
import schedule
import time
from exatrcat_cvs import fetch_and_store_data

def job():
    print("Running scheduled job...")
    fetch_and_store_data()

# Schedule the job every day
schedule.every().day.at("18:20").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)  # wait one minute
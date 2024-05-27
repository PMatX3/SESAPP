from pymongo import MongoClient
import os

def get_mongo_client():
    MONGO_USERNAME = os.getenv('MONGO_USER')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
    mongo_uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.m2e1jl3.mongodb.net/user_db"
    client = MongoClient(mongo_uri)
    return client

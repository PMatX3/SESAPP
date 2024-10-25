from flask import Flask, request, jsonify
from pymongo import MongoClient
import openai
from typing import Dict, Any

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['your_database_name']
collection = db['SES_data']

# OpenAI API configuration
openai.api_key = 'your_openai_api_key'

def query_mongodb(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query MongoDB based on the given criteria.
    """
    return collection.find_one(query)

def generate_response(user_query: str, mongodb_data: Dict[str, Any]) -> str:
    """
    Generate a response using OpenAI API based on the user query and MongoDB data.
    """
    prompt = f"User query: {user_query}\n\nRelevant data: {mongodb_data}\n\nResponse:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided data."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']
    
    # Extract relevant information from the user query to form a MongoDB query
    # This is a simple example; you may need more sophisticated parsing
    query = {}
    if 'name' in user_query.lower():
        query['fullName'] = {'$regex': user_query.split('name')[-1].strip(), '$options': 'i'}
    elif 'email' in user_query.lower():
        query['email'] = {'$regex': user_query.split('email')[-1].strip(), '$options': 'i'}
    
    # Query MongoDB
    mongodb_data = query_mongodb(query)
    
    if mongodb_data:
        # Generate response using OpenAI API
        response = generate_response(user_query, mongodb_data)
    else:
        response = "I'm sorry, I couldn't find any relevant information in the database."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

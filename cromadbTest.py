import pandas as pd
from dotenv import load_dotenv
import os
import openai
import chromadb
from chromadb.utils import embedding_functions
import os
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json
from pymongo import MongoClient

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

m_client = MongoClient('localhost', 27017)

def text_embedding(text):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]

client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

db = m_client['user_db']

collection = client.get_or_create_collection("candidates",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
collection2 = client.get_or_create_collection("candidates2",embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
job_query = client.get_or_create_collection("job_query",embedding_function=openai_ef)
chat_history_collection = db['chat_history']

def add_chat_message(user_id, message, response, chat_id):
    """
    Adds a chat message and its response to the MongoDB collection, including chat_id.
    """
    timestamp = datetime.now().isoformat()
    document = {
        "user_id": user_id,
        "chat_id": chat_id,
        "message": message,
        "response": response,
        "timestamp": timestamp
    }
    chat_history_collection.insert_one(document)
    print('Data saved to MongoDB')

def get_chat_history(user_id, chat_id=None):
    """
    Retrieves the chat history for a given user from MongoDB.
    If chat_id is provided, it filters by that specific chat_id.
    """
    query = {"user_id": user_id}
    if chat_id:
        query["chat_id"] = chat_id
    history = list(chat_history_collection.find(query).sort("timestamp", -1))
    return history

def get_chat_list(user_id):
    """
    Retrieves a list of chats for a given user, showing the first question of each chat,
    sorted by the timestamp of the first message in each chat to ensure consistent ordering.
    """
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"timestamp": 1}},
        {"$group": {
            "_id": "$chat_id", 
            "first_question": {"$first": "$message"},
            "first_timestamp": {"$first": "$timestamp"}  # Capture the timestamp of the first message
        }},
        {"$sort": {"first_timestamp": -1}},  # Ensure consistent ordering based on the first message's timestamp
        {"$project": {"chat_id": "$_id", "first_question": 1, "_id": 0}}
    ]
    return list(chat_history_collection.aggregate(pipeline))

def load_pdf_data(text):
    
    # Add code here to load the extracted text into the collection
    job_query.add(
        documents=[text],
        # metadatas=["job_profiles"],
        ids=["job_profile"]
    )

def load_data(file_name, temp=False):
    df=pd.read_csv(file_name)
    df.head()

    # df['text'] = (
    #     'Candidate ID: ' + df['Candidate Id'].astype(str) + '\n' +
    #     'Employer Name: ' + df['Employer Name'].astype(str) + '\n' +
    #     'Start Date: ' + df['Start Date'].astype(str) + '\n' +
    #     'Country of birth: ' + df['Country of birth'].astype(str) + '\n' +
    #     'Marketing Emails: ' + df['Marketing Emails'].astype(str) + '\n' +
    #     'Current Job Title: ' + df['Current Job Title'].astype(str) + '\n' +
    #     'First Name: ' + df['First Name'].astype(str) + '\n' +
    #     'Gender: ' + df['Gender'].astype(str) + '\n' +
    #     'Email: ' + df['Email'].astype(str) + '\n' +
    #     'Home Phone: ' + df['Home Phone'].astype(str) + '\n' +
    #     'Candidate Owner: ' + df['Candidate Owner'].astype(str) + '\n' +
    #     'Status: ' + df['Status'].astype(str) + '\n' +
    #     'Optout SMS: ' + df['Optout SMS'].astype(str) + '\n' +
    #     'Internal Note: ' + df['Internal Note'].astype(str) + '\n' +
    #     'Address Country: ' + df['Address Country'].astype(str) + '\n' +
    #     'Expected Salary: ' + df['Expected Salary'].astype(str) + '\n' +
    #     'Old Candidate ID: ' + df['Old Candidate ID'].astype(str) + '\n' +
    #     'Current Company: ' + df['Current Company'].astype(str) + '\n' +
    #     'Expected Max Salary: ' + df['Expected Max Salary'].astype(str) + '\n' +
    #     'Preferred Sectors: ' + df['Preferred Sectors'].astype(str) + '\n' +
    #     'Address County/Region: ' + df['Address County/Region'].astype(str) + '\n' +
    #     'Last Contacted: ' + df['Last Contacted'].astype(str) + '\n' +
    #     'Preferred Job Titles: ' + df['Preferred Job Titles'].astype(str) + '\n' +
    #     'LinkedIn: ' + df['LinkedIn'].astype(str) + '\n' +
    #     'National Insurance Number: ' + df['National Insurance Number'].astype(str) + '\n' +
    #     'Candidate Skills: ' + df['Candidate Skills'].astype(str) + '\n' +
    #     'Education Level: ' + df['Education Level'].astype(str) + '\n' +
    #     'Current Salary: ' + df['Current Salary'].astype(str) + '\n' +
    #     'Driving License: ' + df['Driving License'].astype(str) + '\n' +
    #     'Job Types: ' + df['Job Types'].astype(str) + '\n' +
    #     'Rating: ' + df['Rating'].astype(str) + '\n' +
    #     'Preferences: ' + df['Preferences'].astype(str) + '\n' +
    #     'Address Line 1: ' + df['Address Line 1'].astype(str) + '\n' +
    #     'Created On: ' + df['Created On'].astype(str) + '\n' +
    #     'Position 1: ' + df['Position 1'].astype(str) + '\n' +
    #     'Position 2: ' + df['Position 2'].astype(str) + '\n' +
    #     'Current City: ' + df['Current City'].astype(str) + '\n' +
    #     'Date of Birth: ' + df['Date of Birth'].astype(str) + '\n' +
    #     "Father's Name: " + df["Father's Name"].astype(str) + '\n' +
    #     'Annual Leave Days: ' + df['Annual Leave Days'].astype(str) + '\n' +
    #     'Job Title/Headline: ' + df['Job Title/Headline'].astype(str) + '\n' +
    #     'Position 3: ' + df['Position 3'].astype(str) + '\n' +
    #     'Position 4: ' + df['Position 4'].astype(str) + '\n' +
    #     'Nationality: ' + df['Nationality'].astype(str) + '\n' +
    #     'Expected Min Salary: ' + df['Expected Min Salary'].astype(str) + '\n' +
    #     'Address - PIN/Postcode: ' + df['Address - PIN/Postcode'].astype(str) + '\n' +
    #     'Overview: ' + df['Overview'].astype(str) + '\n' +
    #     'Current Country: ' + df['Current Country'].astype(str) + '\n' +
    #     'Tags: ' + df['Tags'].astype(str) + '\n' +
    #     'City of Birth: ' + df['City of Birth'].astype(str) + '\n' +
    #     'Candidate Category: ' + df['Candidate Category'].astype(str) + '\n' +
    #     'End Date: ' + df['End Date'].astype(str) + '\n' +
    #     'Current JobType: ' + df['Current JobType'].astype(str) + '\n' +
    #     'Available From: ' + df['Available From'].astype(str) + '\n' +
    #     'Full Name: ' + df['Full Name'].astype(str) + '\n' +
    #     'Gender.1: ' + df['Gender.1'].astype(str) + '\n' +
    #     'Modified On: ' + df['Modified On'].astype(str) + '\n' +
    #     'Date of Birth.1: ' + df['Date of Birth.1'].astype(str) + '\n' +
    #     'Work Phone: ' + df['Work Phone'].astype(str) + '\n' +
    #     'Address Line 2: ' + df['Address Line 2'].astype(str) + '\n' +
    #     'Surname: ' + df['Surname'].astype(str) + '\n' +
    #     'Alternate Email Address: ' + df['Alternate Email Address'].astype(str) + '\n' +
    #     'University Degree: ' + df['University Degree'].astype(str) + '\n' +
    #     'Nationality.1: ' + df['Nationality.1'].astype(str) + '\n' +
    #     'Relocate: ' + df['Relocate'].astype(str) + '\n' +
    #     'Current Job Title.1: ' + df['Current Job Title.1'].astype(str) + '\n' +
    #     'Marketing SMS: ' + df['Marketing SMS'].astype(str) + '\n' +
    #     'Address City: ' + df['Address City'].astype(str) + '\n' +
    #     'Availability: ' + df['Availability'].astype(str) + '\n' +
    #     'Current Salary.1: ' + df['Current Salary.1'].astype(str) + '\n' +
    #     'Reason For Leaving: ' + df['Reason For Leaving'].astype(str) + '\n' +
    #     'Preferred Industries: ' + df['Preferred Industries'].astype(str) + '\n' +
    #     'Title: ' + df['Title'].astype(str) + '\n' +
    #     'Sick Leave Days: ' + df['Sick Leave Days'].astype(str) + '\n' +
    #     'Conversation Thread: ' + df['Conversation Thread'].astype(str) + '\n' +
    #     'Twitter: ' + df['Twitter'].astype(str) + '\n' +
    #     'Mobile: ' + df['Mobile'].astype(str)
    # )    

    df['text'] = df.apply(lambda row: '\n'.join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    
    docs=df["text"].tolist() 
    ids= [str(x) for x in df.index.tolist()]
    # Define maximum batch size
    max_batch_size = 20
    
    # Splitting the documents and ids into batches and adding them to the collection
    for i in range(0, len(docs), max_batch_size):
        batch_docs = docs[i:i + max_batch_size]
        batch_ids = ids[i:i + max_batch_size]
        if temp:
            collection2.add(
                documents=batch_docs,
                ids=batch_ids
            )
        else:
            collection.add(
                documents=batch_docs,
                ids=batch_ids
            )


def load_json_data(json_data, file=False):
    """
    Converts JSON data to a DataFrame, then generates a new DataFrame with a 'text' column
    that concatenates all column names and data, and finally stores this data into Cromadb.
    Before storing into Cromadb, it saves the DataFrame as a CSV file in the 'csvdata' folder with the name 'json_data.csv'.

    Args:
    - json_data: A list of dictionaries, where each dictionary represents a document to be stored in Cromadb.
    """
    # Convert JSON data to DataFrame
    df = pd.DataFrame(json_data)

    # Save the DataFrame to a CSV file in the 'csvdata' folder
    csv_file_path = 'csvdata/json_data.csv'
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Ensure the directory exists
    df.to_csv(csv_file_path, index=False, escapechar='\\')
    print(f"JSON data saved to {csv_file_path} successfully.")
    
    if not file:
        load_data(csv_file_path, temp=True)
    else:
        load_data(csv_file_path, temp=False)
    print("JSON data loaded into Cromadb successfully.")

def execute_query(query, user_id, temp=False):
    job_desc = job_query.get('job_profile')
    
    if job_desc['documents'] != []:
        embeding_query = ''.join(job_desc['documents'])
    else:
        embeding_query = 'give me top 3 candidates'
    
    if temp:
        vector = text_embedding(embeding_query)
        results = collection2.query(    
            query_embeddings=vector,
            n_results=5000,
            include=["documents"]
        )
    else:
        vector = text_embedding(embeding_query)
        results = collection.query(    
            query_embeddings=vector,
            n_results=5000,
            include=["documents"]
        )
        
    available_tokens_for_results = 128000 - len(query) - 200  # Subtracting an estimated length for static text in the prompt

    # Convert results to string and truncate if necessary
    results_str = "\n".join(str(item) for item in results['documents'][0])
    if len(results_str) > available_tokens_for_results:
        results_str = results_str[:available_tokens_for_results]  # Truncate results to fit within token limits

    prompt = f'```{results_str}```Based on the data in ```, answer {query}'

    print(prompt)
    messages = [
        # {"role": "system", "content": "You answer questions BestCandidate AI Bot. You will always answer in structured format and in markdown format and please don't use markdown word in response"},
        {"role": "system", "content": "Welcome to BestCandidate AI Bot! I am here to answer your questions in a structured format. Please note that I will always respond in Markdown format. Let's get started!"},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
    except openai.error.InvalidRequestError as e:
        response_message = "Error: The input is too long for the model to process."+e
    
    return response_message

def cromadb_test(file_name,query):    
    df=pd.read_csv(file_name)
    df.head()
    

    df['text'] = (
        'Candidate ID: ' + df['Candidate Id'].astype(str) + '\n' +
        'Employer Name: ' + df['Employer Name'].astype(str) + '\n' +
        'Start Date: ' + df['Start Date'].astype(str) + '\n' +
        'Country of birth: ' + df['Country of birth'].astype(str) + '\n' +
        'Marketing Emails: ' + df['Marketing Emails'].astype(str) + '\n' +
        'Current Job Title: ' + df['Current Job Title'].astype(str) + '\n' +
        'First Name: ' + df['First Name'].astype(str) + '\n' +
        'Gender: ' + df['Gender'].astype(str) + '\n' +
        'Email: ' + df['Email'].astype(str) + '\n' +
        'Home Phone: ' + df['Home Phone'].astype(str) + '\n' +
        'Candidate Owner: ' + df['Candidate Owner'].astype(str) + '\n' +
        'Status: ' + df['Status'].astype(str) + '\n' +
        'Optout SMS: ' + df['Optout SMS'].astype(str) + '\n' +
        'Internal Note: ' + df['Internal Note'].astype(str) + '\n' +
        'Address Country: ' + df['Address Country'].astype(str) + '\n' +
        'Expected Salary: ' + df['Expected Salary'].astype(str) + '\n' +
        'Old Candidate ID: ' + df['Old Candidate ID'].astype(str) + '\n' +
        'Current Company: ' + df['Current Company'].astype(str) + '\n' +
        'Expected Max Salary: ' + df['Expected Max Salary'].astype(str) + '\n' +
        'Preferred Sectors: ' + df['Preferred Sectors'].astype(str) + '\n' +
        'Address County/Region: ' + df['Address County/Region'].astype(str) + '\n' +
        'Last Contacted: ' + df['Last Contacted'].astype(str) + '\n' +
        'Preferred Job Titles: ' + df['Preferred Job Titles'].astype(str) + '\n' +
        'LinkedIn: ' + df['LinkedIn'].astype(str) + '\n' +
        'National Insurance Number: ' + df['National Insurance Number'].astype(str) + '\n' +
        'Candidate Skills: ' + df['Candidate Skills'].astype(str) + '\n' +
        'Education Level: ' + df['Education Level'].astype(str) + '\n' +
        'Current Salary: ' + df['Current Salary'].astype(str) + '\n' +
        'Driving License: ' + df['Driving License'].astype(str) + '\n' +
        'Job Types: ' + df['Job Types'].astype(str) + '\n' +
        'Rating: ' + df['Rating'].astype(str) + '\n' +
        'Preferences: ' + df['Preferences'].astype(str) + '\n' +
        'Address Line 1: ' + df['Address Line 1'].astype(str) + '\n' +
        'Created On: ' + df['Created On'].astype(str) + '\n' +
        'Position 1: ' + df['Position 1'].astype(str) + '\n' +
        'Position 2: ' + df['Position 2'].astype(str) + '\n' +
        'Current City: ' + df['Current City'].astype(str) + '\n' +
        'Date of Birth: ' + df['Date of Birth'].astype(str) + '\n' +
        "Father's Name: " + df["Father's Name"].astype(str) + '\n' +
        'Annual Leave Days: ' + df['Annual Leave Days'].astype(str) + '\n' +
        'Job Title/Headline: ' + df['Job Title/Headline'].astype(str) + '\n' +
        'Position 3: ' + df['Position 3'].astype(str) + '\n' +
        'Position 4: ' + df['Position 4'].astype(str) + '\n' +
        'Nationality: ' + df['Nationality'].astype(str) + '\n' +
        'Expected Min Salary: ' + df['Expected Min Salary'].astype(str) + '\n' +
        'Address - PIN/Postcode: ' + df['Address - PIN/Postcode'].astype(str) + '\n' +
        'Overview: ' + df['Overview'].astype(str) + '\n' +
        'Current Country: ' + df['Current Country'].astype(str) + '\n' +
        'Tags: ' + df['Tags'].astype(str) + '\n' +
        'City of Birth: ' + df['City of Birth'].astype(str) + '\n' +
        'Candidate Category: ' + df['Candidate Category'].astype(str) + '\n' +
        'End Date: ' + df['End Date'].astype(str) + '\n' +
        'Current JobType: ' + df['Current JobType'].astype(str) + '\n' +
        'Available From: ' + df['Available From'].astype(str) + '\n' +
        'Full Name: ' + df['Full Name'].astype(str) + '\n' +
        'Gender.1: ' + df['Gender.1'].astype(str) + '\n' +
        'Modified On: ' + df['Modified On'].astype(str) + '\n' +
        'Date of Birth.1: ' + df['Date of Birth.1'].astype(str) + '\n' +
        'Work Phone: ' + df['Work Phone'].astype(str) + '\n' +
        'Address Line 2: ' + df['Address Line 2'].astype(str) + '\n' +
        'Surname: ' + df['Surname'].astype(str) + '\n' +
        'Alternate Email Address: ' + df['Alternate Email Address'].astype(str) + '\n' +
        'University Degree: ' + df['University Degree'].astype(str) + '\n' +
        'Nationality.1: ' + df['Nationality.1'].astype(str) + '\n' +
        'Relocate: ' + df['Relocate'].astype(str) + '\n' +
        'Current Job Title.1: ' + df['Current Job Title.1'].astype(str) + '\n' +
        'Marketing SMS: ' + df['Marketing SMS'].astype(str) + '\n' +
        'Address City: ' + df['Address City'].astype(str) + '\n' +
        'Availability: ' + df['Availability'].astype(str) + '\n' +
        'Current Salary.1: ' + df['Current Salary.1'].astype(str) + '\n' +
        'Reason For Leaving: ' + df['Reason For Leaving'].astype(str) + '\n' +
        'Preferred Industries: ' + df['Preferred Industries'].astype(str) + '\n' +
        'Title: ' + df['Title'].astype(str) + '\n' +
        'Sick Leave Days: ' + df['Sick Leave Days'].astype(str) + '\n' +
        'Conversation Thread: ' + df['Conversation Thread'].astype(str) + '\n' +
        'Twitter: ' + df['Twitter'].astype(str) + '\n' +
        'Mobile: ' + df['Mobile'].astype(str)
    )

    
    def text_embedding(text):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )


    client = chromadb.Client()
    collection = client.get_or_create_collection("candidates",embedding_function=openai_ef)

    docs=df["text"].tolist() 
    ids= [str(x) for x in df.index.tolist()]


    collection.add(
        documents=docs,
        ids=ids
    )
    vector=text_embedding("give me top 5 candidate list")
    results=collection.query(    
        query_embeddings=vector,
        n_results=15,
        include=["documents"]
    )
    res = "\n".join(str(item) for item in results['documents'][0])    
    prompt=f'```{res}```Based on the data in ```, answer {query}'
    messages = [
            {"role": "system", "content": "You answer questions about 95th Oscar awards."},
            {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]    
    return response_message
